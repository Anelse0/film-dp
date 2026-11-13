#!/usr/bin/env python3
"""读成片里的光（film-dp 0.2.0）。需要 ffmpeg / ffprobe 在 PATH 上；不用 PIL、numpy。

用法：
  light_read.py VIDEO [VIDEO ...] [--prompt PROMPT.md] [--at 1.5,7.5,13] [--size 192x108]
                [--frames DIR] [--json]

它不判光好不好，只把每一镜的几个数算出来，放在写下的「光：」旁边，让人（或 Claude）对着
light-writing.md §七 去说"可见 / 部分 / 未执行"。采样点：给了 --prompt 就按 Prompt 里
「镜头N（a-bs）」的中点各抽一帧，并把那一镜的「光：」句摘出来并排；给了 --at 就按秒；都没给
就每 2 秒一帧。给两条以上视频时按同一组采样点并排，算整体档位的差。

每帧的数（在缩小到 192x108 的 RGB 帧上算；全是**画面级**的粗代理，不是脸的——脸上光从哪边来、
眼睛看不看得见，要对着帧看）：
  整体    全帧平均亮度 Y（0–255；Y = 0.2126R + 0.7152G + 0.0722B）；档位差按它算
  亮部    最亮 15% 像素的重心在画面哪里（纵向 偏上 / 居中 / 偏下，横向 偏左 / 居中 / 偏右）；
          玻璃上的城市灯点和手机也算在里面，所以它是"画面里亮的东西在哪"，不是"脸在哪"
  剖面    画面中间竖条从上到下分 8 带的平均亮度：读"亮在哪一带、往下掉得多快"——顶光下人站在画面中间时，
          脸那一带亮、往下一带一带暗下去；躯干被侧前光一起打亮时中间几带一样亮。比"上亮下暗"的标签可靠
  上/下   画面中间竖条上三分之一 ÷ 下三分之一（>1.35 上亮下暗，<0.74 下亮上暗）——画面的，不是身体的：
          人上方是暗天花板、下方是亮手机时它读的是这些
  左/右   画面中间横带左半 ÷ 右半（>1.25 画左亮，<0.80 画右亮）——窗在一侧时读的是窗
  暗部    影子那一带（亮度第 10–35 百分位的像素）的平均 RGB 与颜色标签 蓝黑 / 偏绿 / 棕黑 / 中性黑，
          和最暗 20% 的平均亮度：掉黑(<12) / 深(<30) / 抬黑
  高光    Y > 235 的像素占比；裁切 Y > 250 的占比（"窗外的城任它亮"是小点，在 192 宽的帧上占比很小）
  硬边    相邻像素亮度差 > 40 的占比：清楚的边越多越高——影子边软硬的一个弱代理，受内容影响大
  反差    亮度第 95 百分位 − 第 5 百分位
  档位差  两条视频同一采样点的整体亮度比折成约几档（2.2 × log2，gamma 近似），只用来说"暗一档还是半档"

阈值是在四条夜景成片（examples/example-01-night-glass.md 里的那四条）上定的起点，换场景回头看。
退出码：0 正常；2 用法错误或 ffmpeg 缺失。
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

W, H = 192, 108
SHOT = re.compile(r'^镜头\s*(\d+)\s*[（(]\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*s?\s*[)）]', re.M)
LIGHT = re.compile(r'(?:^|[。；;\s])光[：:]\s*([^\n]*)')
BRIGHT_SHARE = 0.15
DARK_SHARE = 0.20
SHADOW_BAND = (0.10, 0.35)
HIGHLIGHT_Y, CLIP_Y = 235, 250
EDGE_D = 40


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError:
        sys.exit(f'找不到 {cmd[0]}：需要 ffmpeg / ffprobe 在 PATH 上')


def duration(path):
    r = run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)])
    try:
        return float(r.stdout.decode().strip())
    except ValueError:
        return None


def rgb_frame(path, t, w=W, h=H):
    """一帧 RGB 字节（w*h*3），抽不到返回 None。"""
    r = run(['ffmpeg', '-hide_banner', '-v', 'error', '-ss', f'{max(t, 0):.3f}', '-i', str(path),
             '-frames:v', '1', '-vf', f'scale={w}:{h}', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'])
    data = r.stdout
    return data if len(data) == w * h * 3 else None


def save_frame(path, t, target):
    run(['ffmpeg', '-hide_banner', '-v', 'error', '-y', '-ss', f'{max(t, 0):.3f}', '-i', str(path),
         '-frames:v', '1', '-vf', 'scale=320:-2', '-q:v', '4', str(target)])


def luma_rows(data, w=W, h=H):
    rows = []
    for y in range(h):
        base = y * w * 3
        row = []
        for x in range(w):
            i = base + x * 3
            row.append(0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2])
        rows.append(row)
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def percentile(sorted_xs, p):
    if not sorted_xs:
        return 0.0
    k = (len(sorted_xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (k - lo)


def cast_label(r, g, b):
    """影子那一带的颜色：差值按亮度归一，黑得很深时也读得出偏向。"""
    y = max(0.2126 * r + 0.7152 * g + 0.0722 * b, 1.0)
    if (g - b) / y > 0.10 and (g - r) / y > 0.10:
        return '偏绿'
    if (b - r) / y > 0.15:
        return '蓝黑'
    if (r - b) / y > 0.15:
        return '棕黑'
    return '中性黑'


def depth_label(y):
    return '掉黑' if y < 12 else ('深' if y < 30 else '抬黑')


def vertical_label(ratio):
    return '上亮下暗' if ratio > 1.35 else ('下亮上暗' if ratio < 0.74 else '上下均匀')


def horizontal_label(ratio):
    return '画左亮' if ratio > 1.25 else ('画右亮' if ratio < 0.80 else '左右均匀')


def position_label(fy, fx):
    v = '偏上' if fy < 0.42 else ('偏下' if fy > 0.58 else '居中')
    h = '偏左' if fx < 0.42 else ('偏右' if fx > 0.58 else '居中')
    return f'{v}·{h}'


def stats(data, w=W, h=H):
    Y = luma_rows(data, w, h)
    flat = [v for row in Y for v in row]
    n = len(flat)
    overall = mean(flat)
    order = sorted(range(n), key=lambda i: flat[i])
    # 亮部重心
    n_bright = max(int(n * BRIGHT_SHARE), 1)
    bright = order[-n_bright:]
    fy = mean([(i // w) / max(h - 1, 1) for i in bright])
    fx = mean([(i % w) / max(w - 1, 1) for i in bright])
    # 画面上/下、左/右
    x0, x1 = w // 4, 3 * w // 4
    top = [Y[y][x] for y in range(0, h // 3) for x in range(x0, x1)]
    bottom = [Y[y][x] for y in range(2 * h // 3, h) for x in range(x0, x1)]
    vert = mean(top) / max(mean(bottom), 1.0)
    profile = []
    for k in range(8):
        ya, yb = h * k // 8, max(h * (k + 1) // 8, h * k // 8 + 1)
        profile.append(mean([Y[y][x] for y in range(ya, yb) for x in range(x0, x1)]))
    y0, y1 = int(h * 0.15), int(h * 0.85)
    left = [Y[y][x] for y in range(y0, y1) for x in range(0, w // 2)]
    right = [Y[y][x] for y in range(y0, y1) for x in range(w // 2, w)]
    horiz = mean(left) / max(mean(right), 1.0)
    # 暗部：影子带的颜色，最暗 20% 的深度
    a, b = int(n * SHADOW_BAND[0]), max(int(n * SHADOW_BAND[1]), int(n * SHADOW_BAND[0]) + 1)
    band = order[a:b]
    sr = mean([data[i * 3] for i in band])
    sg = mean([data[i * 3 + 1] for i in band])
    sb = mean([data[i * 3 + 2] for i in band])
    dark = order[:max(int(n * DARK_SHARE), 1)]
    dy = mean([flat[i] for i in dark])
    # 高光、硬边、反差
    highlight = sum(1 for v in flat if v > HIGHLIGHT_Y) / n
    clip = sum(1 for v in flat if v > CLIP_Y) / n
    edges = total = 0
    for y in range(h):
        row = Y[y]
        for x in range(w - 1):
            total += 1
            if abs(row[x] - row[x + 1]) > EDGE_D:
                edges += 1
    for y in range(h - 1):
        r0, r1 = Y[y], Y[y + 1]
        for x in range(w):
            total += 1
            if abs(r0[x] - r1[x]) > EDGE_D:
                edges += 1
    srt = sorted(flat)
    contrast = percentile(srt, 0.95) - percentile(srt, 0.05)
    return {
        'overall': overall,
        'bright_y': fy, 'bright_x': fx, 'bright_label': position_label(fy, fx),
        'vertical': vert, 'vertical_label': vertical_label(vert), 'profile': profile,
        'horizontal': horiz, 'horizontal_label': horizontal_label(horiz),
        'shadow_rgb': (sr, sg, sb), 'shadow_cast': cast_label(sr, sg, sb),
        'dark_y': dy, 'dark_depth': depth_label(dy),
        'highlight': highlight, 'clip': clip, 'edges': edges / max(total, 1), 'contrast': contrast,
    }


def parse_shots(text):
    """[(n, a, b, 光句)]：按 Prompt 的「镜头N（a-bs）」段。"""
    found = list(SHOT.finditer(text))
    shots = []
    for i, m in enumerate(found):
        body = text[m.end():found[i + 1].start() if i + 1 < len(found) else len(text)]
        lm = LIGHT.search(body)
        light = lm.group(1).strip() if lm else ''
        shots.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), light))
    return shots


def frange(a, b, step):
    t = a
    while t < b:
        yield round(t, 3)
        t += step


def sample_points(args, dur):
    if args.prompt:
        text = Path(args.prompt).read_text(encoding='utf-8')
        shots = parse_shots(text)
        if not shots:
            sys.exit('Prompt 里没有「镜头N（a-bs）」段')
        return [(f'镜{n}', (a + b) / 2, light) for n, a, b, light in shots]
    if args.at:
        return [(f'{t:g}s', t, '') for t in (float(x) for x in args.at.split(','))]
    if dur is None:
        sys.exit('读不到时长，用 --at 指定采样秒数')
    return [(f'{t:g}s', t, '') for t in frange(1.0, dur, 2.0)]


def stops(ya, yb):
    """两帧整体亮度的档位差（gamma 2.2 近似）；正数 = a 比 b 亮。"""
    if ya <= 0 or yb <= 0:
        return 0.0
    return 2.2 * math.log2(ya / yb)


def fmt_row(label, s, light):
    r, g, b = s['shadow_rgb']
    cells = [label, f"{s['overall']:.0f}", s['bright_label'],
             ' '.join(f"{v:.0f}" for v in s['profile']),
             f"{s['vertical']:.2f} {s['vertical_label']}",
             f"{s['horizontal']:.2f} {s['horizontal_label']}",
             f"{s['shadow_cast']}（{r:.0f},{g:.0f},{b:.0f}）{s['dark_depth']}",
             f"{s['highlight'] * 100:.1f}% / {s['clip'] * 100:.1f}%",
             f"{s['edges'] * 100:.1f}%", f"{s['contrast']:.0f}"]
    if light is not None:
        cells.append(light[:60] + ('…' if len(light) > 60 else '') if light else '（这一镜没有「光：」句）')
    return '| ' + ' | '.join(cells) + ' |'


def main(argv=None):
    p = argparse.ArgumentParser(description='读成片里的光：逐镜几个数，放在写下的「光：」旁边')
    p.add_argument('videos', nargs='+')
    p.add_argument('--prompt', help='Prompt.md：按「镜头N（a-bs）」中点采样，并摘出每镜的「光：」句')
    p.add_argument('--at', help='采样秒数，逗号分隔，如 1.5,7.5,13')
    p.add_argument('--size', default=f'{W}x{H}', help='分析用的缩小尺寸，默认 192x108')
    p.add_argument('--frames', help='把采样帧另存到这个目录（320 宽 jpg），方便对着看')
    p.add_argument('--json', action='store_true')
    args = p.parse_args(argv)
    try:
        w, h = (int(x) for x in args.size.lower().split('x'))
    except ValueError:
        sys.exit('--size 写成 宽x高，如 192x108')
    out = {'videos': []}
    points = None
    for vid in args.videos:
        path = Path(vid)
        if not path.exists():
            sys.exit(f'没有这个文件：{vid}')
        dur = duration(path)
        points = points or sample_points(args, dur)
        rows = []
        for label, t, light in points:
            data = rgb_frame(path, t, w, h)
            s = stats(data, w, h) if data else None
            rows.append({'label': label, 't': t, 'light': light, 'stats': s})
            if args.frames and data:
                d = Path(args.frames) / path.stem
                d.mkdir(parents=True, exist_ok=True)
                save_frame(path, t, d / f'{label}_{t:g}.jpg')
        out['videos'].append({'path': str(path), 'duration': dur, 'rows': rows})

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0

    with_light = bool(args.prompt)
    header = ['采样', '整体 Y', '亮部重心', '纵向剖面（上→下 8 带）', '画面上/下', '画面左/右', '暗部（影子带）', '高光 / 裁切', '硬边', '反差']
    if with_light:
        header.append('写下的「光：」')
    for v in out['videos']:
        title = Path(v['path']).name + (f"（{v['duration']:.2f} s）" if v['duration'] else '')
        print(f'\n## {title}')
        print('| ' + ' | '.join(header) + ' |')
        print('|' + '---|' * len(header))
        for r in v['rows']:
            if r['stats'] is None:
                print(f"| {r['label']} | 抽不到帧 |" + ' |' * (len(header) - 2))
                continue
            print(fmt_row(r['label'], r['stats'], r['light'] if with_light else None))
        ok = [r['stats'] for r in v['rows'] if r['stats']]
        if ok:
            ys = [s['overall'] for s in ok]
            casts = [s['shadow_cast'] for s in ok]
            upper = sum(1 for s in ok if s['bright_y'] < 0.42)
            print(f"clip：整体 Y 均值 {mean(ys):.0f}（{min(ys):.0f}–{max(ys):.0f}）；亮部重心偏上 {upper}/{len(ok)} 帧；"
                  f"影子带多为{max(set(casts), key=casts.count)}；高光均值 {mean([s['highlight'] for s in ok]) * 100:.1f}%")
    if len(out['videos']) > 1:
        base = out['videos'][0]
        print(f"\n## 档位差（相对第一条 {Path(base['path']).name}；负数 = 更暗）")
        for v in out['videos'][1:]:
            pairs = [(rb['label'], stops(rv['stats']['overall'], rb['stats']['overall']))
                     for rb, rv in zip(base['rows'], v['rows']) if rb['stats'] and rv['stats']]
            if pairs:
                per = '，'.join(f'{lab} {d:+.1f}' for lab, d in pairs)
                print(f"- {Path(v['path']).name}：平均 {mean([d for _, d in pairs]):+.1f} 档（{per}）")
    print('\n怎么读：全是画面级的粗代理，当 ±一档看；脸上光从哪边来、眼睛看不看得见要对着帧看；'
          '结论按 light-writing.md §七 写 可见 / 部分 / 未执行，一次成片只记现象。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
