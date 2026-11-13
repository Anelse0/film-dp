"""light_read.py 的单元测试：不需要 ffmpeg，直接喂合成的 RGB 帧给 stats()，再测 parse_shots()。
运行：python3 -m unittest discover -s tests  或  python3 tests/test_light_read.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import light_read as lr  # noqa: E402

W, H = 64, 36


def frame(fn):
    """fn(x, y) -> (r, g, b)，拼成 rawvideo rgb24 字节。"""
    buf = bytearray()
    for y in range(H):
        for x in range(W):
            r, g, b = fn(x, y)
            buf += bytes((max(0, min(255, int(r))), max(0, min(255, int(g))), max(0, min(255, int(b)))))
    return bytes(buf)


class TopLight(unittest.TestCase):
    def test_top_bright_reads_upper_centroid_and_vertical_ratio(self):
        # 上亮下暗的渐变：顶光把头肩打亮、躯干掉黑
        data = frame(lambda x, y: (200 - y * 5,) * 3)
        s = lr.stats(data, W, H)
        self.assertEqual(s['vertical_label'], '上亮下暗')
        self.assertLess(s['bright_y'], 0.42)
        self.assertTrue(s['bright_label'].startswith('偏上'))

    def test_bottom_bright_reads_lower(self):
        data = frame(lambda x, y: (20 + y * 5,) * 3)
        s = lr.stats(data, W, H)
        self.assertEqual(s['vertical_label'], '下亮上暗')
        self.assertGreater(s['bright_y'], 0.58)


class SideLight(unittest.TestCase):
    def test_left_bright(self):
        data = frame(lambda x, y: (200 - x * 3,) * 3)
        s = lr.stats(data, W, H)
        self.assertEqual(s['horizontal_label'], '画左亮')
        self.assertLess(s['bright_x'], 0.42)

    def test_right_bright(self):
        data = frame(lambda x, y: (10 + x * 3,) * 3)
        s = lr.stats(data, W, H)
        self.assertEqual(s['horizontal_label'], '画右亮')


class Shadows(unittest.TestCase):
    def test_blue_black_shadows(self):
        # 亮的脸在中间，影子带偏蓝黑
        data = frame(lambda x, y: (180, 170, 160) if 24 <= x < 40 and 10 <= y < 26 else (8, 12, 20))
        s = lr.stats(data, W, H)
        self.assertEqual(s['shadow_cast'], '蓝黑')
        self.assertEqual(s['dark_depth'], '掉黑')

    def test_green_shadows(self):
        data = frame(lambda x, y: (180, 180, 180) if 24 <= x < 40 and 10 <= y < 26 else (14, 24, 12))
        self.assertEqual(lr.stats(data, W, H)['shadow_cast'], '偏绿')

    def test_warm_shadows(self):
        data = frame(lambda x, y: (180, 180, 180) if 24 <= x < 40 and 10 <= y < 26 else (30, 20, 12))
        self.assertEqual(lr.stats(data, W, H)['shadow_cast'], '棕黑')

    def test_lifted_blacks(self):
        data = frame(lambda x, y: (60, 60, 60))
        self.assertEqual(lr.stats(data, W, H)['dark_depth'], '抬黑')


class Highlights(unittest.TestCase):
    def test_clipping_share(self):
        data = frame(lambda x, y: (255, 255, 255) if x < 16 else (40, 40, 40))
        s = lr.stats(data, W, H)
        self.assertAlmostEqual(s['clip'], 0.25, places=2)
        self.assertAlmostEqual(s['highlight'], 0.25, places=2)

    def test_hard_edge_share_counts_strong_steps(self):
        flat = frame(lambda x, y: (100, 100, 100))
        stepped = frame(lambda x, y: (200, 200, 200) if x % 2 else (40, 40, 40))
        self.assertEqual(lr.stats(flat, W, H)['edges'], 0.0)
        self.assertGreater(lr.stats(stepped, W, H)['edges'], 0.4)


class PromptShots(unittest.TestCase):
    TEXT = ('【分镜时间线】\n'
            '镜头1（0-4s）：【中景】他走过来。光：主光是灯带，从上方略偏前照下来。声音：雨。\n'
            '镜头2（4-6s）：【近景】他抬眼。视线：街。他闭着嘴。声音：雨。\n'
            '镜头3（6-9s）：【中近景】靠窗框；光：同镜头1。\n')

    def test_parse_shots_and_light_sentences(self):
        shots = lr.parse_shots(self.TEXT)
        self.assertEqual([(n, a, b) for n, a, b, _ in shots], [(1, 0.0, 4.0), (2, 4.0, 6.0), (3, 6.0, 9.0)])
        self.assertTrue(shots[0][3].startswith('主光是灯带'))
        self.assertEqual(shots[1][3], '')
        self.assertTrue(shots[2][3].startswith('同镜头1'))

    def test_stops(self):
        self.assertAlmostEqual(lr.stops(50, 50), 0.0)
        self.assertLess(lr.stops(25, 50), 0)


if __name__ == '__main__':
    unittest.main()
