from PIL import Image

from pictor.util import images as U


def test_data_url_roundtrip():
    png = U.encode_png(Image.new("RGB", (4, 4), "red"))
    url = U.to_data_url(png)
    data, mime = U.decode_data_url(url)
    assert mime == "image/png" and data == png
    assert U.open_image(data).size == (4, 4)


def test_size_for_ratio_multiples_and_area():
    w, h = U.size_for_ratio("16:9", base=1024, max_side=2048)
    assert w % 32 == 0 and h % 32 == 0
    assert abs(w / h - 16 / 9) < 0.08
    assert 0.9 < (w * h) / (1024 * 1024) < 1.1


def test_clamp_size_respects_max_side():
    assert max(U.clamp_size(4096, 2048, 2048)) <= 2048
    assert U.clamp_size(1000, 1000, 2048) == (992, 992)


def test_parse_size():
    assert U.parse_size("512x256", (1, 1)) == (512, 256)
    assert U.parse_size(None, (7, 7)) == (7, 7)
    w, h = U.parse_size("9:16", (1024, 1024))
    assert h > w


def test_size_like_keeps_ratio():
    w, h = U.size_like(Image.new("RGB", (300, 600)), base=1024, max_side=2048)
    assert abs(w / h - 0.5) < 0.05
