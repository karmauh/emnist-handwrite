from PIL import Image, ImageDraw
from src.preprocess import preprocess_drawn_image

def test_preprocess_returns_28x28():
    img = Image.new("L", (200, 200), 255)
    d = ImageDraw.Draw(img)
    d.line((50, 150, 100, 50), width=18, fill=0)
    out = preprocess_drawn_image(img)
    assert out.size == (28, 28)
