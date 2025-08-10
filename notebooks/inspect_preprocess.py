from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from src.preprocess import preprocess_drawn_image

img = Image.new("L", (200, 200), 255)
d = ImageDraw.Draw(img)
d.line((50, 150, 100, 50), width=18, fill=0)
d.ellipse((120, 60, 160, 100), outline=0, width=12)

out = preprocess_drawn_image(img, out_size=28, pad_ratio=0.2)

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("raw")
plt.subplot(1,2,2)
plt.imshow(out, cmap="gray")
plt.axis("off")
plt.title("preprocessed")
plt.tight_layout()
plt.show()
