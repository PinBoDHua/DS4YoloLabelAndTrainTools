"""检查 GUI 截图的主题颜色分布。"""
import os
from pathlib import Path

from PIL import Image
import numpy as np

d = Path(__file__).parent / "_shots"
for f in sorted(d.glob("*.png")):
    img = np.asarray(Image.open(f).resize((80, 60))).astype(float)
    avg = img.mean(axis=(0, 1))
    brightness = img.mean()
    print(f"{f.name}: size={Image.open(f).size}, avg_rgb=({avg[0]:.0f},{avg[1]:.0f},{avg[2]:.0f}), "
          f"brightness={brightness:.0f} ({'浅色' if brightness > 120 else '深色'})")
