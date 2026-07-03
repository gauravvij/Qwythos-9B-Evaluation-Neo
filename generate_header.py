#!/usr/bin/env /usr/bin/python3
"""Generate a professional blog header image for the Qwythos-9B Evaluation blog post."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = '/root/models/reports/figures'
os.makedirs(output_dir, exist_ok=True)

# Target: 1200x630 pixels at 150 DPI
# Figure size = 1200/150 x 630/150 = 8 x 4.2 inches
fig = plt.figure(figsize=(8, 4.2), dpi=150)
fig.patch.set_facecolor('#0A0E27')

# Build dark gradient background
height_px = 630
width_px = 1200
gradient = np.zeros((height_px, width_px, 3), dtype=np.float32)
for y in range(height_px):
    t = y / height_px
    r = (0x0A / 255.0) * (1 - t) + (0x1A / 255.0) * t
    g = (0x0E / 255.0) * (1 - t) + (0x0A / 255.0) * t
    b = (0x27 / 255.0) * (1 - t) + (0x3E / 255.0) * t
    gradient[y, :, 0] = r
    gradient[y, :, 1] = g
    gradient[y, :, 2] = b

ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(gradient, extent=[0, 8, 0, 4.2], aspect='auto', origin='lower')
ax.set_xlim(0, 8)
ax.set_ylim(0, 4.2)
ax.axis('off')

# Subtle grid lines
for y in np.arange(0.3, 4.2, 0.7):
    ax.axhline(y=y, xmin=0.05, xmax=0.95, color='white', alpha=0.04, linewidth=0.5)
for x in np.arange(0.3, 8, 1.0):
    ax.axvline(x=x, ymin=0.05, ymax=0.95, color='white', alpha=0.03, linewidth=0.5)

# Decorative horizontal accent lines
ax.plot([0.7, 7.3], [3.0, 3.0], color='#E8A838', alpha=0.5, linewidth=1.5)
ax.plot([0.7, 7.3], [2.9, 2.9], color='#0D7C66', alpha=0.4, linewidth=1.0)

# Abstract data bar motif in background
for bx, bh in [(4.5, 2.6), (5.1, 1.4), (5.7, 2.9), (6.3, 1.0), (6.9, 2.0)]:
    ax.bar(bx, bh, width=0.3, color='#3B6077', alpha=0.12, edgecolor='none')

# Main title
ax.text(4.0, 2.7, 'Qwythos-9B Evaluation',
        fontsize=36, fontweight='bold', fontfamily='sans-serif',
        color='white', ha='center', va='center', alpha=0.95)

# Subtitle
ax.text(4.0, 2.1, 'An Autonomous AI Engineering Case Study',
        fontsize=14, fontfamily='sans-serif',
        color='#B0B8D0', ha='center', va='center',
        alpha=0.85, fontstyle='italic')

# Benchmark names row
ax.text(4.0, 1.75, 'GSM8K  \u2022  IFEval  \u2022  HumanEval',
        fontsize=11, fontfamily='sans-serif',
        color='#8890AA', ha='center', va='center', alpha=0.75)

# Subtle divider line above benchmark names
ax.plot([2.0, 6.0], [1.95, 1.95], color='#B0B8D0', alpha=0.12, linewidth=0.5)

# Quantization label
ax.text(7.2, 0.3, 'Q4_K_M  |  Q8_0',
        fontsize=9, fontfamily='sans-serif',
        color='#8890AA', ha='right', va='center', alpha=0.7)

# Accent diamonds
ax.plot(0.9, 3.6, marker='D', markersize=4, color='#0D7C66', alpha=0.4)
ax.plot(1.1, 3.6, marker='D', markersize=4, color='#E8A838', alpha=0.3)

# Bottom subtle line
ax.plot([0.7, 7.3], [0.5, 0.5], color='#B0B8D0', alpha=0.08, linewidth=0.5)

fig.savefig(f'{output_dir}/blog_header.png', dpi=150,
            facecolor='#0A0E27', edgecolor='none',
            bbox_inches='tight', pad_inches=0)
plt.close(fig)

print(f'Generated blog_header.png at {output_dir}/blog_header.png')

# Verify dimensions
from PIL import Image
img = Image.open(f'{output_dir}/blog_header.png')
print(f'Dimensions: {img.width} x {img.height} pixels')
