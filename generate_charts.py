import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

output_dir = '/root/models/reports/figures'
import os
os.makedirs(output_dir, exist_ok=True)

# ── Chart 1: GSM8K Q4 vs Q8 (the temperature story) ──
fig, ax = plt.subplots(figsize=(8, 5))
categories = ['Q4_K_M\ntemp=0.6', 'Q8_0\ntemp=0.6', 'Q8_0\ntemp=0.0']
flex = [23.28, 21.38, 84.31]
strict = [19.71, 18.12, 83.24]
x = np.arange(len(categories))
w = 0.35
bars1 = ax.bar(x - w/2, flex, w, label='Flexible-extract', color='#4a90d9', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + w/2, strict, w, label='Strict-match', color='#7b68ee', edgecolor='white', linewidth=0.5)
ax.set_ylabel('Accuracy (%)')
ax.set_title('GSM8K: Temperature Dominates, Not Quantization', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(loc='upper left', framealpha=0.9)
ax.set_ylim(0, 100)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{bar.get_height():.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{bar.get_height():.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{output_dir}/gsm8k_temperature_comparison.png', dpi=150)
plt.close(fig)
print(f'Generated gsm8k_temperature_comparison.png')

# ── Chart 2: IFEval Q4 vs Q8 ──
fig, ax = plt.subplots(figsize=(8, 5))
metrics = ['prompt_level\nstrict', 'prompt_level\nloose', 'inst_level\nstrict', 'inst_level\nloose']
q4_ifeval = [62.0, 66.0, 61.84, 65.79]
q8_ifeval = [66.0, 68.0, 69.74, 71.05]
x = np.arange(len(metrics))
w = 0.35
bars1 = ax.bar(x - w/2, q4_ifeval, w, label='Q4_K_M (temp=0.6)', color='#e8856b', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + w/2, q8_ifeval, w, label='Q8_0 (temp=0.0)', color='#5cb85c', edgecolor='white', linewidth=0.5)
ax.set_ylabel('Accuracy (%)')
ax.set_title('IFEval: Instruction Following Comparison', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc='lower right', framealpha=0.9)
ax.set_ylim(0, 90)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f'{bar.get_height():.1f}%',
            ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f'{bar.get_height():.1f}%',
            ha='center', va='bottom', fontsize=9)
# delta annotations
for i in range(len(metrics)):
    delta = q8_ifeval[i] - q4_ifeval[i]
    mid = (q4_ifeval[i] + q8_ifeval[i]) / 2
    ax.annotate(f'+{delta:.1f}pp', xy=(i, mid), fontsize=9, fontweight='bold',
                ha='center', color='#2d6a2d',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#e8f5e9', edgecolor='none', alpha=0.8))
fig.tight_layout()
fig.savefig(f'{output_dir}/ifeval_comparison.png', dpi=150)
plt.close(fig)
print(f'Generated ifeval_comparison.png')

# ── Chart 3: HumanEval extraction rate ──
fig, ax = plt.subplots(figsize=(7, 5))
quant_labels = ['Q4_K_M', 'Q8_0']
extraction = [21.95, 26.83]
pass1 = [0.0, 0.0]
x = np.arange(len(quant_labels))
w = 0.3
bars_ext = ax.bar(x - w/2, extraction, w, label='Code extraction rate', color='#f0ad4e', edgecolor='white', linewidth=0.5)
bars_pass = ax.bar(x + w/2, pass1, w, label='pass@1', color='#d9534f', edgecolor='white', linewidth=0.5)
ax.set_ylabel('Rate (%)')
ax.set_title('HumanEval: Code Extraction vs Actual Pass Rate', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(quant_labels)
ax.legend(loc='upper right', framealpha=0.9)
ax.set_ylim(0, 45)
for bar in bars_ext:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0, f'{bar.get_height():.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
for bar in bars_pass:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0, f'{bar.get_height():.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.text(0.5, -6, 'pass@1 is 0% for both quantizations — model simply cannot generate executable code',
        ha='center', va='top', fontsize=9, fontstyle='italic', color='#666',
        transform=ax.transData)
fig.tight_layout()
fig.savefig(f'{output_dir}/humaneval_comparison.png', dpi=150)
plt.close(fig)
print(f'Generated humaneval_comparison.png')

# ── Chart 4: All benchmarks side by side (key metrics only) ──
fig, ax = plt.subplots(figsize=(9, 5.5))
benchmarks = ['GSM8K\n(flex-extract)', 'IFEval\n(prompt strict)', 'HumanEval\n(pass@1)']
q4_scores = [23.28, 62.0, 0.0]
q8_scores = [84.31, 66.0, 0.0]
x = np.arange(len(benchmarks))
w = 0.35
bars1 = ax.bar(x - w/2, q4_scores, w, label='Q4_K_M', color='#e8856b', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + w/2, q8_scores, w, label='Q8_0', color='#5cb85c', edgecolor='white', linewidth=0.5)
ax.set_ylabel('Score (%)')
ax.set_title('Qwythos-9B: Q4_K_M vs Q8_0 — All Benchmarks', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(benchmarks)
ax.legend(loc='upper left', framealpha=0.9)
ax.set_ylim(0, 100)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{bar.get_height():.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{bar.get_height():.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
# footnote
ax.text(0.5, -8, 'GSM8K Q4 at temp=0.6, Q8 at temp=0.0  |  IFEval Q4 at temp=0.6, Q8 at temp=0.0  |  HumanEval both at temp=0.0',
        ha='center', va='top', fontsize=8.5, fontstyle='italic', color='#666',
        transform=ax.transData)
fig.tight_layout()
fig.savefig(f'{output_dir}/all_benchmarks_overview.png', dpi=150)
plt.close(fig)
print(f'Generated all_benchmarks_overview.png')

# ── Chart 5: File size comparison ──
fig, ax = plt.subplots(figsize=(6, 4.5))
sizes = [5.24, 8.90]
labels = ['Q4_K_M\n(5.24 GB)', 'Q8_0\n(8.90 GB)']
colors = ['#e8856b', '#5cb85c']
bars = ax.bar(labels, sizes, color=colors, edgecolor='white', linewidth=0.5, width=0.5)
ax.set_ylabel('File size (GB)')
ax.set_title('Model File Size: 70% Larger for Q8_0', fontweight='bold')
ax.set_ylim(0, 11)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{bar.get_height():.1f} GB',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.annotate('+70%', xy=(1, 8.9), xytext=(1.35, 7.0),
            fontsize=14, fontweight='bold', color='#d9534f',
            arrowprops=dict(arrowstyle='->', color='#d9534f', lw=2))
fig.tight_layout()
fig.savefig(f'{output_dir}/file_size_comparison.png', dpi=150)
plt.close(fig)
print(f'Generated file_size_comparison.png')

print('\nAll 5 charts generated.')