import matplotlib.pyplot as plt
import numpy as np
from google.colab import files
from matplotlib.font_manager import FontProperties

# === 1. Global font and style settings: Times New Roman (including mathtext) ===
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.0,

    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",

    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

# === 2. Color scheme ===
color_ours = '#3B6293'
color_base = '#A43634'

# === 3. Data ===
actions = [
    'Scaling',
    'Additives',
    'Calibration',
    'Rectification',
    'Decomposition',
    'Anchoring'
]
a_labels = [r'$a_1$', r'$a_2$', r'$a_3$', r'$a_4$', r'$a_5$', r'$a_6$']

ccm_freq = [12.5, 34.0, 11.5, 29.0, 8.5, 4.5]
nutri_freq = [52.0, 8.5, 6.0, 14.5, 9.0, 10.0]

y = np.arange(len(actions))
height = 0.35

# === 4. Plot ===
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

rects1 = ax.barh(
    y - height / 2,
    ccm_freq,
    height,
    label='CCM (Ours)',
    color=color_ours,
    alpha=0.9,
    edgecolor='black',
    linewidth=0.8,
    hatch='///',
    zorder=3
)

rects2 = ax.barh(
    y + height / 2,
    nutri_freq,
    height,
    label='Nutrition2.7K',
    color=color_base,
    alpha=0.9,
    edgecolor='black',
    linewidth=0.8,
    hatch='...',
    zorder=3
)

# === 5. Axes and ticks ===
ax.invert_yaxis()
ax.set_yticks(y)

# English tick labels (no bold)
ax.set_yticklabels(actions, color='black', fontweight='normal')
ax.tick_params(axis='y', pad=18)

# X-axis settings
ax.set_xlabel(
    'Selection Frequency (%)',
    fontsize=14,
    fontweight='bold',
    labelpad=10,
    color='black'
)
ax.set_xlim(0, 60)
for label in ax.get_xticklabels():
    label.set_fontweight('normal')
    label.set_color('black')

# Gridlines and spines
ax.xaxis.grid(True, linestyle='--', color='gray', alpha=0.3, zorder=0)
ax.yaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# Legend
font_properties = FontProperties(weight='bold', size=12)
legend = ax.legend(
    loc='lower right',
    frameon=True,
    fancybox=False,
    edgecolor='black',
    framealpha=1,
    borderpad=0.6,
    prop=font_properties
)
for text in legend.get_texts():
    text.set_color('black')

# Value labels
def autolabel_horizontal(rects):
    for rect in rects:
        width = rect.get_width()
        ax.annotate(
            f'{width:.1f}',
            xy=(width, rect.get_y() + rect.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha='left',
            va='center',
            fontsize=13,
            color='black',
            fontweight='bold'
        )

autolabel_horizontal(rects1)
autolabel_horizontal(rects2)

# ============================================================
# Key step: precisely place a_i centered above each English label
# ============================================================
ai_fontsize = 20   # Font size of a_i
gap_px = 6         # Pixel gap between a_i and English labels

# Force canvas rendering to obtain accurate text positions
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

# Get y-axis tick label objects
yticklabels = ax.get_yticklabels()

for lab_math, lab_text in zip(a_labels, yticklabels):
    bbox = lab_text.get_window_extent(renderer=renderer)  # Display coordinates (pixels)

    # Horizontal center and top of the English label
    x_center_disp = (bbox.x0 + bbox.x1) / 2
    y_top_disp = bbox.y1

    # Add a small vertical gap above the English label
    x_disp = x_center_disp
    y_disp = y_top_disp + gap_px

    # Convert display coordinates to axes fraction
    x_axes, y_axes = ax.transAxes.inverted().transform((x_disp, y_disp))

    # Place a_i using axes fraction coordinates
    ax.text(
        x_axes,
        y_axes,
        lab_math,
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=ai_fontsize,
        fontweight='bold',
        color='black'
    )

plt.tight_layout()
plt.savefig('top_tier_style_multiline.pdf', format='pdf', bbox_inches='tight')
plt.show()
files.download('top_tier_style_multiline.pdf')
