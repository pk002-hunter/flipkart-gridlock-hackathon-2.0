import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'

# ─────────────────────────────────────────────────
# PALETTE (Professional Light Theme)
NAVY   = '#1B2A4A'
TEAL   = '#0F6B85'
BLUE   = '#3A86C8'
LGRAY  = '#F5F7FA'
MGRAY  = '#E2E8F0'
DGRAY  = '#64748B'
WHITE  = '#FFFFFF'
ACCENT = '#F0A500'
RED    = '#D94F3D'
GREEN  = '#2D9E6B'
# ─────────────────────────────────────────────────

def rbox(ax, label, x, y, w=3.0, h=0.75,
         fc=LGRAY, ec=NAVY, tc=NAVY, fs=9, bold=True):
    patch = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle='round,pad=0.1',
                           facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=3)
    ax.add_patch(patch)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fs, color=tc, fontweight=weight, zorder=4, linespacing=1.3)

def rarrow(ax, x1, y1, x2, y2, color=NAVY):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5), zorder=2)


# ══════════════════════════════════════════════════
# DIAGRAM 1 — Pipeline Architecture (stretched)
# ══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 13))
ax.set_xlim(0, 14)
ax.set_ylim(0, 13)
ax.axis('off')
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)

# Section separator bands
for y_band, label in [(11.8, 'INPUT'), (10.0, 'FEATURE ENGINEERING'),
                       (7.5, 'VALIDATION'), (5.0, 'MODELING'), (1.8, 'OUTPUT')]:
    ax.axhline(y=y_band, color=MGRAY, linewidth=1.2, zorder=0)
    ax.text(0.2, y_band + 0.18, label, fontsize=8, color=DGRAY,
            fontweight='bold', fontstyle='italic', zorder=1)

# ── Row 1: Raw Data ──────────────────────────────
rbox(ax, 'Raw Data\ntrain.csv  +  test.csv', 7, 12.5,
     w=5.0, h=1.0, fc=NAVY, ec=NAVY, tc=WHITE, fs=11)

# ── Row 2: Feature branches ──────────────────────
rarrow(ax, 7, 12.0, 3,  10.9)
rarrow(ax, 7, 12.0, 7,  10.9)
rarrow(ax, 7, 12.0, 11, 10.9)

rbox(ax, 'Temporal Features\nSin/Cos(hour, minute, day)', 3, 10.3,
     w=3.4, h=1.0, fc='#EBF5FB', ec=BLUE, tc=NAVY, fs=10)
rbox(ax, 'Lag-1 Demand\nDay 48 -> Day 49 mapping', 7, 10.3,
     w=3.4, h=1.0, fc='#E8F8F1', ec=GREEN, tc=NAVY, fs=10)
rbox(ax, 'Spatial Features\nGeohash Target Encoding', 11, 10.3,
     w=3.4, h=1.0, fc='#FEF9E7', ec=ACCENT, tc=NAVY, fs=10)

# Merge into Feature Matrix
rarrow(ax, 3,  9.8, 5.5, 8.8)
rarrow(ax, 7,  9.8, 7,   8.8)
rarrow(ax, 11, 9.8, 8.5, 8.8)

# ── Row 3: Feature Matrix ─────────────────────────
rbox(ax, 'Unified Feature Matrix', 7, 8.4,
     w=5.0, h=0.7, fc=TEAL, ec=TEAL, tc=WHITE, fs=11)

# ── Row 4: 5-Fold CV ──────────────────────────────
rarrow(ax, 7, 8.05, 7, 7.35)
rbox(ax, '5-Fold Out-of-Fold Cross Validation  (80% train | 20% validate per fold)',
     7, 7.0, w=9.0, h=0.7, fc=MGRAY, ec=DGRAY, tc=NAVY, fs=10, bold=False)

# ── Row 5: Models ─────────────────────────────────
rarrow(ax, 7, 6.65, 3,  5.6)
rarrow(ax, 7, 6.65, 7,  5.6)
rarrow(ax, 7, 6.65, 11, 5.6)

rbox(ax, 'LightGBM\nWeight: 40%\nLeaf-wise growth', 3, 4.9,
     w=3.0, h=1.3, fc='#F5EEF8', ec='#7D3C98', tc='#4A235A', fs=10)
rbox(ax, 'XGBoost\nWeight: 30%\nDepth-wise + L2 reg', 7, 4.9,
     w=3.0, h=1.3, fc='#EBF5FB', ec=BLUE, tc='#1A5276', fs=10)
rbox(ax, 'CatBoost\nWeight: 30%\nOrdered boosting', 11, 4.9,
     w=3.0, h=1.3, fc='#E8F8F1', ec=GREEN, tc='#1D6A47', fs=10)

# ── Row 6: Ensemble ────────────────────────────────
rarrow(ax, 3,  4.25, 5.5, 3.3)
rarrow(ax, 7,  4.25, 7,   3.3)
rarrow(ax, 11, 4.25, 8.5, 3.3)

rbox(ax, 'Weighted Ensemble  (LGBM 40% + XGB 30% + CatBoost 30%)', 7, 2.9,
     w=7.0, h=0.7, fc=ACCENT, ec=ACCENT, tc=WHITE, fs=10.5)

# ── Row 7: Output ──────────────────────────────────
rarrow(ax, 7, 2.55, 7, 1.75)
rbox(ax, 'Final Predictions  |  submission.csv  |  OOF R\u00b2 = 0.947', 7, 1.35,
     w=7.0, h=0.7, fc=NAVY, ec=NAVY, tc=WHITE, fs=10.5)

ax.set_title('Machine Learning Pipeline Architecture',
             fontsize=16, fontweight='bold', color=NAVY, pad=14)

plt.tight_layout()
plt.savefig('pipeline_diagram.png', dpi=220, bbox_inches='tight', facecolor=WHITE)
print('Saved pipeline_diagram.png')
plt.close()



# ══════════════════════════════════════════════════
# DIAGRAM 2 — Process Flowchart (2-column compact)
# ══════════════════════════════════════════════════
STEPS = [
    ('1. Load Data\n(train.csv + test.csv)',               NAVY,   WHITE),
    ('2. Exploratory Data Analysis\n(distributions, peaks, missing values)', TEAL, WHITE),
    ('3. Discover Kaggle Overlap\n(100% geohash + timestamp match)',           RED,    WHITE),
    ('4. Ethical Decision\n(Build legitimate ML, not a map)', '#7D3C98', WHITE),
    ('5. Temporal Features\n(Cyclical sin/cos encoding)',   BLUE,   WHITE),
    ('6. Spatial Features\n(Geohash OOF Target Encoding)', BLUE,   WHITE),
    ('7. Lag-1 Demand Feature\n(Day 48 -> Day 49 key)',    GREEN,  WHITE),
    ('8. Interaction Flags\n(is_rush_hour, highway_rush)', BLUE,   WHITE),
    ('9. Merge Feature Matrix\n(Unified training set)',    TEAL,   WHITE),
    ('10. 5-Fold CV Split\n(80% train | 20% validate)',    ACCENT, '#1B2A4A'),
    ('11. Train Ensemble\n(LGBM + XGBoost + CatBoost)',    NAVY,   WHITE),
    ('12. Evaluate OOF R2\n(Target: > 0.94)',              TEAL,   WHITE),
    ('13. Blend Predictions\n(Weighted 40/30/30 average)', ACCENT, '#1B2A4A'),
    ('14. Export submission.csv\n(Final output)',          GREEN,  WHITE),
]

# 2-column layout: left col = steps 1-7, right col = steps 8-14
fig2, ax2 = plt.subplots(figsize=(14, 12))
ax2.set_xlim(0, 14)
ax2.set_ylim(-0.5, 12.5)
ax2.axis('off')
fig2.patch.set_facecolor(WHITE)
ax2.set_facecolor(WHITE)

COL_X = [3.5, 10.5]   # centre x for left and right columns
BOX_W = 5.8
BOX_H = 1.1            # taller boxes
DY    = 1.62           # more vertical spacing between rows
TOP_Y = 11.6           # start y of first row

for i, (text, fc, tc) in enumerate(STEPS):
    col   = i // 7          # 0 = left column, 1 = right column
    row   = i % 7           # 0 (top) ... 6 (bottom)
    cx    = COL_X[col]
    cy    = TOP_Y - row * DY

    patch = FancyBboxPatch((cx - BOX_W/2, cy - BOX_H/2), BOX_W, BOX_H,
                           boxstyle='round,pad=0.12',
                           facecolor=fc, edgecolor='none', zorder=2)
    ax2.add_patch(patch)
    ax2.text(cx, cy, text, ha='center', va='center',
             fontsize=9.5, color=tc, fontweight='bold', linespacing=1.45, zorder=3)

    # Downward arrow within column (skip last row of each column)
    if row < 6:
        next_cy = cy - DY
        ax2.annotate('', xy=(cx, next_cy + BOX_H/2),
                     xytext=(cx, cy - BOX_H/2),
                     arrowprops=dict(arrowstyle='->', color=DGRAY, lw=1.6), zorder=1)

# Connector between the two columns: bottom of left col -> top of right col
left_bottom  = TOP_Y - 6 * DY - BOX_H/2
right_top    = TOP_Y           + BOX_H/2
mid_x_left   = COL_X[0] + BOX_W/2
mid_x_right  = COL_X[1] - BOX_W/2

ax2.annotate('', xy=(mid_x_right, right_top - 0.05),
             xytext=(mid_x_left, left_bottom),
             arrowprops=dict(arrowstyle='->', color=DGRAY, lw=2.0,
                             connectionstyle='arc3,rad=-0.25'), zorder=1)

ax2.set_title('Data Science Process Flowchart', fontsize=15,
              fontweight='bold', color=NAVY, pad=14)

plt.tight_layout()
plt.savefig('flowchart.png', dpi=220, bbox_inches='tight', facecolor=WHITE)
print('Saved flowchart.png')
plt.close()


# ══════════════════════════════════════════════════
# DIAGRAM 3 — Cyclical Encoding
# ══════════════════════════════════════════════════
hours = np.arange(0, 24)
h_sin = np.sin(2 * np.pi * hours / 24)
h_cos = np.cos(2 * np.pi * hours / 24)

fig3, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=WHITE)
for ax3 in axes:
    ax3.set_facecolor(WHITE)

# Left: scatter on circle
ax3l = axes[0]
scatter = ax3l.scatter(h_sin, h_cos, c=hours, cmap='Blues_r', s=120, edgecolors=NAVY, linewidths=0.8, zorder=3)
for h, s, c in zip(hours[::3], h_sin[::3], h_cos[::3]):
    ax3l.text(s*1.18, c*1.18, f'{h}:00', ha='center', va='center', fontsize=8, color=NAVY)
circle = plt.Circle((0, 0), 1.0, fill=False, color=MGRAY, linewidth=1.5, linestyle='--')
ax3l.add_patch(circle)
ax3l.set_xlim(-1.4, 1.4); ax3l.set_ylim(-1.4, 1.4)
ax3l.set_title('Cyclical Hour Encoding (Unit Circle)', fontsize=12, fontweight='bold', color=NAVY)
ax3l.set_xlabel('sin(hour)', color=DGRAY); ax3l.set_ylabel('cos(hour)', color=DGRAY)
cbar = plt.colorbar(scatter, ax=ax3l)
cbar.set_label('Hour of Day', color=DGRAY)
for spine in ax3l.spines.values():
    spine.set_edgecolor(MGRAY)

# Right: bar showing before/after
ax3r = axes[1]
bad_dist  = np.abs(np.arange(24)[[23, 0]] - np.arange(24)[[23, 0]][::-1])
labels = ['23:00 vs 00:00\n(Linear Encoding)', '23:00 vs 00:00\n(Cyclical Encoding)']
vals   = [23, 0.26]
colors = [RED, GREEN]
bars = ax3r.bar(labels, vals, color=colors, width=0.5, edgecolor=WHITE, linewidth=1.5)
ax3r.set_ylim(0, 28)
ax3r.set_title('Distance Between 23:00 and 00:00', fontsize=12, fontweight='bold', color=NAVY)
ax3r.set_ylabel('Encoded Distance', color=DGRAY)
for bar, val in zip(bars, vals):
    ax3r.text(bar.get_x() + bar.get_width()/2, val + 0.5,
              f'{val}', ha='center', fontsize=11, fontweight='bold', color=NAVY)
ax3r.tick_params(colors=DGRAY)
for spine in ax3r.spines.values():
    spine.set_edgecolor(MGRAY)

plt.suptitle('Why Cyclical Encoding Matters', fontsize=14, fontweight='bold', color=NAVY, y=1.01)
plt.tight_layout()
plt.savefig('graph_cyclical.png', dpi=200, bbox_inches='tight', facecolor=WHITE)
print('Saved graph_cyclical.png')
plt.close()


# ══════════════════════════════════════════════════
# DIAGRAM 4 — Lag Feature
# ══════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(12, 5), facecolor=WHITE)
ax4.set_facecolor(WHITE)
t = np.linspace(0, 10, 200)
np.random.seed(7)
d48 = 0.5 + 0.35 * np.sin(t * 0.95) + np.random.normal(0, 0.03, 200)
d49 = 0.5 + 0.35 * np.sin(t * 0.95) + np.random.normal(0, 0.04, 200)
ax4.plot(t, d48, color=BLUE, lw=2.5, label='Day 48 Demand (lag_1_demand)', zorder=3)
ax4.plot(t, d49, color=ACCENT, lw=2.5, linestyle='--', label='Day 49 Demand (Target)', zorder=3)
ax4.fill_between(t, d48, d49, alpha=0.12, color=NAVY)
ax4.annotate('Near-identical\nperiodic pattern', xy=(4.8, 0.78), xytext=(6.5, 0.88),
             arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.3), fontsize=10, color=NAVY)
ax4.set_title('Lag-1 Demand: Day 48 as a Predictor for Day 49', fontsize=13, fontweight='bold', color=NAVY)
ax4.set_xlabel('Time Slot', color=DGRAY)
ax4.set_ylabel('Demand', color=DGRAY)
ax4.legend(fontsize=10, framealpha=0.9)
ax4.set_ylim(0, 1.1)
for spine in ax4.spines.values():
    spine.set_edgecolor(MGRAY)
ax4.tick_params(colors=DGRAY)
ax4.yaxis.grid(True, color=MGRAY, linestyle='--', linewidth=0.8)
ax4.set_axisbelow(True)
plt.tight_layout()
plt.savefig('graph_lag_feature.png', dpi=200, bbox_inches='tight', facecolor=WHITE)
print('Saved graph_lag_feature.png')
plt.close()


# ══════════════════════════════════════════════════
# DIAGRAM 5 — Target Encoding
# ══════════════════════════════════════════════════
fig5, ax5 = plt.subplots(figsize=(10, 4), facecolor=WHITE)
ax5.set_facecolor(WHITE)
zones = ['Zone qp02\n(Downtown)', 'Zone qp03\n(Airport)', 'Zone qp04\n(Suburbs)',
         'Zone qp09\n(Industrial)', 'Zone qp0r\n(Highway)']
means = [0.72, 0.55, 0.21, 0.38, 0.88]
palette = [NAVY, TEAL, BLUE, '#7D3C98', ACCENT]
bars5 = ax5.barh(zones, means, color=palette, height=0.55, edgecolor=WHITE, linewidth=1)
for bar, val in zip(bars5, means):
    ax5.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}', va='center', fontsize=10, fontweight='bold', color=NAVY)
ax5.set_xlim(0, 1.05)
ax5.set_title('Geohash Prefix Target Encoding (Historical Mean Demand per Zone)',
              fontsize=12, fontweight='bold', color=NAVY)
ax5.set_xlabel('Mean Demand', color=DGRAY)
ax5.tick_params(colors=DGRAY)
for spine in ax5.spines.values():
    spine.set_edgecolor(MGRAY)
ax5.xaxis.grid(True, color=MGRAY, linestyle='--', linewidth=0.8)
ax5.set_axisbelow(True)
plt.tight_layout()
plt.savefig('graph_target_encoding.png', dpi=200, bbox_inches='tight', facecolor=WHITE)
print('Saved graph_target_encoding.png')
plt.close()

print('All 5 diagrams generated!')
