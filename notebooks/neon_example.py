import sys
sys.path.insert(0, "/home/gus/Documents/REPOS/pyxations")

from pathlib import Path
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyxations as pyx
from pyxations.formats.neon.parse import NeonParse
from pyxations.export import FEATHER_EXPORT

RECORDING_FOLDER    = Path("/tmp/neon_data/2026-03-27-15-12-46")
SCREEN_WIDTH        = 1600
SCREEN_HEIGHT       = 1200
DETECTION_ALGORITHM = "neon"

OUTPUT_FOLDER = Path("/tmp/neon_derivatives")
OUTPUT_FOLDER.mkdir(exist_ok=True)

# ── Parse ─────────────────────────────────────────────────────────────────────
print("Parsing recording...")
NeonParse(OUTPUT_FOLDER, FEATHER_EXPORT).parse(
    RECORDING_FOLDER, detection_algorithm=DETECTION_ALGORITHM, overwrite=True
)

# ── Load ──────────────────────────────────────────────────────────────────────
events_folder = OUTPUT_FOLDER / f"{DETECTION_ALGORITHM}_events"

samples   = pl.read_ipc(OUTPUT_FOLDER / "samples.feather")
fixations = pl.read_ipc(events_folder / "fix.feather")
saccades  = pl.read_ipc(events_folder / "sacc.feather")
blinks    = pl.read_ipc(events_folder / "blink.feather")

print(f"Samples:   {len(samples):>6}  ({samples['tSample'].max() / 1000:.1f} s)")
print(f"Fixations: {len(fixations):>6}  median duration {fixations['duration'].median():.0f} ms")
print(f"Saccades:  {len(saccades):>6}  median amplitude {saccades['ampDeg'].median():.1f}°")
print(f"Blinks:    {len(blinks):>6}")

vis = pyx.Visualization(OUTPUT_FOLDER, DETECTION_ALGORITHM)
plots_folder = OUTPUT_FOLDER / "plots"
plots_folder.mkdir(exist_ok=True)

# ── Scanpath ──────────────────────────────────────────────────────────────────
print("Plotting scanpath...")
vis.scanpath(
    fixations,
    screen_height=SCREEN_HEIGHT,
    screen_width=SCREEN_WIDTH,
    saccades=saccades,
    samples=samples,
    folder_path=plots_folder,
    display=False,
)

# ── Fixation duration ─────────────────────────────────────────────────────────
print("Plotting fixation duration...")
vis.fix_duration(fixations)
plt.savefig(plots_folder / "fix_duration.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Saccade amplitude ─────────────────────────────────────────────────────────
print("Plotting saccade amplitude...")
vis.sacc_amplitude(saccades)
plt.savefig(plots_folder / "sacc_amplitude.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Main sequence ─────────────────────────────────────────────────────────────
print("Plotting main sequence...")
vis.sacc_main_sequence(saccades)
plt.savefig(plots_folder / "main_sequence.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Saccade direction ─────────────────────────────────────────────────────────
print("Plotting saccade direction...")
vis.sacc_direction(saccades)
plt.savefig(plots_folder / "sacc_direction.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Multipanel ────────────────────────────────────────────────────────────────
print("Plotting multipanel...")
vis.plot_multipanel(fixations, saccades)
plt.savefig(plots_folder / "multipanel.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Raw gaze over time ────────────────────────────────────────────────────────
print("Plotting raw gaze...")
df = samples.to_pandas()
blinks_pd = blinks.to_pandas()

t = df["tSample"].to_numpy() / 1000
fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
axes[0].plot(t, df["X"].to_numpy(), lw=0.5, color="steelblue")
axes[0].set_ylabel("X (px)")
axes[0].set_ylim(0, SCREEN_WIDTH)
axes[1].plot(t, df["Y"].to_numpy(), lw=0.5, color="tomato")
axes[1].set_ylabel("Y (px)")
axes[1].set_ylim(0, SCREEN_HEIGHT)
axes[1].set_xlabel("Time (s)")
for _, b in blinks_pd.iterrows():
    for ax in axes:
        ax.axvspan(b["tStart"] / 1000, b["tEnd"] / 1000, alpha=0.2, color="gray")
axes[0].set_title("Raw gaze signal (gray = blinks)")
plt.tight_layout()
plt.savefig(plots_folder / "gaze_over_time.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Pupil diameter ────────────────────────────────────────────────────────────
print("Plotting pupil...")
fig, ax = plt.subplots(figsize=(14, 3))
ax.plot(t, df["Pupil"].to_numpy(), lw=0.6, color="mediumpurple")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Pupil diameter (mm)")
ax.set_title("Mean pupil diameter (L+R average)")
plt.tight_layout()
plt.savefig(plots_folder / "pupil.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nDone. Plots saved to {plots_folder}")
for p in sorted(plots_folder.iterdir()):
    print(f"  {p.name}")
