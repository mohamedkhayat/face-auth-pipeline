"""Build the README training-schedule figure by calling the real schedule code in src/.

  schedule-{light,dark}.svg  triplet margin (utils.margin_schedule) and cosine-loss weight per epoch

Run:  .venv/bin/python assets/readme/build_assets.py
"""

import sys
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import config  # noqa: E402
from utils import margin_schedule  # noqa: E402

FONTS = HERE / ".fonts"
FONT_URLS = {
    "Inter-400.ttf": "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuLyfMZg.ttf",
    "Inter-600.ttf": "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuGKYMZg.ttf",
}
THEMES = {
    "light": {"bg": "#ffffff", "ink": "#1f2328", "muted": "#59636e", "grid": "#d8dee4", "blue": "#2a78d6", "orange": "#eb6834"},
    "dark": {"bg": "#0d1117", "ink": "#f0f6fc", "muted": "#9198a1", "grid": "#262c36", "blue": "#3987e5", "orange": "#e8692f"},
}


def ensure_fonts():
    FONTS.mkdir(exist_ok=True)
    for name, url in FONT_URLS.items():
        if not (FONTS / name).exists():
            urllib.request.urlretrieve(url, FONTS / name)
        font_manager.fontManager.addfont(str(FONTS / name))
    plt.rcParams.update({"svg.fonttype": "path", "font.family": "Inter", "font.size": 10, "svg.hashsalt": "face-auth",
                         "axes.spines.top": False, "axes.spines.right": False, "xtick.major.size": 0, "ytick.major.size": 0})


def schedule_chart(theme):
    t = THEMES[theme]
    epochs = list(range(config.EPOCHS))
    # loss.hybrid_triplet_loss: margin from margin_schedule(epoch); cosine term only when epoch > COSINE_START_EPOCH
    margin = [margin_schedule(e) for e in epochs]
    cos_w = [config.COSINE_WEIGHT if e > config.COSINE_START_EPOCH else 0.0 for e in epochs]

    fig, ax = plt.subplots(figsize=(8.2, 3.3), dpi=100)
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["bg"])
    ax.plot(epochs, margin, color=t["blue"], lw=2.2)
    ax.step(epochs, cos_w, where="post", color=t["orange"], lw=2.2)
    ax.text(config.MARGIN_PEAK_EPOCH + 1, margin[config.MARGIN_PEAK_EPOCH] + 0.04,
            f"triplet margin  {config.INITIAL_MARGIN} → {config.PEAK_MARGIN} (epoch {config.MARGIN_PEAK_EPOCH}) "
            f"→ {config.FINAL_MARGIN} (epoch {config.N_EPOCHS_MARGIN})", color=t["blue"], fontsize=9.5)
    ax.text(config.COSINE_START_EPOCH + 1, config.COSINE_WEIGHT + 0.03,
            f"cosine-similarity loss weight {config.COSINE_WEIGHT} after epoch {config.COSINE_START_EPOCH}",
            color=t["orange"], fontsize=9.5)
    ax.axvline(config.EARLY_STOP_WARMUP, color=t["muted"], lw=0.8, ls=":")
    ax.text(config.EARLY_STOP_WARMUP + 0.5, max(margin) + 0.06, f"early stopping can fire from epoch {config.EARLY_STOP_WARMUP}",
            color=t["muted"], fontsize=8.5, va="center")
    if config.EPOCHS > config.N_EPOCHS_MARGIN:  # margin_schedule does not clamp, so it keeps falling past FINAL_MARGIN
        ax.text(config.EPOCHS - 1.5, margin[-1] - 0.08, f"no clamp: {margin[-1]:.2f} at epoch {config.EPOCHS - 1}",
                color=t["blue"], fontsize=8.5, ha="right")
    ax.set(xlim=(0, config.EPOCHS - 1), ylim=(0, max(margin) + 0.15), xlabel="epoch")
    ax.xaxis.label.set_color(t["muted"])
    ax.tick_params(colors=t["muted"])
    ax.grid(color=t["grid"], lw=0.6)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["grid"])
    fig.text(0.015, 0.93, "Training schedule of the hybrid triplet loss (from src/config.py)", color=t["ink"], fontsize=11.5,
             weight=600)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.16)
    fig.savefig(HERE / f"schedule-{theme}.svg", facecolor=t["bg"])
    fig.savefig(HERE / f"schedule-{theme}.png", facecolor=t["bg"], dpi=110)  # preview only, git-ignored
    plt.close(fig)


if __name__ == "__main__":
    ensure_fonts()
    for th in THEMES:
        schedule_chart(th)
