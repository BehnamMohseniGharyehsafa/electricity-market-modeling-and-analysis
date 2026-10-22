"""Publication-style figures generated directly from the saved real-data experiment."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .data import TARGET, ZONES

NAVY, TEAL, BLUE, ORANGE = "#10283f", "#00a88f", "#4477cc", "#ed9045"
COLORS = [TEAL, BLUE, ORANGE, "#a35cb5", "#e46477", "#6b8e56", "#8a7968", "#5a9db0"]


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": NAVY,
            "text.color": NAVY,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.edgecolor": "#cbd5df",
            "grid.color": "#e7edf2",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "electricity-market-lab",
        }
    )


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=170, bbox_inches="tight")
    svg = directory / f"{name}.svg"
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n", encoding="utf-8"
    )
    plt.close(fig)


def box(ax, x, y, width, height, title, detail, color=TEAL):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.015",
            linewidth=1,
            edgecolor=color,
            facecolor="#f3f8fb",
        )
    )
    ax.text(
        x + 0.02,
        y + height - min(0.045, height * 0.22),
        title,
        fontsize=12,
        weight="bold",
        va="top",
        color=color,
    )
    ax.text(
        x + 0.02,
        y + height - min(0.105, height * 0.5),
        detail,
        fontsize=9.5 if height < 0.2 else 10,
        va="top",
        linespacing=1.35,
    )


def arrow(ax, a, b):
    ax.add_patch(
        FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, color="#8b9fb0", lw=1.5)
    )


def diagrams(directory):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(
        0.015, 0.95, "From market history to tomorrow's price curve", fontsize=24, weight="bold"
    )
    ax.text(
        0.015,
        0.88,
        "Forecast origin: 04:45 New York time on D-1 | Target: 24 hourly NYC prices on D",
        fontsize=12,
        color="#61758a",
    )
    box(
        ax,
        0.025,
        0.39,
        0.20,
        0.37,
        "01  Official data",
        "NYISO P-2A archives\n2022–2025 • USD/MWh\n11 internal load zones\nZIP checksums + audit",
    )
    box(
        ax,
        0.275,
        0.39,
        0.20,
        0.37,
        "02  Known inputs",
        "7 daily price curves\n168 × 11 prices + DST flag\n"
        "Calendar + US holidays\nNo future actuals",
    )
    box(
        ax,
        0.525,
        0.39,
        0.20,
        0.37,
        "03  Model families",
        "Ridge / trees / XGBoost\nLSTM / DLinear\nPatchTST / iTransformer\nValidation ensemble",
    )
    box(
        ax,
        0.775,
        0.39,
        0.20,
        0.37,
        "04  Outputs",
        "24 hourly point forecasts\nNominal 80% intervals\n"
        "Peak / spike diagnostics\nMAE + paired bootstrap",
    )
    for x in [0.225, 0.475, 0.725]:
        arrow(ax, (x + 0.012, 0.58), (x + 0.038, 0.58))
    for x, title, detail, color in [
        (0.025, "TRAIN", "2022–2023 • fit weights", BLUE),
        (0.275, "VALIDATE", "2024 H1 • select settings", TEAL),
        (0.525, "CALIBRATE", "2024 H2 • interval radii", ORANGE),
        (0.775, "TEST", "2025 • frozen evaluation", "#a35cb5"),
    ]:
        box(ax, x, 0.13, 0.20, 0.16, title, detail, color)
    ax.text(
        0.025,
        0.045,
        "Archives include later corrections. DST targets are excluded; historical DST curves "
        "are normalized.",
        fontsize=10,
        color="#61758a",
    )
    save(fig, directory, "pipeline")

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(
        0.02, 0.95, "Four neural views of the same forecasting problem", fontsize=23, weight="bold"
    )
    rows = [
        (
            0.72,
            "DLinear",
            "NYC history\n168 hourly values",
            "Moving-average split\nseasonal + trend",
            "Two linear projections\n+ calendar correction",
            BLUE,
        ),
        (
            0.51,
            "LSTM",
            "11-zone history\n+ historical DST flag",
            "Recurrent encoder\n48 hidden units",
            "Final hidden state\n+ calendar correction",
            TEAL,
        ),
        (
            0.30,
            "PatchTST",
            "NYC history\ninstance normalization",
            "24-hour patches\nstride 12 + positions",
            "Temporal attention\n+ linear forecast head",
            ORANGE,
        ),
        (
            0.09,
            "iTransformer",
            "11 price channels\ninstance normalization",
            "One token per zone\nfull 168-hour embedding",
            "Cross-zone attention\n+ NYC forecast head",
            "#a35cb5",
        ),
    ]
    for y, name, a, b, c, color in rows:
        ax.text(0.02, y + 0.055, name, color=color, weight="bold", fontsize=14)
        for x, title, detail in [
            (0.17, "INPUT", a),
            (0.40, "REPRESENTATION", b),
            (0.66, "MODEL", c),
        ]:
            box(ax, x, y - 0.015, 0.19 if x != 0.66 else 0.23, 0.15, title, detail, color)
        arrow(ax, (0.365, y + 0.06), (0.395, y + 0.06))
        arrow(ax, (0.595, y + 0.06), (0.655, y + 0.06))
        ax.text(0.925, y + 0.065, "24\nprices", ha="center", va="center", weight="bold")
        arrow(ax, (0.895, y + 0.06), (0.908, y + 0.06))
    save(fig, directory, "neural_architectures")


def generate(root):
    style()
    directory = root / "docs/assets"
    directory.mkdir(parents=True, exist_ok=True)
    diagrams(directory)
    metrics = pd.read_csv(root / "reports/metrics.csv")
    info = json.loads((root / "reports/run.json").read_text())
    with np.load(root / "data/processed/nyiso_daily.npz") as data:
        curves, dates = data["curves"], pd.DatetimeIndex(data["dates"])
    with np.load(root / "artifacts/backtest.npz") as data:
        backtest = {k: data[k] for k in data.files}
    name = info["validation_selected_model"]
    mask = backtest["test_mask"]
    test_dates = pd.DatetimeIndex(backtest["dates"][mask])
    y, pred = backtest["actual"][mask], backtest[name][mask]
    lower, upper = backtest["lower"][mask], backtest["upper"][mask]

    fig = plt.figure(figsize=(16, 6), facecolor=NAVY)
    ax = fig.add_axes([0.55, 0.22, 0.40, 0.58], facecolor=NAVY)
    mean = y.mean(0)
    ax.plot(range(24), mean, color="#4fe0c4", lw=3, label="2025 mean actual curve")
    ax.plot(range(24), pred.mean(0), color="#efbd7e", lw=2, ls="--", label=f"Mean {name} forecast")
    ax.set(xlabel="Delivery hour • America/New_York", ylabel="USD/MWh", xticks=[0, 6, 12, 18, 23])
    ax.tick_params(colors="#dce7ef")
    ax.xaxis.label.set_color("#dce7ef")
    ax.yaxis.label.set_color("#dce7ef")
    ax.legend(facecolor=NAVY, edgecolor="none", labelcolor="white", fontsize=9, loc="upper left")
    ax.grid(alpha=0.12)
    fig.text(
        0.045,
        0.79,
        "ELECTRICITY\nMARKET LAB",
        fontsize=34,
        weight="bold",
        color="white",
        linespacing=1.12,
    )
    fig.text(
        0.047,
        0.44,
        "Real US prices. Modern models.\nAuditable forecasting experiments.",
        fontsize=17,
        color="#a9c4d5",
    )
    fig.text(
        0.047, 0.18, "NYISO  /  11 ZONES  /  2022–2025", fontsize=12, weight="bold", color="#4fe0c4"
    )
    # Preserve the dark background for the hero only.
    fig.savefig(directory / "hero.png", dpi=170, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes[0, 0].plot(dates, curves[:, :, TARGET].mean(1), color=BLUE, lw=1)
    axes[0, 0].set(title="NYC daily mean day-ahead price", ylabel="USD/MWh")
    axes[0, 0].grid(axis="y", alpha=0.7)
    table = pd.DataFrame(curves[:, :, TARGET], index=dates)
    monthly = table.loc["2025"].groupby(table.loc["2025"].index.month).mean()
    im = axes[0, 1].imshow(monthly, aspect="auto", cmap="YlGnBu", origin="lower")
    axes[0, 1].set(
        title="2025 price shape by month and hour",
        xlabel="Local delivery hour",
        ylabel="Month",
        yticks=range(12),
        yticklabels=range(1, 13),
    )
    fig.colorbar(im, ax=axes[0, 1], label="USD/MWh", shrink=0.85)
    corr = np.corrcoef(curves.reshape(-1, 11), rowvar=False)
    im = axes[1, 0].imshow(corr, vmin=0.3, vmax=1, cmap="Blues")
    axes[1, 0].set(
        title="Cross-zone price correlation • 2022–2025",
        xticks=range(11),
        yticks=range(11),
        xticklabels=ZONES,
        yticklabels=ZONES,
    )
    axes[1, 0].tick_params(axis="x", rotation=60, labelsize=8)
    axes[1, 0].tick_params(axis="y", labelsize=8)
    fig.colorbar(im, ax=axes[1, 0], shrink=0.85)
    spreads = curves[:, :, TARGET] - curves[:, :, ZONES.index("WEST")]
    axes[1, 1].plot(dates, np.median(spreads, axis=1), color=TEAL, lw=1)
    axes[1, 1].axhline(0, color="#8b9fb0", lw=1)
    axes[1, 1].set(title="NYC–WEST median daily price spread", ylabel="USD/MWh")
    axes[1, 1].grid(axis="y", alpha=0.7)
    for ax in [axes[0, 0], axes[1, 1]]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("The market behind the models", fontsize=23, weight="bold")
    save(fig, directory, "market_landscape")

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    ranked = metrics.sort_values("mae", ascending=False)
    colors = [
        TEAL if n == name else ("#aebfcb" if n.endswith("Naive") else BLUE) for n in ranked.model
    ]
    bars = ax.barh(ranked.model, ranked.mae, color=colors, height=0.68)
    ax.bar_label(bars, fmt="%.2f", padding=5, fontsize=10)
    ax.set(xlabel="Test MAE • USD/MWh • lower is better", xlim=(0, ranked.mae.max() * 1.15))
    ax.set_title("2025 held-out price forecasting", loc="left", fontsize=23, pad=32)
    ax.text(
        0,
        1.025,
        f"{mask.sum():,} eligible days × 24 hours | teal = validation-selected model ({name})",
        transform=ax.transAxes,
        color="#61758a",
    )
    ax.grid(axis="x", alpha=0.7)
    ax.set_axisbelow(True)
    save(fig, directory, "leaderboard")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True, sharex=True)
    for ax, day in zip(axes, ["2025-01-15", "2025-07-15", "2025-12-15"]):
        idx = test_dates.get_loc(day)
        ax.fill_between(
            range(24),
            lower[idx],
            upper[idx],
            color=TEAL,
            alpha=0.15,
            label="Nominal 80% interval",
        )
        ax.plot(y[idx], color=NAVY, lw=2, label="Actual")
        ax.plot(pred[idx], color=TEAL, lw=2, label=name)
        ax.set(title=day, xlabel="Local delivery hour", xticks=[0, 6, 12, 18, 23])
        ax.grid(axis="y", alpha=0.7)
    axes[0].set_ylabel("USD/MWh")
    axes[0].legend(fontsize=8)
    fig.suptitle("Inputs become a full price curve + uncertainty", fontsize=21, weight="bold")
    save(fig, directory, "forecast_examples")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for model, color in zip([name, "WeeklyNaive", "PatchTST", "iTransformer"], COLORS):
        axes[0].plot(np.abs(backtest[model][mask] - y).mean(0), label=model, color=color, lw=2)
    axes[0].set(
        title="Error varies across the delivery day",
        xlabel="Local delivery hour",
        ylabel="MAE • USD/MWh",
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.6)
    cov = metrics.sort_values("coverage")
    axes[1].barh(cov.model, cov.coverage * 100, color=BLUE, height=0.65)
    axes[1].axvline(80, color=ORANGE, ls="--", label="80% nominal")
    axes[1].set(
        title="Measured interval coverage under market shift",
        xlabel="Test hourly coverage • %",
        xlim=(0, 100),
    )
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="x", alpha=0.6)
    fig.suptitle("Accuracy and calibration must both be measured", fontsize=21, weight="bold")
    save(fig, directory, "risk_diagnostics")

    foundation_path = root / "artifacts/foundation.npz"
    if foundation_path.exists():
        with np.load(foundation_path, allow_pickle=False) as data:
            f = {k: data[k] for k in data.files}
        fd = pd.DatetimeIndex(f["dates"])
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
        for ax, day in zip(axes, ["2025-01-15", "2025-07-15", "2025-12-15"]):
            idx = fd.get_loc(day)
            ax.fill_between(
                range(24),
                f["lower"][idx],
                f["upper"][idx],
                color=BLUE,
                alpha=0.15,
                label="Native q10–q90",
            )
            ax.plot(f["actual"][idx], color=NAVY, lw=2, label="Actual")
            ax.plot(f["predictions"][idx], color=BLUE, lw=2, label="Chronos-2 median")
            ax.set(title=day, xlabel="Local delivery hour", xticks=[0, 6, 12, 18, 23])
            ax.grid(axis="y", alpha=0.7)
        axes[0].set_ylabel("USD/MWh")
        axes[0].legend(fontsize=8)
        fig.suptitle(
            "Chronos-2 research track • pretraining overlap unknown", fontsize=20, weight="bold"
        )
        save(fig, directory, "foundation_examples")
    print(f"Saved reproducible diagrams and figures to {directory}")
