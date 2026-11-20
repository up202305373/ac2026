import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "database" / "hockey.db"
OUT = ROOT / "docs" / "figures"

sns.set(style="whitegrid", context="paper", font_scale=1.1)

def read(query):
    with sqlite3.connect(DB) as con:
        return pd.read_sql_query(query, con)

def coverage_by_season():
    df = read("""
        SELECT year, COUNT(*) AS rows FROM Teams GROUP BY year
        UNION ALL
        SELECT year, COUNT(*) FROM Scoring GROUP BY year
        UNION ALL
        SELECT year, COUNT(*) FROM Goalies GROUP BY year
    """)
    # The UNION above loses the table name; rebuild explicitly.
    frames = []
    for table in ("Teams", "Scoring", "Goalies"):
        t = read(f'SELECT year, COUNT(*) AS rows FROM "{table}" GROUP BY year')
        t["table"] = table
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    for table, g in df.groupby("table"):
        ax.plot(g["year"], g["rows"], marker="o", markersize=3, linewidth=1.5, label=table)
    ax.axvline(1995, color="gray", linestyle="--", linewidth=1)
    ax.text(1985, ax.get_ylim()[1] * 0.92, "training start (1995)", color="gray", fontsize=9)
    ax.axvline(2004, color="red", linestyle=":", linewidth=1.5)
    ax.text(2004.3, ax.get_ylim()[1] * 0.75, "2004 lockout", color="red", fontsize=9)
    ax.set(xlabel="Season (starting year)", ylabel="Rows", title="Data coverage per season",
           xlim=(1905, 2010))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "coverage_by_season.png", dpi=150)
    plt.close(fig)

def target_distributions():
    pts = read('SELECT Pts FROM Teams WHERE lgID="NHL" AND year BETWEEN 2000 AND 2009')["Pts"].dropna()
    pim = read('SELECT PIM FROM Scoring WHERE lgID="NHL" AND stint=1 AND year BETWEEN 2000 AND 2009')["PIM"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, values, title, xlabel in (
        (axes[0], pts, "Team points (task a target)", "Pts"),
        (axes[1], pim, "Player penalty minutes (task c target)", "PIM"),
    ):
        ax.hist(values, bins=40, edgecolor="black", color=sns.color_palette("deep")[0])
        ax.axvline(values.mean(), color="red", linestyle="--", label=f"mean {values.mean():.1f}")
        ax.axvline(values.median(), color="green", linestyle="--", label=f"median {values.median():.1f}")
        ax.set(title=title, xlabel=xlabel, ylabel="Count")
        ax.legend()
    fig.suptitle("Target distributions, NHL 2000-2009", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "target_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def era_drift():
    pts = read('SELECT year, AVG(Pts) AS mean_pts FROM Teams WHERE lgID="NHL" AND year BETWEEN 1995 AND 2009 GROUP BY year')
    pim = read('SELECT year, AVG(PIM) AS mean_pim FROM Scoring WHERE lgID="NHL" AND stint=1 AND year BETWEEN 1995 AND 2009 GROUP BY year')

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(pts["year"], pts["mean_pts"], marker="o", markersize=3, color=sns.color_palette("deep")[0])
    axes[0].set(ylabel="Mean Pts per team", title="League-wide trends (1995-2009)")
    axes[1].plot(pim["year"], pim["mean_pim"], marker="o", markersize=3, color=sns.color_palette("deep")[1])
    axes[1].set(xlabel="Season (starting year)", ylabel="Mean PIM per player")
    axes[1].axvline(2004, color="red", linestyle=":", linewidth=1.5)
    fig.tight_layout()
    fig.savefig(OUT / "era_drift.png", dpi=150)
    plt.close(fig)

def pim_by_position():
    df = read('''
        SELECT pos, PIM FROM Scoring
        WHERE lgID="NHL" AND stint=1 AND year BETWEEN 2000 AND 2009 AND PIM IS NOT NULL
    ''')
    order = ["C", "L", "R", "D", "G"]
    df = df[df["pos"].isin(order)]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="pos", y="PIM", order=order, ax=ax, showfliers=False)
    ax.set(xlabel="Listed position", ylabel="PIM",
           title="Penalty minutes by position (NHL 2000-2009, outliers hidden)")
    fig.tight_layout()
    fig.savefig(OUT / "pim_by_position.png", dpi=150)
    plt.close(fig)

def column_availability():
    df = read('''
        SELECT year,
               AVG(CASE WHEN T IS NOT NULL THEN 1.0 ELSE 0 END) AS T,
               AVG(CASE WHEN OTL IS NOT NULL THEN 1.0 ELSE 0 END) AS OTL,
               AVG(CASE WHEN SoW IS NOT NULL THEN 1.0 ELSE 0 END) AS SoW
        FROM Teams WHERE lgID="NHL" AND year BETWEEN 1985 AND 2009 GROUP BY year
    ''')
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for col, label in (("T", "T (ties)"), ("OTL", "OTL"), ("SoW", "SoW (shootout wins)")):
        ax.plot(df["year"], df[col], marker="o", markersize=3, label=label)
    ax.axvline(2004, color="red", linestyle=":", linewidth=1.5)
    ax.set(xlabel="Season (starting year)", ylabel="Fraction of teams with value",
           title="When does each standings column exist?", ylim=(-0.05, 1.05))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "column_availability.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    coverage_by_season()
    target_distributions()
    era_drift()
    pim_by_position()
    column_availability()
    print(f"Figures written to {OUT}")
