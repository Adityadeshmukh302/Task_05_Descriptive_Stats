# time_based_analysis.py
# Time-based analysis of international football matches (refactored)
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Visual theme (orange) ---
mpl.rcParams.update({
    'axes.prop_cycle': mpl.cycler(color=['#ff9800', '#ffa726', '#fb8c00', '#ffb300', '#ff7043', '#ffcc80']),
    'axes.facecolor': '#fff3e0',
    'figure.facecolor': '#fff3e0',
    'axes.edgecolor': '#ff9800',
    'xtick.color': '#fb8c00',
    'ytick.color': '#fb8c00',
    'axes.labelcolor': '#fb8c00',
    'text.color': '#e65100',
    'axes.titleweight': 'bold',
    'axes.titlesize': 16,
    'axes.titlecolor': '#e65100',
    'grid.color': '#ffb300',
    'grid.linestyle': '--',
    'grid.alpha': 0.5
})

# ---------- Helpers ----------
def load_results(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p.resolve()}")
    df = pd.read_csv(p, parse_dates=['date'])
    # Derived columns in one place
    df = df.assign(
        year=lambda d: d['date'].dt.year,
        decade=lambda d: (d['date'].dt.year // 10) * 10,
        month=lambda d: d['date'].dt.month,
        total_goals=lambda d: d['home_score'] + d['away_score'],
        goal_difference=lambda d: d['home_score'] - d['away_score'],
    )
    return df

def save_or_show(figpath: Path | None, show: bool):
    if figpath:
        figpath.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(figpath, dpi=150)
        plt.close()
    elif show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()

# ---------- Analyses ----------
def analyze_decades(df: pd.DataFrame, outdir: Path | None, show: bool):
    decade_stats = (
        df.groupby('decade')
          .agg(home_mean=('home_score', 'mean'),
               home_std=('home_score', 'std'),
               away_mean=('away_score', 'mean'),
               away_std=('away_score', 'std'),
               matches=('date', 'count'))
          .round(2)
    )

    ax = decade_stats[['home_mean', 'away_mean']].plot(kind='bar', figsize=(12, 6), alpha=0.85)
    ax.set_title('Average Goals per Game by Decade')
    ax.set_xlabel('Decade')
    ax.set_ylabel('Average Goals')
    ax.grid(True)

    save_or_show(outdir / 'avg_goals_by_decade.png' if outdir else None, show)
    return decade_stats

def analyze_seasons(df: pd.DataFrame, outdir: Path | None, show: bool):
    monthly_stats = (
        df.groupby('month')
          .agg(home_mean=('home_score', 'mean'),
               away_mean=('away_score', 'mean'),
               matches=('date', 'count'))
          .round(2)
    )

    ax = monthly_stats[['home_mean', 'away_mean']].plot(kind='line', marker='o', figsize=(12, 6))
    ax.set_title('Seasonal Goal Scoring Patterns')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Goals')
    ax.grid(True)

    save_or_show(outdir / 'seasonal_goals.png' if outdir else None, show)
    return monthly_stats

def analyze_tournament_impact(
    df: pd.DataFrame,
    outdir: Path | None,
    show: bool,
    window_days: int = 90,
    tournaments_regex: str = r'FIFA World Cup|UEFA Euro'
):
    """
    Approximate 'before vs after' impact by finding tournament editions:
    - Identify matches whose tournament matches regex.
    - Group by (tournament, year) and use the MIN date as the start of that edition.
    - Compute average goals in windows [start - W, start) vs (start, start + W].
    """
    tdf = df[df['tournament'].str.contains(tournaments_regex, case=False, na=False)].copy()
    if tdf.empty:
        return pd.DataFrame(columns=['tournament', 'year', 'start_date', 'avg_goals_before', 'avg_goals_after', 'diff'])

    editions = (
        tdf.groupby(['tournament', 'year'])
           .agg(start_date=('date', 'min'), end_date=('date', 'max'), matches=('date', 'count'))
           .reset_index()
    )

    W = pd.Timedelta(days=window_days)
    records = []
    for _, row in editions.iterrows():
        start = row['start_date']
        before = df[(df['date'] >= (start - W)) & (df['date'] < start)]
        after  = df[(df['date'] > start) & (df['date'] <= (start + W))]

        records.append({
            'tournament': row['tournament'],
            'year': int(row['year']),
            'start_date': start.date(),
            'avg_goals_before': before[['home_score', 'away_score']].mean().mean() if not before.empty else np.nan,
            'avg_goals_after':  after[['home_score', 'away_score']].mean().mean() if not after.empty else np.nan,
        })

    impact_df = pd.DataFrame(records)
    impact_df['diff'] = impact_df['avg_goals_after'] - impact_df['avg_goals_before']

    ax = impact_df['diff'].dropna().plot(kind='hist', bins=20, color='#ff9800', edgecolor='#e65100', figsize=(12, 6))
    ax.set_title(f'Impact of Major Tournaments on Scoring (±{window_days} days)')
    ax.set_xlabel('Difference in Average Goals (After - Before)')
    ax.set_ylabel('Frequency')
    ax.grid(True)

    save_or_show(outdir / 'tournament_impact_hist.png' if outdir else None, show)
    return impact_df.sort_values(['tournament', 'year'])

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Time-based analysis for international football results.")
    ap.add_argument('--csv', default='Data/results.csv', help='Path to results.csv')
    ap.add_argument('--outdir', default='reports/time_based', help='Directory to save figures')
    ap.add_argument('--show', action='store_true', help='Show plots instead of saving')
    ap.add_argument('--window_days', type=int, default=90, help='Tournament before/after window size (days)')
    ap.add_argument('--tournaments_regex', default=r'FIFA World Cup|UEFA Euro',
                    help='Regex to identify tournaments (case-insensitive)')

    args = ap.parse_args()
    outdir = Path(args.outdir) if not args.show else None

    df = load_results(args.csv)

    print("\n=== Decade by Decade Evolution ===")
    decade_stats = analyze_decades(df, outdir, args.show)
    print(decade_stats)

    print("\n=== Seasonal Patterns ===")
    monthly_stats = analyze_seasons(df, outdir, args.show)
    print(monthly_stats)

    print("\n=== Tournament Impact Analysis ===")
    impact_stats = analyze_tournament_impact(
        df, outdir, args.show, window_days=args.window_days, tournaments_regex=args.tournaments_regex
    )
    print(impact_stats.head(10))
    print("\nAverage impact of tournaments on scoring rates (diff = after - before):")
    print(impact_stats['diff'].describe())

if __name__ == "__main__":
    main()
