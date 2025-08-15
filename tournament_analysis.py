# tournament_analysis.py
# Tournament analysis of international football matches (improved filters + safer ops)
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Visual theme
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

def load_results(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p.resolve()}")
    return pd.read_csv(p, parse_dates=['date'])

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

def analyze_tournament_performance(df: pd.DataFrame, outdir: Path | None, show: bool):
    # More robust tournament name matching
    pattern = r"(FIFA World Cup|UEFA Euro|Copa Am[eé]rica|AFC Asian Cup|Africa[n]? Cup of Nations)"
    tournament_matches = df[df['tournament'].str.contains(pattern, case=False, na=False)].copy()

    if tournament_matches.empty:
        print("No matches found for the selected tournaments.")
        return pd.DataFrame()

    tournament_stats = (
        tournament_matches
        .assign(goals=lambda d: d['home_score'] + d['away_score'])
        .groupby('tournament')
        .agg(matches=('date', 'count'),
             home_mean=('home_score', 'mean'),
             away_mean=('away_score', 'mean'),
             start=('date', 'min'),
             end=('date', 'max'))
        .round(2)
    )
    tournament_stats['goals_per_game'] = (tournament_stats['home_mean'] + tournament_stats['away_mean']).round(2)

    plt.figure(figsize=(12, 6))
    tournament_stats['goals_per_game'].sort_values(ascending=False).plot(kind='bar', color='#ff9800')
    plt.title('Goals per Game in Major Tournaments')
    plt.xlabel('Tournament'); plt.ylabel('Average Goals per Game')
    plt.xticks(rotation=45, ha='right'); plt.grid(True)

    save_or_show(outdir / 'tournament_goals_per_game.png' if outdir else None, show)
    return tournament_stats

def analyze_tournament_winners(outdir: Path | None, show: bool):
    """
    NOTE: This is reference data (not derived from results.csv).
    If you want this fully data-driven, you’ll need a winners dataset.
    """
    tournament_winners = {
        'FIFA World Cup': {
            'Brazil': 5, 'Germany': 4, 'Italy': 4, 'Argentina': 3,
            'France': 2, 'Uruguay': 2, 'England': 1, 'Spain': 1
        },
        'UEFA Euro': {
            'Germany': 3, 'Spain': 3, 'France': 2, 'Italy': 2,
            'Netherlands': 1, 'Denmark': 1, 'Greece': 1, 'Portugal': 1
        }
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    pd.Series(tournament_winners['FIFA World Cup']).sort_values(ascending=False).plot(kind='bar', ax=ax1, color='#ff9800')
    ax1.set_title('FIFA World Cup Titles'); ax1.set_ylabel('Number of Titles'); ax1.tick_params(axis='x', rotation=45)

    pd.Series(tournament_winners['UEFA Euro']).sort_values(ascending=False).plot(kind='bar', ax=ax2, color='#fb8c00')
    ax2.set_title('UEFA Euro Titles'); ax2.set_ylabel('Number of Titles'); ax2.tick_params(axis='x', rotation=45)

    save_or_show(outdir / 'tournament_winners.png' if outdir else None, show)
    return tournament_winners

def analyze_tournament_evolution(df: pd.DataFrame, outdir: Path | None, show: bool):
    tournament_matches = df[df['tournament'].notna()].copy()
    tournament_matches = tournament_matches.assign(
        decade=(tournament_matches['date'].dt.year // 10) * 10
    )
    decade_stats = (
        tournament_matches
        .groupby(['decade', 'tournament'])
        .agg(matches=('date', 'count'),
             home_mean=('home_score', 'mean'),
             away_mean=('away_score', 'mean'))
        .round(2)
    )
    decade_stats['goals_per_game'] = (decade_stats['home_mean'] + decade_stats['away_mean']).round(2)

    plt.figure(figsize=(12, 6))
    for t in ['FIFA World Cup', 'UEFA Euro']:
        if (t in decade_stats.index.get_level_values(1)):
            series = decade_stats.xs(t, level=1)['goals_per_game'].sort_index()
            plt.plot(series.index, series.values, marker='o', label=t)

    plt.title('Evolution of Goals per Game in Major Tournaments')
    plt.xlabel('Decade'); plt.ylabel('Goals per Game'); plt.legend(); plt.grid(True)
    save_or_show(outdir / 'tournament_evolution.png' if outdir else None, show)
    return decade_stats

def main():
    ap = argparse.ArgumentParser(description="Tournament analysis for international football results.")
    ap.add_argument('--csv', default='Data/results.csv', help='Path to results.csv')
    ap.add_argument('--outdir', default='reports/tournaments', help='Directory to save figures')
    ap.add_argument('--show', action='store_true', help='Show plots instead of saving')
    args = ap.parse_args()

    outdir = Path(args.outdir) if not args.show else None
    df = load_results(args.csv)

    print("\n=== Tournament Performance Analysis ===")
    tournament_stats = analyze_tournament_performance(df, outdir, args.show)
    print(tournament_stats)

    print("\n=== Tournament Winners (reference) ===")
    winners_stats = analyze_tournament_winners(outdir, args.show)
    for t, winners in winners_stats.items():
        print(f"\n{t}:")
        for team, titles in sorted(winners.items(), key=lambda x: x[1], reverse=True):
            print(f"{team}: {titles} titles")

    print("\n=== Tournament Evolution Analysis ===")
    evolution_stats = analyze_tournament_evolution(df, outdir, args.show)
    print(evolution_stats)

if __name__ == "__main__":
    main()
