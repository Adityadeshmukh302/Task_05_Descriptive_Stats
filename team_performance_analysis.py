# team_performance.py
# Team performance analysis of international football matches (faster win%, ordered streaks, safer ops)
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

# 1) Win % (fast)
def calculate_team_stats(df: pd.DataFrame, outdir: Path | None, show: bool, min_games: int = 50):
    home_wins = df[df['home_score'] > df['away_score']]['home_team'].value_counts()
    away_wins = df[df['away_score'] > df['home_score']]['away_team'].value_counts()
    total_wins = home_wins.add(away_wins, fill_value=0)

    total_games = df['home_team'].value_counts().add(df['away_team'].value_counts(), fill_value=0)

    team_stats = pd.DataFrame({
        'total_games': total_games.astype(int),
        'total_wins': total_wins.fillna(0).astype(int)
    })
    team_stats['win_percentage'] = (team_stats['total_wins'] / team_stats['total_games'] * 100).round(2)

    qualified = team_stats[team_stats['total_games'] >= min_games].sort_values('win_percentage', ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(qualified)), qualified['win_percentage'], color='#ff9800')
    plt.xticks(range(len(qualified)), qualified.index, rotation=45, ha='right')
    plt.title(f'Top 10 Teams by Win Percentage (min. {min_games} games)')
    plt.xlabel('Team'); plt.ylabel('Win Percentage'); plt.grid(True)

    save_or_show(outdir / 'top_win_percent.png' if outdir else None, show)
    return team_stats

# 2) Win streaks (chronological)
def analyze_streaks(df: pd.DataFrame, outdir: Path | None, show: bool):
    sorted_df = df.sort_values('date')
    teams = pd.Index(sorted_df['home_team']).append(pd.Index(sorted_df['away_team'])).unique()

    streaks_max = {}
    for team in teams:
        t = sorted_df[(sorted_df['home_team'] == team) | (sorted_df['away_team'] == team)].copy()
        t['won'] = np.where(
            (t['home_team'] == team) & (t['home_score'] > t['away_score']), True,
            np.where((t['away_team'] == team) & (t['away_score'] > t['home_score']), True, False)
        )

        max_streak = 0
        curr = 0
        for w in t['won'].values:
            curr = (curr + 1) if w else 0
            if curr > max_streak:
                max_streak = curr
        if max_streak > 0:
            streaks_max[team] = max_streak

    top = pd.Series(streaks_max).sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top)), top.values, color='#ff9800')
    plt.xticks(range(len(top)), top.index, rotation=45, ha='right')
    plt.title('Top 10 Longest Win Streaks'); plt.xlabel('Team'); plt.ylabel('Consecutive Wins'); plt.grid(True)

    save_or_show(outdir / 'longest_win_streaks.png' if outdir else None, show)
    return pd.Series(streaks_max)

# 3) Home vs Away performance
def analyze_home_away(df: pd.DataFrame, outdir: Path | None, show: bool, min_home_games: int = 20):
    teams = pd.Index(df['home_team']).append(pd.Index(df['away_team'])).unique()
    recs = []
    for team in teams:
        home = df[df['home_team'] == team]
        away = df[df['away_team'] == team]
        recs.append({
            'team': team,
            'home_games': len(home),
            'home_goals_scored': home['home_score'].mean(),
            'home_goals_conceded': home['away_score'].mean(),
            'home_wins_%': (home['home_score'] > home['away_score']).mean() * 100 if len(home) else np.nan,
            'away_games': len(away),
            'away_goals_scored': away['away_score'].mean(),
            'away_goals_conceded': away['home_score'].mean(),
            'away_wins_%': (away['away_score'] > away['home_score']).mean() * 100 if len(away) else np.nan,
        })
    perf = pd.DataFrame(recs)
    perf['home_advantage_%'] = perf['home_wins_%'] - perf['away_wins_%']
    top = perf[perf['home_games'] >= min_home_games].sort_values('home_advantage_%', ascending=False).head(10)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top)), top['home_advantage_%'], color='#ff9800')
    plt.xticks(range(len(top)), top['team'], rotation=45, ha='right')
    plt.title(f'Top 10 Teams with Strongest Home Advantage (≥ {min_home_games} home games)')
    plt.xlabel('Team'); plt.ylabel('Home Advantage (pp)'); plt.grid(True)

    save_or_show(outdir / 'home_advantage.png' if outdir else None, show)
    return perf

def main():
    ap = argparse.ArgumentParser(description="Team performance analysis for international football results.")
    ap.add_argument('--csv', default='Data/results.csv', help='Path to results.csv')
    ap.add_argument('--outdir', default='reports/teams', help='Directory to save figures')
    ap.add_argument('--show', action='store_true', help='Show plots instead of saving')
    ap.add_argument('--min_games', type=int, default=50, help='Min games for win% ranking')
    ap.add_argument('--min_home_games', type=int, default=20, help='Min home games for home-advantage ranking')
    args = ap.parse_args()

    outdir = Path(args.outdir) if not args.show else None
    df = load_results(args.csv)

    print("\n=== Team Performance Statistics ===")
    team_stats = calculate_team_stats(df, outdir, args.show, min_games=args.min_games)
    print(team_stats.sort_values('win_percentage', ascending=False).head(10)[['total_games', 'win_percentage']])

    print("\n=== Win Streak Analysis ===")
    streak_stats = analyze_streaks(df, outdir, args.show)
    print(streak_stats.sort_values(ascending=False).head(10))

    print("\n=== Home vs Away Analysis ===")
    performance_stats = analyze_home_away(df, outdir, args.show, min_home_games=args.min_home_games)
    print(performance_stats[performance_stats['home_games'] >= args.min_home_games]
          .sort_values('home_advantage_%', ascending=False)
          .head(10)[['team', 'home_advantage_%', 'home_games', 'away_games']])

if __name__ == "__main__":
    main()
