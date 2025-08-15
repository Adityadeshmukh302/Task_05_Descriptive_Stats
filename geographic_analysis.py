# geographic_analysis.py
# Geographic analysis of international football matches (symmetric rivalries + improved hosting)
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

# 1. Continental Analysis (quick, manual lists)
def analyze_continental_patterns(df: pd.DataFrame, outdir: Path | None, show: bool):
    continents = {
        'Europe': ['England', 'France', 'Germany', 'Italy', 'Spain', 'Netherlands', 'Portugal', 'Belgium'],
        'South_America': ['Brazil', 'Argentina', 'Uruguay', 'Chile', 'Colombia', 'Paraguay'],
        'North_America': ['Mexico', 'United States', 'Costa Rica', 'Honduras', 'Jamaica'],
        'Africa': ['Nigeria', 'Cameroon', 'Senegal', 'Egypt', 'Morocco', 'Ghana'],
        'Asia': ['Japan', 'South Korea', 'Iran', 'Saudi Arabia', 'Australia'],
    }

    rows = []
    for continent, teams in continents.items():
        # home side calculations
        home = df[df['home_team'].isin(teams)]
        away = df[df['away_team'].isin(teams)]
        matches = len(home) + len(away)
        if matches == 0:
            rows.append({'continent': continent, 'matches': 0, 'goals_scored': 0, 'wins': 0,
                         'win_rate': np.nan, 'goals_per_game': np.nan})
            continue

        goals_scored = home['home_score'].sum() + away['away_score'].sum()
        wins = (home['home_score'] > home['away_score']).sum() + (away['away_score'] > away['home_score']).sum()

        win_rate = (wins / matches * 100) if matches else np.nan
        gpg = (goals_scored / matches) if matches else np.nan
        rows.append({'continent': continent, 'matches': matches, 'goals_scored': goals_scored,
                     'wins': wins, 'win_rate': round(win_rate, 2), 'goals_per_game': round(gpg, 2)})

    stats_df = pd.DataFrame(rows).set_index('continent')

    # Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    stats_df['win_rate'].plot(kind='bar', ax=ax1, color='#ff9800')
    ax1.set_title('Continental Win Rates'); ax1.set_ylabel('Win Rate (%)'); ax1.grid(True)
    stats_df['goals_per_game'].plot(kind='bar', ax=ax2, color='#fb8c00')
    ax2.set_title('Goals per Game by Continent'); ax2.set_ylabel('Goals per Game'); ax2.grid(True)

    save_or_show(outdir / 'continental_performance.png' if outdir else None, show)
    return stats_df

# 2. Regional Rivalries Analysis (symmetric)
def analyze_regional_rivalries(df: pd.DataFrame, outdir: Path | None, show: bool):
    pairs = df[['home_team', 'away_team']].copy()
    pairs['pair'] = pairs.apply(lambda r: tuple(sorted([r['home_team'], r['away_team']])), axis=1)
    rivalry_counts = (
        pairs['pair'].value_counts().reset_index(name='matches').rename(columns={'index': 'pair'})
    )

    top = rivalry_counts.head(10)
    labels = [f"{a} vs {b}" for a, b in top['pair']]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top)), top['matches'], color='#ff9800')
    plt.xticks(range(len(top)), labels, rotation=45, ha='right')
    plt.title('Most Frequent International Rivalries (symmetric)')
    plt.xlabel('Teams'); plt.ylabel('Number of Matches'); plt.grid(True)

    save_or_show(outdir / 'top_rivalries.png' if outdir else None, show)
    return rivalry_counts

# 3. Host Nation Analysis (neutral matches where host didn’t play)
def analyze_host_nations(df: pd.DataFrame, outdir: Path | None, show: bool):
    host_df = df.copy()
    host_df['neutral_nonparticipant'] = (
        (host_df['neutral'] == True) &
        (host_df['country'] != host_df['home_team']) &
        (host_df['country'] != host_df['away_team'])
    )

    host_stats = host_df.groupby('country').agg(
        total_matches=('date','count'),
        neutral_matches=('neutral','sum'),
        neutral_nonparticipant=('neutral_nonparticipant','sum')
    )
    host_stats['share_neutral_nonparticipant_%'] = (
        host_stats['neutral_nonparticipant'] / host_stats['total_matches'] * 100
    ).round(2)

    top_hosts = host_stats.nlargest(10, 'neutral_nonparticipant')

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_hosts)), top_hosts['neutral_nonparticipant'], color='#fb8c00', alpha=0.85,
            label='Neutral (host not playing)')
    plt.bar(range(len(top_hosts)), top_hosts['total_matches'], color='#ffcc80', alpha=0.4,
            label='All matches in country')
    plt.xticks(range(len(top_hosts)), top_hosts.index, rotation=45, ha='right')
    plt.title('Top Neutral Hosts (host nation not participating)')
    plt.xlabel('Country'); plt.ylabel('Matches'); plt.legend(); plt.grid(True)

    save_or_show(outdir / 'neutral_hosting.png' if outdir else None, show)
    return host_stats

def main():
    ap = argparse.ArgumentParser(description="Geographic analysis for international football results.")
    ap.add_argument('--csv', default='Data/results.csv', help='Path to results.csv')
    ap.add_argument('--outdir', default='reports/geographic', help='Directory to save figures')
    ap.add_argument('--show', action='store_true', help='Show plots instead of saving')
    args = ap.parse_args()

    outdir = Path(args.outdir) if not args.show else None
    df = load_results(args.csv)

    print("\n=== Continental Performance Analysis ===")
    continental_stats = analyze_continental_patterns(df, outdir, args.show)
    print(continental_stats)

    print("\n=== Top Regional Rivalries ===")
    rivalry_stats = analyze_regional_rivalries(df, outdir, args.show)
    print(rivalry_stats.head(10))

    print("\n=== Host Nation Analysis ===")
    host_stats = analyze_host_nations(df, outdir, args.show)
    print(host_stats.sort_values('neutral_nonparticipant', ascending=False).head(10))

if __name__ == "__main__":
    main()
