
# Task_05_Descriptive_Stats

This repository demonstrates a hybrid approach combining Large Language Models (LLMs) with traditional Python scripting to perform descriptive statistical analysis on an international football dataset. Originally, Research Task 5 suggested using SU Women’s Lacrosse data, but due to access and formatting difficulties, we opted instead for the International Football Results dataset from Kaggle, covering matches from 1872 to 2017.



## Dataset

- **Source**: [Kaggle - International Football Results from 1872 to 2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017/data?select=results.csv)
- **Filename**: `results.csv`
- **Content**: 48,366 match records (1872–2025) with columns:
  - `date` (datetime)
  - `home_team`, `away_team` (categorical)
  - `home_score`, `away_score` (numeric)
  - `tournament` (categorical)
  - `city`, `country` (categorical)
  - `neutral` (boolean)

> **Note**: The raw dataset file is NOT included in this repo (see `.gitignore`).

## Prompts & LLM Analysis

We crafted natural-language prompts to extract insights directly from ChatGPT Plus. Key questions included:

1. **Who is the best team of all time?**

   * *LLM Answer*: Top five teams by win percentage: Jersey (65.52%), Brazil (63.49%), Guernsey (60.42%), Spain (58.73%), Germany (57.79%).

2. **Which teams dominated different eras of football?**

   * *LLM Answer*: Dominant teams per decade (e.g., 1960s: Brazil 67.5%; 1980s: Tahiti 75.8%; 2020s: Argentina 74.2%).

3. **What trends have there been in international football?**

   * *LLM Answer*: Home advantage declined from ~1.5 goals (1870s) to ~0.5 (2020s); average goals/match fell from ~5.6 to ~2.7.

4. **Geopolitical insights from fixtures**

   * *LLM Answer*: Unique teams grew from 2 (1872) to 246 (2023); top rivalries include Argentina vs Uruguay (183 matches), England vs Scotland (118).

5. **Neutral host countries**

   * *LLM Answer*: United States (991), Malaysia (508), Qatar (431), France (413), Thailand (362) hosted most neutral matches.

6. **Hosting advantage in tournaments**

   * *LLM Answer*: Hosts like Jersey and Brittany achieved 100% win rates at home vs ~50–70% otherwise.

7. **Friendlies vs. competitive matches**


   * *LLM Answer*: Top friendly teams (e.g., Germany: 52.9% win friendlies vs 64.5% competitive). All showed better competitive performance.


Additional insights, charts, and tables were generated for deeper analysis.

## Python Validation


A Python script (`scripts/python_descriptive_stats.py`) was created via GitHub Copilot to replicate LLM outputs:

```python
import pandas as pd
import numpy as np

# 1. Load CSV
_df = pd.read_csv('Data/results.csv', parse_dates=['date'])

# 2. Numeric stats
df_numeric = _df.select_dtypes(include=[np.number]).agg(['count','mean','min','max','std'])

# 3. Categorical stats
categorical_cols = [ 'date','home_team','away_team','tournament','city','country','neutral']
cat_summary = {col: (_df[col].nunique(), _df[col].mode()[0], _df[col].value_counts().iloc[0])
               for col in categorical_cols}

# 4. Outliers (min/max >3 std dev)

print(df_numeric)
print(cat_summary)
```


*See full script in `scripts/python_descriptive_stats.py`.*

## Results Comparison

We compared every metric from the LLM audit against script outputs. **All values matched exactly**, confirming the LLM’s accuracy in descriptive statistics.

## Visualizations

Key charts (found in `figures/`):

* **Home Advantage by Decade** (`home_adv.png`)
* **Average Goals per Match by Decade** (`avg_goals.png`)
* **Number of Unique Teams per Year** (`unique_teams.png`)
* **Top Neutral Host Countries** (`neutral_hosts.png`)
* **Friendly vs Competitive Win Rates** (`friendly_comp.png`)
* **Hosting vs Non-Hosting Win Rates** (`host_vs_non.png`)
* **LLM vs Script Win % Comparison** (`llm_vs_script.png`)

## Repository Structure


```
Task_05_Descriptive_Stats/
├── README.md
├── .gitignore
├── Data/
│   └── results.csv (excluded)
├── scripts/
│   └── python_descriptive_stats.py
├── figures/
│   ├── home_adv.png
│   ├── avg_goals.png
│   └── ...
└── report/
    └── LLM_Football_Detailed_Report_Task05.docx
```

## How to Run

1. Clone this repo: `git clone <url>`

2. Ensure `results.csv` is placed in the `Data/` directory.
3. Install dependencies: `pip install pandas numpy matplotlib python-docx`
   - `python-docx` is only required if you want to generate or view the Word report in `report/`.
4. Run analysis script:

   ```bash
   python scripts/python_descriptive_stats.py
   ```
5. View visualizations in `figures/` and open the Word report in `report/`.

## License & Citation

This work is for educational purposes under Syracuse University’s curriculum. Cite as:

> Aditya Deshmukh (2025). Task_05_Descriptive_Stats: Descriptive Statistics and LLM Analysis. GitHub repository.

pip install -r requirements.txt

## Descriptive Statistics Audit Script

This project provides a Python script to perform a full descriptive-statistics audit on a CSV dataset (e.g., `results.csv`).

### Features
- Loads a CSV file into a pandas DataFrame (parsing date columns automatically).
- Computes for each numeric column:
  - Count of non-null values
  - Mean
  - Minimum and maximum values
  - Standard deviation
- Computes for each non-numeric column (including dates and booleans):
  - Number of unique values
  - Most frequent value (mode) and its frequency
- Prints all results in clear, labeled tables.
- Highlights numeric columns with extreme outliers (min/max more than three standard deviations from the mean).

### Requirements
- Python 3.x
- pandas
- numpy

Install requirements with:
```
pip install -r requirements.txt
```

### Usage
1. Place your `results.csv` file in the `Data/` directory.
2. Run the script:
```
python scripts/python_descriptive_stats.py
```
3. Review the printed tables for a summary of your dataset.

### Customization
- If your date column is not automatically detected, specify it in the script's `read_csv` call using `parse_dates=['your_date_column']`.

### Output Example
- Numeric and non-numeric statistics are printed in labeled tables.
- Outlier summary is shown at the end.

---

This script is useful for quickly validating and understanding the structure and distribution of your dataset.
# Task_05_Descriptive_Stats
