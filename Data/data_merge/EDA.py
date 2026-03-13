import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


DATA_FILE = "feature_interpolation_data/processed_Wrocław___wyb._Conrada_Korzeniowskiego.csv"

# Folder na wyniki
OUTPUT_DIR = Path("eda_results")
OUTPUT_DIR.mkdir(exist_ok=True)


STATS_DIR = OUTPUT_DIR / "statistics"
PLOTS_DIR = OUTPUT_DIR / "plots"
STATS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# rozmiary wykresów do inżynierki
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")


print("EKSPLORACJA DANYCH")
print(f"Plik danych: {DATA_FILE}")
print(f"Folder wyników: {OUTPUT_DIR}")





try:
    df = pd.read_csv(DATA_FILE, parse_dates=['datetime'])
    print(f"Wczytano {len(df)} rekordów")
    print(f"Zakres czasowy: {df['datetime'].min()} do {df['datetime'].max()}")
    print(f"Kolumny: {list(df.columns)}")
except Exception as e:
    print(f"Błąd wczytywania: {e}")
    exit(1)

# kolumny do analizy
pollutant_cols = ['pył zawieszony PM10', 'pył zawieszony PM2.5', 'dwutlenek azotu']
meteo_cols = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'wind_direction_10m',
              'pressure_msl', 'precipitation']

# sprawdź które kolumny istnieją
available_cols = [col for col in pollutant_cols + meteo_cols if col in df.columns]
print(f"Znaleziono {len(available_cols)} kolumn do analizy")


stats = df[available_cols].describe().T
stats['variance'] = df[available_cols].var()
stats['skewness'] = df[available_cols].skew()
stats['kurtosis'] = df[available_cols].kurtosis()

stats_file = STATS_DIR / "descriptive_statistics.csv"
stats.to_csv(stats_file)
print(f"Zapisano statystyki: {stats_file}")


latex_file = STATS_DIR / "descriptive_statistics.tex"
with open(latex_file, 'w', encoding='utf-8') as f:
    f.write(stats.round(2).to_latex())


# statystyki podstawowe
print("\nPodstawowe statystyki:")
print(stats[['mean', 'std', 'min', 'max']].round(2))


# histogramy
pollutants_present = [col for col in pollutant_cols if col in df.columns]

if pollutants_present:
    fig, axes = plt.subplots(1, len(pollutants_present), figsize=(15, 4))
    if len(pollutants_present) == 1:
        axes = [axes]

    for idx, col in enumerate(pollutants_present):
        axes[idx].hist(df[col].dropna(), bins=50, color='steelblue',
                       alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel(f'{col} [µg/m³]')
        axes[idx].set_ylabel('Częstość')
        axes[idx].set_title(f'Rozkład {col}')
        axes[idx].grid(alpha=0.3)

        # statystyki
        mean_val = df[col].mean()
        median_val = df[col].median()
        axes[idx].axvline(mean_val, color='red', linestyle='--',
                          linewidth=2, label=f'Średnia: {mean_val:.1f}')
        axes[idx].axvline(median_val, color='green', linestyle='--',
                          linewidth=2, label=f'Mediana: {median_val:.1f}')
        axes[idx].legend()

    plt.tight_layout()
    dist_file = PLOTS_DIR / "pollutants_distributions.png"
    plt.savefig(dist_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"histogramy są tu: {dist_file}")


if pollutants_present:
    fig, ax = plt.subplots(figsize=(10, 6))
    df[pollutants_present].boxplot(ax=ax)
    ax.set_ylabel('Stężenie [µg/m³]')
    ax.set_title('Rozkłady zanieczyszczeń - wykres box')
    ax.grid(alpha=0.3)

    boxplot_file = PLOTS_DIR / "pollutants_boxplot.png"
    plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Zapisano boxplot: {boxplot_file}")


print("\n================ Analiza sezonowości =============")

df['month'] = df['datetime'].dt.month
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek


if pollutants_present:
    monthly_pollutants = df.groupby('month')[pollutants_present].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in pollutants_present:
        ax.plot(monthly_pollutants.index, monthly_pollutants[col],
                marker='o', linewidth=2, markersize=8, label=col)

    ax.set_xlabel('Miesiąc')
    ax.set_ylabel('Średnie stężenie [µg/m³]')
    ax.set_title('Sezonowość zanieczyszczeń - średnie miesięczne')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Sty', 'Lut', 'Mar', 'Kwi', 'Maj', 'Cze',
                        'Lip', 'Sie', 'Wrz', 'Paź', 'Lis', 'Gru'])
    ax.legend()
    ax.grid(alpha=0.3)

    seasonal_poll_file = PLOTS_DIR / "seasonal_pollutants.png"
    plt.savefig(seasonal_poll_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"sezonowość zanieczyszczeń tutaj: {seasonal_poll_file}")

meteo_present = [col for col in meteo_cols if col in df.columns and col != 'wind_direction']

if meteo_present:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, col in enumerate(meteo_present[:4]):
        monthly_data = df.groupby('month')[col].mean()
        axes[idx].plot(monthly_data.index, monthly_data.values,
                       marker='o', linewidth=2, markersize=8, color='coral')
        axes[idx].set_xlabel('Miesiąc')
        axes[idx].set_ylabel(col.replace('_', ' ').title())
        axes[idx].set_title(f'Sezonowość - {col.replace("_", " ").title()}')
        axes[idx].set_xticks(range(1, 13))
        axes[idx].set_xticklabels(['Sty', 'Lut', 'Mar', 'Kwi', 'Maj', 'Cze',
                        'Lip', 'Sie', 'Wrz', 'Paź', 'Lis', 'Gru'])
        axes[idx].grid(alpha=0.3)

    plt.tight_layout()
    seasonal_meteo_file = PLOTS_DIR / "seasonal_meteorology.png"
    plt.savefig(seasonal_meteo_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano sezonowosc pogody: {seasonal_meteo_file}")

if pollutants_present:
    hourly_pollutants = df.groupby('hour')[pollutants_present].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in pollutants_present:
        ax.plot(hourly_pollutants.index, hourly_pollutants[col],
                marker='o', linewidth=2, markersize=6, label=col)

    ax.set_xlabel('Godzina dnia')
    ax.set_ylabel('Średnie stężenie [µg/m³]')
    ax.set_title('Wzorce dobowe zanieczyszczeń')
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(alpha=0.3)

    hourly_file = PLOTS_DIR / "hourly_patterns.png"
    plt.savefig(hourly_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Zapisano wzorce dobowe: {hourly_file}")



print("\n korealcja.")


corr_cols = [col for col in pollutants_present + meteo_present if col in df.columns]

if len(corr_cols) > 1:
    corr_matrix = df[corr_cols].corr()

    corr_file = STATS_DIR / "correlation_matrix.csv"
    corr_matrix.to_csv(corr_file)
    print(f"macierz korelacji: {corr_file}")

    # heatmapa
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Macierz korelacji Pearsona', fontsize=14, pad=20)

    corr_plot_file = PLOTS_DIR / "correlation_matrix.png"
    plt.savefig(corr_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Zapisano heatmap korelacji: {corr_plot_file}")

    print("\nNajsilniejsze korelacje:")
    corr_unstacked = corr_matrix.unstack()
    corr_unstacked = corr_unstacked[corr_unstacked != 1.0]
    strong_corr = corr_unstacked[abs(corr_unstacked) > 0.5].sort_values(ascending=False)

    seen = set()
    for idx, val in strong_corr.items():
        pair = tuple(sorted(idx))
        if pair not in seen:
            print(f"  {idx[0]} <-> {idx[1]}: {val:.3f}")
            seen.add(pair)



print("\nidentyfikacja wartości odstających")

outliers_summary = []

for col in pollutants_present:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    n_outliers = len(outliers)
    pct_outliers = (n_outliers / len(df)) * 100

    outliers_summary.append({
        'Variable': col.upper(),
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound,
        'N_Outliers': n_outliers,
        'Percent_Outliers': pct_outliers
    })

    print(f"  {col.upper()}: {n_outliers} outlierów ({pct_outliers:.2f}%)")

outliers_df = pd.DataFrame(outliers_summary)
outliers_file = STATS_DIR / "outliers_summary.csv"
outliers_df.to_csv(outliers_file, index=False)
print(f"podsumowanie odstajacych wartosci: {outliers_file}")


if pollutants_present:
    fig, axes = plt.subplots(1, len(pollutants_present), figsize=(15, 5))
    if len(pollutants_present) == 1:
        axes = [axes]

    for idx, col in enumerate(pollutants_present):
        bp = axes[idx].boxplot([df[col].dropna()],
                               labels=[col.upper()],
                               patch_artist=True,
                               showfliers=True)

        # Koloruj
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        axes[idx].set_ylabel('Stężenie [µg/m³]')
        axes[idx].set_title(f'Outliers - {col.upper()}')
        axes[idx].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    outliers_plot_file = PLOTS_DIR / "outliers_boxplot.png"
    plt.savefig(outliers_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"wykres wartosci extremalnych - {outliers_plot_file}")



print("\n" + "=" * 80)


print(f"Zakres czasowy: {df['datetime'].min()} - {df['datetime'].max()}")
print('wyniki zostały zapisane')
print("\n" + "=" * 80)
