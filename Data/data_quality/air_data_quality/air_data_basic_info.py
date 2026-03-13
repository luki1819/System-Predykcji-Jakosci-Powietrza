import os
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === KONFIGURACJA ===
path_to_db = "../../AQPS.db"
db_engine = create_engine(f'sqlite:///{path_to_db}')

# === ZAPYTANIE SQL - Łączenie tabel stacje, sensory, pomiary ===
sql_query_air_quality = """
                        SELECT p.data_pomiaru  AS czas_pomiaru, \
                               p.wartosc       AS wartosc_pomiaru, \
                               s.nazwa_sensora AS parametr, \
                               s.jednostka     AS jednostka, \
                               st.nazwa        AS nazwa_stacji, \
                               st.szerokosc    AS szerokosc_geo, \
                               st.dlugosc      AS dlugosc_geo
                        FROM pomiary AS p \
                                 JOIN \
                             sensory AS s ON p.id_sensora = s.id_sensora \
                                 JOIN \
                             stacje AS st ON s.id_stacji = st.id_stacji; \
                        """

print("=" * 80)
print("ANALIZA JAKOŚCI I KOMPLETNOŚCI DANYCH - POMIARY ZANIECZYSZCZEŃ POWIETRZA")
print("=" * 80)

print("\n[1/7] Wczytywanie danych z bazy...")
df_air_quality = pd.read_sql(sql_query_air_quality, db_engine)
print(f"✓ Wczytano {len(df_air_quality):,} rekordów z bazy danych")

print("\n[2/7] Przetwarzanie znaczników czasowych...")
# Konwersja czasu do datetime
df_air_quality['czas_pomiaru'] = pd.to_datetime(
    df_air_quality['czas_pomiaru'],
    errors='coerce'
)

# Zaokrąglenie do pełnej godziny
df_air_quality['czas_pomiaru'] = df_air_quality['czas_pomiaru'].dt.round('h')

# Zakres czasowy danych
min_date = df_air_quality['czas_pomiaru'].min()
max_date = df_air_quality['czas_pomiaru'].max()
date_range_days = (max_date - min_date).days

print(f"✓ Zakres czasowy: {min_date.strftime('%Y-%m-%d')} do {max_date.strftime('%Y-%m-%d')}")
print(f"✓ Okres analizy: {date_range_days} dni ({date_range_days / 365:.1f} lat)")

# Tworzenie unikalnego identyfikatora czujnika
df_air_quality['unikalny_pomiar'] = (
        df_air_quality['nazwa_stacji'] + '_' +
        df_air_quality['parametr']
)

print("\n[3/7] Analiza parametrów monitorowanych...")
# Statystyki parametrów
parametry_unique = df_air_quality['parametr'].unique()
stacje_unique = df_air_quality['nazwa_stacji'].unique()
czujniki_unique = df_air_quality['unikalny_pomiar'].unique()

print(f"✓ Liczba monitorowanych parametrów: {len(parametry_unique)}")
for param in parametry_unique:
    count = len(df_air_quality[df_air_quality['parametr'] == param])
    print(f"  • {param}: {count:,} pomiarów")

print(f"✓ Liczba stacji pomiarowych: {len(stacje_unique)}")
print(f"✓ Łączna liczba czujników (stacja × parametr): {len(czujniki_unique)}")

print("\n[4/7] Tworzenie macierzy danych (pivot table)...")

# Pivot table - każdy czujnik jako osobna kolumna
df_air_quality_wide_sensors = df_air_quality.pivot_table(
    index='czas_pomiaru',
    columns='unikalny_pomiar',
    values='wartosc_pomiaru',
    aggfunc='mean'
).reset_index()

print(
    f"✓ Utworzono macierz: {df_air_quality_wide_sensors.shape[0]} wierszy × {df_air_quality_wide_sensors.shape[1] - 1} kolumn")

# Sprawdzanie kompletności pomiarów za pomocą siatki godzinowej
print("\n[5/7] Uzupełnianie siatki czasowej (reindeksacja)...")
full_date_range = pd.date_range(
    start=df_air_quality_wide_sensors['czas_pomiaru'].min(),
    end=df_air_quality_wide_sensors['czas_pomiaru'].max(),
    freq='h'
)

expected_hours = len(full_date_range)
print(f"✓ Oczekiwana liczba godzinowych pomiarów: {expected_hours:,}")

df_air_quality_wide_sensors.set_index('czas_pomiaru', inplace=True)
df_complete = df_air_quality_wide_sensors.reindex(full_date_range)
df_complete.reset_index(inplace=True)
df_complete.rename(columns={'index': 'czas_pomiaru'}, inplace=True)

df_hourly_analysis = df_complete.copy()

print("\n[6/7] Analiza braków danych...")
czujniki = [col for col in df_complete.columns if col != 'czas_pomiaru']

# Szczegółowa analiza braków
missing_counts = df_complete[czujniki].isnull().sum()
total_cells = len(df_complete) * len(czujniki)
total_missing = missing_counts.sum()
avg_missing_pct = (total_missing / total_cells) * 100

print(f"\n{'=' * 60}")
print("PODSUMOWANIE BRAKÓW DANYCH")
print(f"{'=' * 60}")
print(f"Całkowita liczba komórek w macierzy: {total_cells:,}")
print(f"Liczba brakujących wartości: {total_missing:,}")
print(f"Średni procent braków: {avg_missing_pct:.2f}%")
print(f"Procent kompletności: {100 - avg_missing_pct:.2f}%")

print(f"\n{'Parametr':<50} {'Braki':<10} {'Procent [%]':<12}")
print("-" * 72)
for czujnik in czujniki:
    missing = missing_counts[czujnik]
    pct = (missing / len(df_complete)) * 100
    print(f"{czujnik:<50} {missing:<10} {pct:>10.2f}")

# Statystyki opisowe dla każdego parametru
print(f"\n{'=' * 60}")
print("STATYSTYKI OPISOWE POMIARÓW")
print(f"{'=' * 60}")

for czujnik in czujniki:
    parametr_name = czujnik.split('_')[-1]
    data = df_complete[czujnik].dropna()

    if len(data) > 0:
        print(f"\n{czujnik}:")
        print(f"  Liczba pomiarów: {len(data):,}")
        print(f"  Średnia: {data.mean():.2f}")
        print(f"  Mediana: {data.median():.2f}")
        print(f"  Odchylenie std: {data.std():.2f}")
        print(f"  Min: {data.min():.2f}")
        print(f"  Max: {data.max():.2f}")
        print(f"  Q1 (25%): {data.quantile(0.25):.2f}")
        print(f"  Q3 (75%): {data.quantile(0.75):.2f}")

print("\n[7/7] Generowanie wizualizacji...")

# Przykładowy wykres
if len(czujniki) > 0:
    pierwszy_czujnik = czujniki[0]
    plt.figure(figsize=(10, 5))
    plt.plot(df_complete.index, df_complete[pierwszy_czujnik], label=pierwszy_czujnik, color='tab:blue')
    plt.xlabel('Indeks czasu')
    plt.ylabel(
        f'Wartość [{df_air_quality[df_air_quality["unikalny_pomiar"] == pierwszy_czujnik]["jednostka"].iloc[0] if len(df_air_quality[df_air_quality["unikalny_pomiar"] == pierwszy_czujnik]) > 0 else ""}]')
    plt.title(f'Zmienność {pierwszy_czujnik} w czasie')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('wykres_przykladowy.png')
    print("✅ Zapisano wykres przykładowy jako 'wykres_przykladowy.png'")

# === ANALIZA KOMPLETNOŚCI DANYCH ===
print("\n" + "=" * 80)
print("GENEROWANIE HEATMAPY KOMPLETNOŚCI")
print("=" * 80)

df_analysis = df_complete.set_index('czas_pomiaru')

# 1. Próbkowanie dzienne - zliczamy braki
df_daily_missing = df_analysis.isnull().resample('D').sum()

# 2. Wizualizacja siatki
plt.figure(figsize=(20, 10))
sns.heatmap(
    df_daily_missing.transpose(),
    cmap='viridis_r',
    cbar_kws={'label': 'Liczba brakujących godzin w ciągu dnia'}
)
plt.title('Siatka kompletności danych: Brakujące pomiary (godziny) na dzień')
plt.xlabel('Data')
plt.ylabel('Unikalny czujnik')
plt.tight_layout()
plt.savefig('heatmap_completeness_daily.png', dpi=150)
print("✅ Zapisano heatmapę dzienną jako 'heatmap_completeness_daily.png'")

# === ANALIZA CYKLU DOBOWEGO ===
print("\n" + "=" * 80)
print("ANALIZA CYKLU DOBOWEGO BRAKÓW DANYCH")
print("=" * 80)

df_hourly_analysis['godzina_dnia'] = df_hourly_analysis['czas_pomiaru'].dt.hour

sensor_columns = df_hourly_analysis.columns.drop(['czas_pomiaru', 'godzina_dnia'])
hourly_missing = df_hourly_analysis[sensor_columns].isnull().groupby(df_hourly_analysis['godzina_dnia']).sum()

total_missing_by_hour = hourly_missing.sum(axis=1)

print("\nRozkład braków wg godziny (0-23):")
for hour, missing in total_missing_by_hour.items():
    print(f"  Godzina {hour:02d}: {int(missing):,} brakujących pomiarów")

plt.figure(figsize=(12, 6))
total_missing_by_hour.plot(
    kind='bar',
    color='salmon',
    edgecolor='black'
)
plt.title('Suma brakujących pomiarów wg godziny dnia (Cykl Dobowy)')
plt.xlabel('Godzina (0-23)')
plt.ylabel('Łączna liczba brakujących pomiarów (NaN)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('missing_by_hour.png')
print("\n✅ Zapisano wykres cyklu dobowego jako 'missing_by_hour.png'")

# === WYKRESY SZEREGÓW CZASOWYCH DLA KAŻDEJ STACJI ===
print("\n" + "=" * 80)
print("GENEROWANIE INTERAKTYWNYCH WYKRESÓW SZEREGÓW CZASOWYCH")
print("=" * 80)

folder_timeseries = "air_quality_timeseries"
os.makedirs(folder_timeseries, exist_ok=True)

# Grupowanie czujników według stacji
stacje_dict = {}
for czujnik in czujniki:
    parts = czujnik.rsplit('_', 1)
    if len(parts) == 2:
        nazwa_stacji = parts[0]
        parametr = parts[1]
        if nazwa_stacji not in stacje_dict:
            stacje_dict[nazwa_stacji] = []
        stacje_dict[nazwa_stacji].append(czujnik)

print(f"\nGenerowanie wykresów dla {len(stacje_dict)} stacji...")

# Tworzenie wykresów dla każdej stacji
for nazwa_stacji, sensory_list in stacje_dict.items():
    print(f"  • Stacja: {nazwa_stacji} ({len(sensory_list)} parametrów)")

    num_sensors = len(sensory_list)
    rows = (num_sensors + 1) // 2
    cols = 2 if num_sensors > 1 else 1

    subplot_titles = []
    for sensor in sensory_list:
        param_name = sensor.rsplit('_', 1)[-1] if '_' in sensor else sensor
        unit_data = df_air_quality[df_air_quality['unikalny_pomiar'] == sensor]['jednostka']
        unit = f" [{unit_data.iloc[0]}]" if len(unit_data) > 0 and pd.notna(unit_data.iloc[0]) else ""
        subplot_titles.append(f"{param_name}{unit}")

    while len(subplot_titles) < rows * cols:
        subplot_titles.append("")

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for idx, sensor in enumerate(sensory_list):
        row = (idx // cols) + 1
        col = (idx % cols) + 1
        color = colors[idx % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=df_complete['czas_pomiaru'],
                y=df_complete[sensor],
                mode='lines',
                name=sensor.rsplit('_', 1)[-1] if '_' in sensor else sensor,
                line=dict(color=color, width=1),
                showlegend=False
            ),
            row=row,
            col=col
        )

    fig.update_layout(
        height=300 * rows,
        title_text=f"Parametry jakości powietrza w czasie - {nazwa_stacji}",
        showlegend=False
    )

    safe_name = nazwa_stacji.replace(",", "").replace(" ", "_").replace("-", "_")
    output_path = os.path.join(folder_timeseries, f"timeseries_{safe_name}.html")
    fig.write_html(output_path)

print(f"\n✅ Zapisano {len(stacje_dict)} wykresów szeregów czasowych")

# === HEATMAPY DLA KAŻDEGO CZUJNIKA ===
print("\n" + "=" * 80)
print("GENEROWANIE HEATMAP KOMPLETNOŚCI DLA KAŻDEGO CZUJNIKA")
print("=" * 80)

folder_output = "heatmaps"
os.makedirs(folder_output, exist_ok=True)

print(f"Generowanie heatmap dla {len(czujniki)} czujników...")

for idx, SENSOR_DO_ANALIZY in enumerate(czujniki, 1):
    print(f"  [{idx}/{len(czujniki)}] {SENSOR_DO_ANALIZY}")

    df_sensor = df_complete[['czas_pomiaru', SENSOR_DO_ANALIZY]].copy()
    df_sensor['czy_jest_pomiar'] = df_sensor[SENSOR_DO_ANALIZY].notnull().astype(int)
    df_sensor['data'] = df_sensor['czas_pomiaru'].dt.date
    df_sensor['godzina_dnia'] = df_sensor['czas_pomiaru'].dt.hour

    try:
        df_heatmap_data = df_sensor.pivot_table(
            index='data',
            columns='godzina_dnia',
            values='czy_jest_pomiar',
            aggfunc='max'
        )

        fig = px.imshow(
            df_heatmap_data,
            aspect='auto',
            color_continuous_scale=[(0, '#ff4444'), (1, '#44ff44')],
            labels=dict(x="Godzina dnia", y="Data", color="Pomiar")
        )

        fig.update_layout(
            title=f"Kompletność danych: {SENSOR_DO_ANALIZY}",
            xaxis_nticks=24,
            height=800
        )

        safe_name = SENSOR_DO_ANALIZY.replace(",", "").replace(" ", "_").replace("-", "_")
        output_path = os.path.join(folder_output, f"heatmap_{safe_name}.html")
        fig.write_html(output_path)

    except Exception as e:
        print(f"    ❌ Błąd: {e}")

print(f"\n✅ Zapisano {len(czujniki)} heatmap")

# === ANALIZA MIESIĘCZNA ===
print("\n" + "=" * 80)
print("GENEROWANIE WYKRESÓW MIESIĘCZNYCH BRAKÓW DANYCH")
print("=" * 80)

folder_monthly = "monthly_missing_charts"
os.makedirs(folder_monthly, exist_ok=True)

df_monthly_analysis = df_complete.copy()
df_monthly_analysis['rok_miesiac'] = df_monthly_analysis['czas_pomiaru'].dt.to_period('M')

print(f"Generowanie wykresów miesięcznych dla {len(czujniki)} czujników...")

for idx, SENSOR_DO_ANALIZY in enumerate(czujniki, 1):
    print(f"  [{idx}/{len(czujniki)}] {SENSOR_DO_ANALIZY}")

    try:
        monthly_missing = df_monthly_analysis.groupby('rok_miesiac')[SENSOR_DO_ANALIZY].apply(
            lambda x: x.isnull().sum()
        )

        monthly_missing.index = monthly_missing.index.astype(str)

        fig_monthly = px.bar(
            x=monthly_missing.index,
            y=monthly_missing.values,
            labels={'x': 'Miesiąc', 'y': 'Liczba brakujących pomiarów'},
            title=f'Brakujące pomiary wg miesiąca: {SENSOR_DO_ANALIZY}',
            color=monthly_missing.values,
            color_continuous_scale='Reds'
        )

        fig_monthly.update_layout(
            xaxis_tickangle=-45,
            height=600,
            showlegend=False
        )

        safe_name = SENSOR_DO_ANALIZY.replace(",", "").replace(" ", "_").replace("-", "_")
        output_path_monthly = os.path.join(folder_monthly, f"monthly_missing_{safe_name}.html")
        fig_monthly.write_html(output_path_monthly)

    except Exception as e:
        print(f"    ❌ Błąd: {e}")

print(f"\n✅ Zapisano {len(czujniki)} wykresów miesięcznych")

# === FINALNE PODSUMOWANIE ===
print("\n" + "=" * 80)
print("ANALIZA ZAKOŃCZONA")
print("=" * 80)
print(f"\n📊 Przeanalizowano:")
print(f"  • {len(df_air_quality):,} rekordów pomiarowych")
print(f"  • {len(czujniki)} czujników")
print(f"  • {len(stacje_dict)} stacji pomiarowych")
print(f"  • Okres: {date_range_days} dni ({min_date.strftime('%Y-%m-%d')} - {max_date.strftime('%Y-%m-%d')})")
print(f"\n📁 Wygenerowane pliki:")
print(f"  • {os.path.abspath(folder_timeseries)}/ - szeregi czasowe (HTML)")
print(f"  • {os.path.abspath(folder_output)}/ - heatmapy kompletności (HTML)")
print(f"  • {os.path.abspath(folder_monthly)}/ - wykresy miesięczne (HTML)")
print(f"  • wykres_przykladowy.png - przykładowy wykres statyczny")
print(f"  • heatmap_completeness_daily.png - heatmapa dzienna")
print(f"  • missing_by_hour.png - cykl dobowy braków")
print("=" * 80)