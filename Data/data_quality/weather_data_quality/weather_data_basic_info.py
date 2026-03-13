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

# === ZAPYTANIE SQL - Łączenie tabel pogoda_lokalizacje i pogoda_pomiary ===
sql_query_weather = """
                    SELECT p.data_pomiaru AS czas_pomiaru, \
                           p.temperature_2m, \
                           p.relative_humidity_2m, \
                           p.precipitation, \
                           p.wind_speed_10m, \
                           p.wind_direction_10m, \
                           p.pressure_msl, \
                           p.boundary_layer_height, \
                           l.nazwa        AS lokalizacja, \
                           l.szerokosc, \
                           l.dlugosc
                    FROM pogoda_pomiary AS p \
                             JOIN \
                         pogoda_lokalizacje AS l ON p.id_lokalizacji = l.id_lokalizacji
                    ORDER BY p.data_pomiaru; \
                    """

print("Wczytywanie danych pogodowych z bazy...")
df_weather = pd.read_sql(sql_query_weather, db_engine)

print("Rozpoczynam konwersję czasu...")
# Konwersja czasu do datetime
df_weather['czas_pomiaru'] = pd.to_datetime(
    df_weather['czas_pomiaru'],
    errors='coerce'
)

# Usuń wiersze z błędnym formatem daty
original_rows = len(df_weather)
df_weather.dropna(subset=['czas_pomiaru'], inplace=True)
deleted_rows = original_rows - len(df_weather)
if deleted_rows > 0:
    print(f"Usunięto {deleted_rows} wierszy z błędnym formatem daty.")

print("--- Poprawnie przetworzone dane pogodowe ---")
print(df_weather.head())
print("\nInformacje o typie danych:")
df_weather.info()

# Sprawdzenie dostępnych lokalizacji
lokalizacje = df_weather['lokalizacja'].unique()
print(f"\n--- Dostępne lokalizacje ({len(lokalizacje)}) ---")
for lok in lokalizacje:
    liczba_pomiarow = len(df_weather[df_weather['lokalizacja'] == lok])
    print(f"  • {lok}: {liczba_pomiarow} pomiarów")

# === ANALIZA KOMPLETNOŚCI DANYCH ===
print("\n--- Tworzenie pełnej siatki godzinowej ---")

# Dla każdej lokalizacji osobno
for lokalizacja in lokalizacje:
    print(f"\n{'=' * 60}")
    print(f"ANALIZA LOKALIZACJI: {lokalizacja}")
    print('=' * 60)

    df_loc = df_weather[df_weather['lokalizacja'] == lokalizacja].copy()

    # Tworzenie pełnej siatki czasowej
    full_date_range = pd.date_range(
        start=df_loc['czas_pomiaru'].min(),
        end=df_loc['czas_pomiaru'].max(),
        freq='h'
    )

    df_loc.set_index('czas_pomiaru', inplace=True)
    df_complete = df_loc.reindex(full_date_range)
    df_complete.reset_index(inplace=True)
    df_complete.rename(columns={'index': 'czas_pomiaru'}, inplace=True)

    print(f"\n--- Statystyki dla {lokalizacja} ---")
    print(f"Zakres dat: od {full_date_range.min()} do {full_date_range.max()}")
    print(f"Całkowita liczba godzin: {len(full_date_range)}")

    # Parametry pogodowe
    weather_params = [
        'temperature_2m', 'relative_humidity_2m', 'precipitation',
        'wind_speed_10m', 'wind_direction_10m', 'pressure_msl',
        'boundary_layer_height'
    ]

    print("\n--- Sprawdzenie braków danych ---")
    missing_counts = df_complete[weather_params].isnull().sum()
    print(missing_counts)

    total_cells = len(df_complete) * len(weather_params)
    total_missing = missing_counts.sum()
    print(f"\nŚredni procent braków: {(total_missing / total_cells * 100):.2f}%")

    # === WYKRESY SZEREGÓW CZASOWYCH ===
    print("\n--- Tworzenie wykresów szeregów czasowych ---")

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Temperatura [°C]', 'Wilgotność względna [%]',
            'Opady [mm]', 'Prędkość wiatru [m/s]',
            'Kierunek wiatru [°]', 'Ciśnienie [hPa]',
            'Wysokość warstwy granicznej [m]', ''
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )

    # Temperatura
    fig.add_trace(
        go.Scatter(x=df_complete['czas_pomiaru'], y=df_complete['temperature_2m'],
                   mode='lines', name='Temperatura', line=dict(color='red')),
        row=1, col=1
    )

    # Wilgotność
    fig.add_trace(
        go.Scatter(x=df_complete['czas_pomiaru'], y=df_complete['relative_humidity_2m'],
                   mode='lines', name='Wilgotność', line=dict(color='blue')),
        row=1, col=2
    )

    # Opady
    fig.add_trace(
        go.Scatter(x=df_complete['czas_pomiaru'], y=df_complete['precipitation'],
                   mode='lines', name='Opady', line=dict(color='green'), fill='tozeroy'),
        row=2, col=1
    )

    # Prędkość wiatru
    fig.add_trace(
        go.Scatter(x=df_complete['czas_pomiaru'], y=df_complete['wind_speed_10m'],
                   mode='lines', name='Prędkość wiatru', line=dict(color='purple')),
        row=2, col=2
    )

    # Kierunek wiatru
    fig.add_trace(
        go.Scatter(x=df_complete['czas_pomiaru'], y=df_complete['wind_direction_10m'],
                   mode='markers', name='Kierunek wiatru', marker=dict(color='orange', size=2)),
        row=3, col=1
    )

    # Ciśnienie
    fig.add_trace(
        go.Scatter(x=df_complete['czas_pomiaru'], y=df_complete['pressure_msl'],
                   mode='lines', name='Ciśnienie', line=dict(color='brown')),
        row=3, col=2
    )

    # Wysokość warstwy granicznej
    fig.add_trace(
        go.Scatter(x=df_complete['czas_pomiaru'], y=df_complete['boundary_layer_height'],
                   mode='lines', name='Wys. warstwy gran.', line=dict(color='darkgreen')),
        row=4, col=1
    )

    fig.update_layout(
        height=1200,
        title_text=f"Parametry pogodowe w czasie - {lokalizacja}",
        showlegend=False
    )

    safe_name = lokalizacja.replace(",", "").replace(" ", "_").replace("-", "_")
    folder_timeseries = "weather_timeseries"
    os.makedirs(folder_timeseries, exist_ok=True)
    output_path = os.path.join(folder_timeseries, f"timeseries_{safe_name}.html")
    fig.write_html(output_path)
    print(f"✓ Wykres szeregów czasowych zapisany jako: {output_path}")

    # === HEATMAPA KOMPLETNOŚCI DANYCH ===
    print("\n--- Tworzenie heatmapy kompletności danych ---")

    df_analysis = df_complete.set_index('czas_pomiaru')
    df_daily_missing = df_analysis[weather_params].isnull().resample('D').sum()

    plt.figure(figsize=(20, 8))
    sns.heatmap(
        df_daily_missing.transpose(),
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Liczba brakujących godzin w ciągu dnia'},
        vmin=0, vmax=24
    )
    plt.title(f'Kompletność danych pogodowych - {lokalizacja}')
    plt.xlabel('Data')
    plt.ylabel('Parametr pogodowy')
    plt.tight_layout()

    folder_heatmaps = "weather_heatmaps"
    os.makedirs(folder_heatmaps, exist_ok=True)
    output_path_hm = os.path.join(folder_heatmaps, f"heatmap_completeness_{safe_name}.png")
    plt.savefig(output_path_hm, dpi=150)
    plt.close()
    print(f"✓ Heatmapa kompletności zapisana jako: {output_path_hm}")

    # === ANALIZA CYKLU DOBOWEGO ===
    print("\n--- Analiza cyklu dobowego ---")

    df_hourly = df_complete.copy()
    df_hourly['godzina_dnia'] = df_hourly['czas_pomiaru'].dt.hour

    hourly_missing = df_hourly[weather_params].isnull().groupby(df_hourly['godzina_dnia']).sum()
    total_missing_by_hour = hourly_missing.sum(axis=1)

    plt.figure(figsize=(12, 6))
    total_missing_by_hour.plot(
        kind='bar',
        color='steelblue',
        edgecolor='black'
    )
    plt.title(f'Brakujące pomiary wg godziny dnia - {lokalizacja}')
    plt.xlabel('Godzina (0-23)')
    plt.ylabel('Łączna liczba brakujących pomiarów')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    folder_hourly = "weather_hourly"
    os.makedirs(folder_hourly, exist_ok=True)
    output_path_h = os.path.join(folder_hourly, f"hourly_missing_{safe_name}.png")
    plt.savefig(output_path_h)
    plt.close()
    print(f"✓ Wykres cyklu dobowego zapisany jako: {output_path_h}")

    # === ANALIZA MIESIĘCZNA ===
    print("\n--- Analiza miesięczna ---")

    df_monthly = df_complete.copy()
    df_monthly['rok_miesiac'] = df_monthly['czas_pomiaru'].dt.to_period('M')

    folder_monthly = "weather_monthly"
    os.makedirs(folder_monthly, exist_ok=True)

    for param in weather_params:
        monthly_missing = df_monthly.groupby('rok_miesiac')[param].apply(
            lambda x: x.isnull().sum()
        )

        monthly_missing.index = monthly_missing.index.astype(str)

        fig_monthly = px.bar(
            x=monthly_missing.index,
            y=monthly_missing.values,
            labels={'x': 'Miesiąc', 'y': 'Liczba brakujących pomiarów'},
            title=f'Brakujące pomiary: {param} - {lokalizacja}',
            color=monthly_missing.values,
            color_continuous_scale='Reds'
        )

        fig_monthly.update_layout(
            xaxis_tickangle=-45,
            height=600,
            showlegend=False
        )

        output_path_m = os.path.join(folder_monthly, f"monthly_{param}_{safe_name}.html")
        fig_monthly.write_html(output_path_m)

    print(f"✓ Wykresy miesięczne zapisane w folderze: {folder_monthly}")

    # === STATYSTYKI OPISOWE ===
    print("\n--- Statystyki opisowe ---")
    print(df_complete[weather_params].describe())

    # === KORELACJE ===
    print("\n--- Macierz korelacji ---")

    correlation_matrix = df_complete[weather_params].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1
    )
    plt.title(f'Macierz korelacji parametrów pogodowych - {lokalizacja}')
    plt.tight_layout()

    folder_corr = "weather_correlation"
    os.makedirs(folder_corr, exist_ok=True)
    output_path_corr = os.path.join(folder_corr, f"correlation_{safe_name}.png")
    plt.savefig(output_path_corr, dpi=150)
    plt.close()
    print(f"✓ Macierz korelacji zapisana jako: {output_path_corr}")

print("\n" + "=" * 60)
print("ANALIZA ZAKOŃCZONA DLA WSZYSTKICH LOKALIZACJI")
print("\nWygenerowane foldery:")
print(f"  • {os.path.abspath('weather_timeseries')} - szeregi czasowe")
print(f"  • {os.path.abspath('weather_heatmaps')} - heatmapy kompletności")
print(f"  • {os.path.abspath('weather_hourly')} - analiza godzinowa")
print(f"  • {os.path.abspath('weather_monthly')} - analiza miesięczna")
print(f"  • {os.path.abspath('weather_correlation')} - macierze korelacji")
print("=" * 60)