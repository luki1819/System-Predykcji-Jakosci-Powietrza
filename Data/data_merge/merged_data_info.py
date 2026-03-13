import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === KONFIGURACJA ===
MERGED_DATA_FOLDER = "merged_data"
OUTPUT_FOLDER = "merged_data_analysis"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def load_merged_files():
    """Wczytuje wszystkie pliki CSV z folderu merged_data"""
    files = [f for f in os.listdir(MERGED_DATA_FOLDER) if f.endswith('.csv')]

    if not files:
        print(f"Brak plików CSV w folderze: {MERGED_DATA_FOLDER}")
        return []

    datasets = []
    for file in files:
        file_path = os.path.join(MERGED_DATA_FOLDER, file)
        print(f"Wczytywanie: {file}")
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        datasets.append({
            'filename': file,
            'dataframe': df,
            'station_name': df['station_name'].iloc[0] if 'station_name' in df.columns else 'Unknown'
        })

    return datasets


def analyze_missing_data(df, station_name):
    """Szczegółowa analiza braków danych"""
    print(f"\n{'=' * 60}")
    print(f"ANALIZA BRAKÓW DANYCH: {station_name}")
    print('=' * 60)

    # Podstawowe statystyki
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    print(f"\nPodstawowe informacje:")
    print(f"   - Liczba wierszy: {total_rows}")
    print(f"   - Liczba kolumn: {total_cols}")
    print(f"   - Zakres dat: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"   - Czas trwania: {(df['datetime'].max() - df['datetime'].min()).days} dni")

    # Analiza braków dla każdej kolumny
    missing_stats = []

    for col in df.columns:
        if col in ['station_name', 'station_id', 'datetime']:
            continue

        total = len(df)
        missing = df[col].isnull().sum()
        present = total - missing
        missing_pct = (missing / total) * 100

        missing_stats.append({
            'Kolumna': col,
            'Obecne': present,
            'Brakujące': missing,
            'Procent braków': missing_pct,
            'Min': df[col].min() if present > 0 else None,
            'Max': df[col].max() if present > 0 else None,
            'Średnia': df[col].mean() if present > 0 else None,
            'Mediana': df[col].median() if present > 0 else None,
            'Odch. std.': df[col].std() if present > 0 else None
        })

    df_missing = pd.DataFrame(missing_stats)
    df_missing = df_missing.sort_values('Procent braków', ascending=False)

    print(f"\n📋 Braki danych według kolumn:")
    print(df_missing[['Kolumna', 'Obecne', 'Brakujące', 'Procent braków']].to_string(index=False))

    # Całkowity procent braków
    total_missing = df_missing['Brakujące'].sum()
    overall_missing_pct = (total_missing / (total_rows * len(df_missing))) * 100
    print(f"\nCałkowity procent braków danych: {overall_missing_pct:.2f}%")

    return df_missing


def analyze_column_statistics(df_missing, station_name):
    """Analiza statystyk kolumn"""
    print(f"\n{'=' * 60}")
    print(f"STATYSTYKI KOLUMN: {station_name}")
    print('=' * 60)

    # Kategorie kolumn
    air_quality_keywords = ['pm', 'no2', 'o3', 'so2', 'co', 'pył', 'tlenek', 'ozon', 'dwutlenek']
    weather_keywords = ['temperature', 'humidity', 'precipitation', 'wind', 'pressure', 'boundary']

    air_quality_cols = []
    weather_cols = []
    other_cols = []

    for _, row in df_missing.iterrows():
        col_name = row['Kolumna'].lower()
        if any(kw in col_name for kw in air_quality_keywords):
            air_quality_cols.append(row)
        elif any(kw in col_name for kw in weather_keywords):
            weather_cols.append(row)
        else:
            other_cols.append(row)

    print(f"\nPARAMETRY JAKOŚCI POWIETRZA ({len(air_quality_cols)}):")
    if air_quality_cols:
        df_air = pd.DataFrame(air_quality_cols)
        print(df_air[['Kolumna', 'Procent braków', 'Min', 'Max', 'Średnia']].to_string(index=False))
    else:
        print("   Brak danych")

    print(f"\nPARAMETRY POGODOWE ({len(weather_cols)}):")
    if weather_cols:
        df_weather = pd.DataFrame(weather_cols)
        print(df_weather[['Kolumna', 'Procent braków', 'Min', 'Max', 'Średnia']].to_string(index=False))
    else:
        print("   Brak danych")

    if other_cols:
        print(f"\nINNE PARAMETRY ({len(other_cols)}):")
        df_other = pd.DataFrame(other_cols)
        print(df_other[['Kolumna', 'Procent braków', 'Min', 'Max', 'Średnia']].to_string(index=False))


def create_missing_data_visualizations(df, df_missing, station_name):
    safe_name = station_name.replace(",", "").replace(" ", "_").replace("-", "_")


    fig1 = px.bar(
        df_missing.sort_values('Procent braków', ascending=True),
        y='Kolumna',
        x='Procent braków',
        title=f'Procent braków danych według parametrów - {station_name}',
        labels={'Procent braków': 'Procent braków [%]', 'Kolumna': 'Parametr'},
        color='Procent braków',
        color_continuous_scale='Reds',
        orientation='h',
        height=max(400, len(df_missing) * 25)
    )
    fig1.update_layout(showlegend=False)
    output_path1 = os.path.join(OUTPUT_FOLDER, f"missing_bars_{safe_name}.html")
    fig1.write_html(output_path1)
    print(f"Wykres słupkowy zapisany: {output_path1}")

    # 2. Heatmapa braków w czasie (dziennie)
    df_temp = df.copy()
    df_temp['date'] = df_temp['datetime'].dt.date

    # Wybierz kolumny do analizy (pomijamy metadane)
    cols_to_analyze = [col for col in df.columns if col not in ['station_name', 'station_id', 'datetime', 'date']]

    # Agregacja dzienna - zlicz braki dla każdego dnia
    daily_missing = df_temp.groupby('date')[cols_to_analyze].apply(lambda x: x.isnull().sum())

    if not daily_missing.empty and len(daily_missing) > 0:
        fig2 = px.imshow(
            daily_missing.T,
            aspect='auto',
            color_continuous_scale='Reds',
            labels=dict(x="Data", y="Parametr", color="Liczba braków"),
            title=f'Heatmapa braków danych (dziennie) - {station_name}'
        )
        fig2.update_layout(height=max(500, len(cols_to_analyze) * 20))
        output_path2 = os.path.join(OUTPUT_FOLDER, f"missing_heatmap_{safe_name}.html")
        fig2.write_html(output_path2)
        print(f"Heatmapa zapisana: {output_path2}")

    df_temp['total_missing'] = df_temp[cols_to_analyze].isnull().sum(axis=1)
    daily_totals = df_temp.groupby('date')['total_missing'].sum()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=daily_totals.index,
        y=daily_totals.values,
        mode='lines',
        name='Braki',
        line=dict(color='red', width=2)
    ))
    fig3.update_layout(
        title=f'Trend braków danych w czasie - {station_name}',
        xaxis_title='Data',
        yaxis_title='Liczba braków dziennie',
        height=400
    )
    output_path3 = os.path.join(OUTPUT_FOLDER, f"missing_trend_{safe_name}.html")
    fig3.write_html(output_path3)
    print(f"Wykres trendu zapisany: {output_path3}")


def create_correlation_matrix(df, station_name):
    safe_name = station_name.replace(",", "").replace(" ", "_").replace("-", "_")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'station_id' in numeric_cols:
        numeric_cols.remove('station_id')

    if len(numeric_cols) < 2:
        print(f"Za mało kolumn numerycznych do obliczenia korelacji")
        return

    # Oblicz korelacje
    corr_matrix = df[numeric_cols].corr()

    # Wizualizacja
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=0,
        title=f'Macierz korelacji parametrów - {station_name}'
    )
    fig.update_layout(
        width=max(800, len(numeric_cols) * 40),
        height=max(800, len(numeric_cols) * 40)
    )

    output_path = os.path.join(OUTPUT_FOLDER, f"correlation_{safe_name}.html")
    fig.write_html(output_path)
    print(f"Macierz korelacji zapisana: {output_path}")


def create_data_completeness_timeline(df, station_name):
    safe_name = station_name.replace(",", "").replace(" ", "_").replace("-", "_")

    df_temp = df.copy()
    cols_to_analyze = [col for col in df.columns if col not in ['station_name', 'station_id', 'datetime']]

    # dla każdego wiersza oblicza procent kompletności
    df_temp['completeness'] = (df_temp[cols_to_analyze].notna().sum(axis=1) / len(cols_to_analyze)) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_temp['datetime'],
        y=df_temp['completeness'],
        mode='lines',
        fill='tozeroy',
        name='Kompletność',
        line=dict(color='green', width=1)
    ))
    fig.update_layout(
        title=f'Kompletność danych w czasie - {station_name}',
        xaxis_title='Data',
        yaxis_title='Procent kompletności [%]',
        yaxis_range=[0, 100],
        height=400
    )

    output_path = os.path.join(OUTPUT_FOLDER, f"completeness_timeline_{safe_name}.html")
    fig.write_html(output_path)
    print(f"timeline kompletności zapisany: {output_path}")


def create_summary_report(all_datasets):
    """Tworzy podsumowanie dla wszystkich stacji"""
    print(f"\n{'=' * 60}")
    print("RAPORT ZBIORCZY - WSZYSTKIE STACJE")
    print('=' * 60)

    summary_data = []

    for dataset in all_datasets:
        df = dataset['dataframe']
        station_name = dataset['station_name']

        cols_to_analyze = [col for col in df.columns if col not in ['station_name', 'station_id', 'datetime']]

        total_cells = len(df) * len(cols_to_analyze)
        missing_cells = df[cols_to_analyze].isnull().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0

        summary_data.append({
            'Stacja': station_name,
            'Liczba rekordów': len(df),
            'Liczba parametrów': len(cols_to_analyze),
            'Komórki z danymi': total_cells - missing_cells,
            'Braki': missing_cells,
            'Procent braków': missing_pct,
            'Zakres dat': f"{df['datetime'].min().date()} do {df['datetime'].max().date()}"
        })

    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('procent braków', ascending=True)

    print("\n" + df_summary.to_string(index=False))


    output_path = os.path.join(OUTPUT_FOLDER, "summary_report.csv")
    df_summary.to_csv(output_path, index=False)
    print(f"\nzapisane: {output_path}")

    fig = px.bar(
        df_summary,
        x='Stacja',
        y='Procent braków',
        title='Porównanie braków danych między stacjami',
        color='Procent braków',
        color_continuous_scale='Reds',
        text='Procent braków'
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(height=500, showlegend=False)

    output_path_chart = os.path.join(OUTPUT_FOLDER, "summary_comparison.html")
    fig.write_html(output_path_chart)
    print(f"Wykres porównawczy zapisany: {output_path_chart}")


def main():
    # Wczytaj dane
    datasets = load_merged_files()

    if not datasets:
        return

    print(f"\n wczytano {len(datasets)} plików")

    # Analizuj każdy dataset
    for dataset in datasets:
        df = dataset['dataframe']
        station_name = dataset['station_name']

        # brakówi
        df_missing = analyze_missing_data(df, station_name)
        # statystyki kolumn
        analyze_column_statistics(df_missing, station_name)

        print(f"\nwizualizacja: {station_name}")
        create_missing_data_visualizations(df, df_missing, station_name)
        create_correlation_matrix(df, station_name)
        create_data_completeness_timeline(df, station_name)

    create_summary_report(datasets)

    print("\n" + "=" * 60)
    print("KONIEC")
    print(f"wyniki są w: {os.path.abspath(OUTPUT_FOLDER)}")
    print("=" * 60)


if __name__ == "__main__":
    main()