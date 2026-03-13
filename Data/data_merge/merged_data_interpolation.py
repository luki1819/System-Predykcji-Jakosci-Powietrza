import pandas as pd
import numpy as np
import os
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# === KONFIGURACJA ===
MERGED_DATA_FOLDER = "merged_data"
OUTPUT_FOLDER = "feature_interpolation_data"
REPORTS_FOLDER = "feature_interpolation_reports"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Parametry obsługi braków
CONFIG = {
    'SHORT_GAP_HOURS': 4,  # Luki <= 4h: interpolacja liniowa
    'MEDIUM_GAP_HOURS': 24,  # Luki <= 24h: mediana sąsiednich dni
    'LONG_GAP_HOURS': 300,  # Luki <= 300h: imputacja historyczna + hourly mean
    'EXTREME_GAP_HOURS': 300,  # Luki > 300h: usuń całą kolumnę
    'MIN_DATA_THRESHOLD': 0.2,  # Usuń kolumny z < 20% danych
    'HISTORICAL_WINDOW_DAYS': 7,  # Okno wyszukiwania ±7 dni
    'HISTORICAL_YEARS_BACK': 2,  # Szukaj w ±2 lata wstecz
    'MIN_HISTORICAL_VALUES': 5,  # Minimum wartości do imputacji historycznej
    'MIN_HOURLY_MEAN_VALUES': 10,  # Minimum wartości do metody hourly mean
}

# Parametry jakości powietrza (nie mogą być ujemne)
AIR_QUALITY_PARAMS = [
    'pm10', 'pm2.5', 'pm25', 'no2', 'o3', 'so2', 'co',
    'pył', 'tlenek', 'dwutlenek', 'ozon'
]


def load_merged_file(filepath):
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    return df


def analyze_gap_lengths(df, column):
    is_missing = df[column].isnull()

    gaps = []
    gap_start = None
    gap_length = 0

    for idx, missing in enumerate(is_missing):
        if missing:
            if gap_start is None:
                gap_start = idx
            gap_length += 1
        else:
            if gap_start is not None:
                gaps.append({
                    'start': gap_start,
                    'end': gap_start + gap_length - 1,
                    'length': gap_length,
                    'start_time': df.index[gap_start],
                    'end_time': df.index[gap_start + gap_length - 1]
                })
                gap_start = None
                gap_length = 0

    # a jak dane kończą się luką
    if gap_start is not None:
        gaps.append({
            'start': gap_start,
            'end': len(df) - 1,
            'length': gap_length,
            'start_time': df.index[gap_start],
            'end_time': df.index[-1]
        })

    return gaps


def is_air_quality_parameter(column_name):
    column_lower = column_name.lower()
    return any(param in column_lower for param in AIR_QUALITY_PARAMS)


def clip_negative_values(df, column):
    """Zastępuje wartości ujemne zerem dla parametrów jakości powietrza"""
    negative_count = (df[column] < 0).sum()
    if negative_count > 0:
        df[column] = df[column].clip(lower=0).round(2)
        print(f"     zaminiono {negative_count} wartości ujemnych → 0")
    return df


def categorize_gaps(gaps, config):
    # do statystyk żeby wiedzieć jak duzo
    categorized = {
        'short': [],  # <= SHORT_GAP_HOURS
        'medium': [],  # <= MEDIUM_GAP_HOURS
        'long': [],  # <= LONG_GAP_HOURS
        'extreme': []  # > EXTREME_GAP_HOURS
    }

    for gap in gaps:
        length = gap['length']
        if length <= config['SHORT_GAP_HOURS']:
            categorized['short'].append(gap)
        elif length <= config['MEDIUM_GAP_HOURS']:
            categorized['medium'].append(gap)
        elif length <= config['LONG_GAP_HOURS']:
            categorized['long'].append(gap)
        else:
            categorized['extreme'].append(gap)

    return categorized


def fill_short_gaps(df, column, gaps, method='linear'):
    """wypełnia krótkie luki (≤4h) - interpolacja liniowa"""
    df_filled = df.copy()
    filled_count = 0
    is_air_quality = is_air_quality_parameter(column)

    for gap in gaps:
        start = gap['start']
        end = gap['end']
        gap_length = gap['length']

        # Sprawdź czy są wartości przed i po luce
        has_before = start > 0 and pd.notna(df_filled.iloc[start - 1][column])
        has_after = end < len(df_filled) - 1 and pd.notna(df_filled.iloc[end + 1][column])

        if has_before and has_after:
            # Interpolacja liniowa między wartościami
            value_before = df_filled.iloc[start - 1][column]
            value_after = df_filled.iloc[end + 1][column]

            for i, idx in enumerate(range(start, end + 1)):
                interpolated_value = value_before + (value_after - value_before) * (i + 1) / (gap_length + 1)
                df_filled.iloc[idx, df_filled.columns.get_loc(column)] = round(interpolated_value, 2)
                filled_count += 1

    # dal parametrów jakości powietrza - usuń wartości ujemne
    if is_air_quality:
        df_filled = clip_negative_values(df_filled, column)

    return df_filled, filled_count


def get_historical_values(df, column, target_datetime, config):
    #wartości z lat poprzednich żeby móc uzupełnić nimi
    historical_values = []

    month = target_datetime.month
    day = target_datetime.day
    hour = target_datetime.hour

    # szukaj wartości z ±HISTORICAL_YEARS_BACK lat
    for year_offset in range(-config['HISTORICAL_YEARS_BACK'], config['HISTORICAL_YEARS_BACK'] + 1):
        if year_offset == 0:
            continue

        try:
            target_year = target_datetime.year + year_offset

            # szukaj wartości w oknie ±HISTORICAL_WINDOW_DAYS dni
            for day_offset in range(-config['HISTORICAL_WINDOW_DAYS'],
                                    config['HISTORICAL_WINDOW_DAYS'] + 1):
                try:
                    target_date = target_datetime.replace(year=target_year) + timedelta(days=day_offset)

                    # Znajdź wartość dla tej daty
                    if target_date in df.index:
                        value = df.loc[target_date, column]
                        if pd.notna(value):
                            historical_values.append(value)
                except (ValueError, KeyError):
                    continue
        except (ValueError, KeyError):
            continue

    return historical_values


def get_hourly_mean_value(df, column, target_datetime, config):
    """srednie godzinowe z roku"""
    hour = target_datetime.hour

    # znajdź wszystkie wartości z tej samej godziny w całym zbiorze
    same_hour_mask = df.index.hour == hour
    same_hour_values = df.loc[same_hour_mask, column].dropna()

    if len(same_hour_values) >= config['MIN_HOURLY_MEAN_VALUES']:
        # mediana
        return np.median(same_hour_values)

    return None


def fill_medium_gaps(df, column, gaps, config):
    """
    Wypełnia średnie luki (≤24h) - mediana z sąsiednich dni
    Dla każdej brakującej godziny oblicza medianę z tej samej godziny w sąsiednich dniach
    Jeśli nie ma wystarczająco sąsiednich wartości, używa interpolacji liniowej
    """
    df_filled = df.copy()
    filled_count = 0
    is_air_quality = is_air_quality_parameter(column)

    for gap in gaps:
        start = gap['start']
        end = gap['end']

        for idx in range(start, end + 1):
            current_datetime = df_filled.index[idx]
            hour = current_datetime.hour

            # Szukaj wartości z tej samej godziny w sąsiednich dniach (±3 dni)
            neighboring_values = []

            for day_offset in range(-3, 4):  # ±3 dni
                if day_offset == 0:
                    continue

                try:
                    target_date = current_datetime + timedelta(days=day_offset)

                    if target_date in df_filled.index:
                        value = df_filled.loc[target_date, column]
                        if pd.notna(value):
                            neighboring_values.append(value)
                except (ValueError, KeyError):
                    continue

            # Jeśli znaleziono co najmniej 2 wartości, użyj mediany
            if len(neighboring_values) >= 2:
                filled_value = round(np.median(neighboring_values), 2)
                df_filled.iloc[idx, df_filled.columns.get_loc(column)] = filled_value
                filled_count += 1


    for gap in gaps:
        start = gap['start']
        end = gap['end']


        start_idx = df_filled.index[start]
        end_idx = df_filled.index[end]

        mask = (df_filled.index >= start_idx) & (df_filled.index <= end_idx)
        df_filled.loc[mask, column] = df_filled.loc[mask, column].interpolate(method='linear')

    filled_count = df_filled[column].notna().sum() - df[column].notna().sum()

    # Usuń wartości ujemne dla parametrów jakości powietrza
    if is_air_quality:
        df_filled = clip_negative_values(df_filled, column)

    return df_filled, filled_count


def fill_long_gaps(df, column, gaps, config):
    """Wypełnia długie luki (≤300h) - imputacja historyczna + hourly mean jako fallback"""
    df_filled = df.copy()
    filled_count = 0
    historical_count = 0
    hourly_mean_count = 0
    is_air_quality = is_air_quality_parameter(column)

    for gap in gaps:
        start = gap['start']
        end = gap['end']

        # Dla każdego punktu w luce
        for idx in range(start, end + 1):
            current_datetime = df_filled.index[idx]

            # METODA 1: Imputacja historyczna (priorytet)
            historical_values = get_historical_values(df_filled, column, current_datetime, config)

            if len(historical_values) >= config['MIN_HISTORICAL_VALUES']:
                filled_value = round(np.median(historical_values), 2)
                df_filled.iloc[idx, df_filled.columns.get_loc(column)] = filled_value
                filled_count += 1
                historical_count += 1
                continue

            # METODA 2: Hourly Mean (fallback)
            hourly_mean_value = get_hourly_mean_value(df_filled, column, current_datetime, config)

            if hourly_mean_value is not None:
                filled_value = round(hourly_mean_value, 2)
                df_filled.iloc[idx, df_filled.columns.get_loc(column)] = filled_value
                filled_count += 1
                hourly_mean_count += 1

    # Dla parametrów jakości powietrza - usuń wartości ujemne
    if is_air_quality:
        df_filled = clip_negative_values(df_filled, column)

    if historical_count > 0 or hourly_mean_count > 0:
        print(f"        - Imputacja historyczna: {historical_count} wartości")
        print(f"        - Hourly Mean (fallback): {hourly_mean_count} wartości")

    return df_filled, filled_count


def process_column(df, column, config, report):
    """Przetwarza pojedynczą kolumnę"""
    if column in ['station_name', 'station_id']:
        return df, report

    is_air_quality = is_air_quality_parameter(column)

    print(f"\n  📊 Przetwarzanie kolumny: {column}")
    if is_air_quality:
        print(f"     🌫️  Parametr jakości powietrza - wartości ujemne będą zastąpione zerem")

    # Analiza braków
    total_values = len(df)
    missing_values = df[column].isnull().sum()
    missing_pct = (missing_values / total_values) * 100

    print(f"     - Braki: {missing_values}/{total_values} ({missing_pct:.2f}%)")

    # Sprawdź czy kolumna ma wystarczająco dużo danych
    if missing_pct > (100 - config['MIN_DATA_THRESHOLD'] * 100):
        print(f"     Kolumna ma < {config['MIN_DATA_THRESHOLD'] * 100}% danych - będzie usunięta")
        report['columns_removed'].append({
            'column': column,
            'reason': 'insufficient_data',
            'missing_pct': missing_pct
        })
        return df.drop(columns=[column]), report

    # Analizuj długości luk
    gaps = analyze_gap_lengths(df, column)

    if not gaps:
        print(f"     ✓ Brak braków")
        report['columns_complete'].append(column)
        return df, report

    categorized = categorize_gaps(gaps, config)

    print(f"     - Krótkie luki (≤{config['SHORT_GAP_HOURS']}h): {len(categorized['short'])} - interpolacja liniowa")
    print(
        f"     - Średnie luki (≤{config['MEDIUM_GAP_HOURS']}h): {len(categorized['medium'])} - mediana z sąsiednich dni")
    print(
        f"     - Długie luki (≤{config['LONG_GAP_HOURS']}h): {len(categorized['long'])} - imputacja historyczna + hourly mean")
    print(f"     - Ekstremalne luki (>{config['EXTREME_GAP_HOURS']}h): {len(categorized['extreme'])}")

    # Sprawdź czy są ekstremalne luki
    if categorized['extreme']:
        print(f"     Znaleziono ekstremalnie długie luki - kolumna będzie usunięta")
        report['columns_removed'].append({
            'column': column,
            'reason': 'extreme_gaps',
            'extreme_gaps': len(categorized['extreme']),
            'max_gap_length': max([g['length'] for g in categorized['extreme']])
        })
        return df.drop(columns=[column]), report

    # Wypełnianie luk
    df_result = df.copy()
    total_filled = 0

    if categorized['short']:
        df_result, filled = fill_short_gaps(df_result, column, categorized['short'])
        total_filled += filled
        print(f"     Wypełniono {filled} wartości (krótkie luki - interpolacja liniowa)")

    if categorized['medium']:
        df_result, filled = fill_medium_gaps(df_result, column, categorized['medium'], config)
        total_filled += filled
        print(f"      Wypełniono {filled} wartości (średnie luki - interpolacja + imputacja historyczna)")

    if categorized['long']:
        df_result, filled = fill_long_gaps(df_result, column, categorized['long'], config)
        total_filled += filled
        print(f"      Wypełniono {filled} wartości (długie luki - imputacja historyczna + hourly mean)")

    # Sprawdź czy wszystkie braki zostały wypełnione
    remaining_missing = df_result[column].isnull().sum()
    if remaining_missing > 0:
        print(f"      Pozostało {remaining_missing} niewypełnionych braków - usuwanie kolumny")
        report['columns_removed'].append({
            'column': column,
            'reason': 'unfilled_gaps',
            'remaining_missing': remaining_missing
        })
        return df.drop(columns=[column]), report

    report['columns_processed'].append({
        'column': column,
        'original_missing': missing_values,
        'filled': total_filled,
        'gaps_short': len(categorized['short']),
        'gaps_medium': len(categorized['medium']),
        'gaps_long': len(categorized['long'])
    })

    return df_result, report


def create_before_after_comparison(df_before, df_after, column, station_name):
    """Tworzy wykres porównawczy przed/po wypełnieniu"""
    safe_name = station_name.replace(",", "").replace(" ", "_").replace("-", "_")
    safe_col = column.replace(" ", "_").replace("/", "_")

    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    # Przed
    axes[0].plot(df_before.index, df_before[column], 'b-', linewidth=0.5, alpha=0.7)
    axes[0].set_title(f'Przed wypełnieniem - {column}')
    axes[0].set_ylabel('Wartość')
    axes[0].grid(True, alpha=0.3)
    missing_before = df_before[column].isnull().sum()
    axes[0].text(0.02, 0.95, f'Braki: {missing_before}', transform=axes[0].transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Po
    axes[1].plot(df_after.index, df_after[column], 'g-', linewidth=0.5, alpha=0.7)
    axes[1].set_title(f'Po wypełnieniu - {column}')
    axes[1].set_xlabel('Data')
    axes[1].set_ylabel('Wartość')
    axes[1].grid(True, alpha=0.3)
    missing_after = df_after[column].isnull().sum()
    axes[1].text(0.02, 0.95, f'Braki: {missing_after}', transform=axes[1].transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(REPORTS_FOLDER, f"comparison_{safe_name}_{safe_col}.png")
    plt.savefig(output_path, dpi=100)
    plt.close()

    return output_path


def process_dataset(filepath, config):
    """Przetwarza jeden plik danych"""
    filename = os.path.basename(filepath)
    print(f"\n{'=' * 60}")
    print(f"PRZETWARZANIE: {filename}")
    print('=' * 60)

    # Wczytaj dane
    df_original = load_merged_file(filepath)
    station_name = df_original['station_name'].iloc[0] if 'station_name' in df_original.columns else 'Unknown'

    print(f"\n Informacje wstępne:")
    print(f"   - Stacja: {station_name}")
    print(f"   - Liczba wierszy: {len(df_original)}")
    print(f"   - Liczba kolumn: {len(df_original.columns)}")
    print(f"   - Zakres dat: {df_original.index.min()} → {df_original.index.max()}")

    # Raport
    report = {
        'station_name': station_name,
        'filename': filename,
        'original_rows': len(df_original),
        'original_cols': len(df_original.columns),
        'columns_processed': [],
        'columns_removed': [],
        'columns_complete': []
    }

    # Przetwarzaj każdą kolumnę
    df_processed = df_original.copy()
    columns_to_process = [col for col in df_original.columns if col not in ['station_name', 'station_id']]

    for column in columns_to_process:
        df_before = df_processed.copy()
        df_processed, report = process_column(df_processed, column, config, report)

        # Jeśli kolumna została przetworzona (nie usunięta), utwórz wykres porównawczy
        if column in df_processed.columns and column not in report['columns_complete']:
            if df_before[column].isnull().sum() > 0:  # Tylko jeśli były braki
                create_before_after_comparison(df_before, df_processed, column, station_name)

    # Podsumowanie
    report['final_rows'] = len(df_processed)
    report['final_cols'] = len(df_processed.columns)
    report['removed_cols_count'] = len(report['columns_removed'])
    report['processed_cols_count'] = len(report['columns_processed'])
    report['complete_cols_count'] = len(report['columns_complete'])

    print(f"\n{'=' * 60}")
    print(f"PODSUMOWANIE: {station_name}")
    print('=' * 60)
    print(f"✓ Kolumny kompletne (bez braków): {report['complete_cols_count']}")
    print(f"✓ Kolumny przetworzone (wypełnione): {report['processed_cols_count']}")
    print(f"✗ Kolumny usunięte: {report['removed_cols_count']}")

    if report['columns_removed']:
        print(f"\n Usunięte kolumny:")
        for removed in report['columns_removed']:
            print(f"   - {removed['column']}: {removed['reason']}")

    # Zapisz przetworzone dane
    safe_name = station_name.replace(",", "").replace(" ", "_").replace("-", "_")
    output_path = os.path.join(OUTPUT_FOLDER, f"processed_{safe_name}.csv")
    df_processed.reset_index().to_csv(output_path, index=False)
    print(f"\nDane zapisane: {output_path}")

    return df_processed, report


def create_summary_report(all_reports):
    """Tworzy zbiorczy raport z przetwarzania"""
    print(f"\n{'=' * 60}")
    print("RAPORT ZBIORCZY")
    print('=' * 60)

    summary_data = []

    for report in all_reports:
        summary_data.append({
            'Stacja': report['station_name'],
            'Oryg. kolumny': report['original_cols'],
            'Finalne kolumny': report['final_cols'],
            'Kompletne': report['complete_cols_count'],
            'Przetworzone': report['processed_cols_count'],
            'Usunięte': report['removed_cols_count']
        })

    df_summary = pd.DataFrame(summary_data)
    print("\n" + df_summary.to_string(index=False))

    # Zapisz raport
    output_path = os.path.join(REPORTS_FOLDER, "processing_summary.csv")
    df_summary.to_csv(output_path, index=False)
    print(f"\n Raport zbiorczy zapisany: {output_path}")

    # Szczegółowy raport
    detailed_report_path = os.path.join(REPORTS_FOLDER, "detailed_report.txt")
    with open(detailed_report_path, 'w', encoding='utf-8') as f:
        f.write("SZCZEGÓŁOWY RAPORT PRZETWARZANIA\n")
        f.write("=" * 60 + "\n\n")

        for report in all_reports:
            f.write(f"\nSTACJA: {report['station_name']}\n")
            f.write("-" * 60 + "\n")

            f.write(f"\nKolumny kompletne ({len(report['columns_complete'])}):\n")
            for col in report['columns_complete']:
                f.write(f"  - {col}\n")

            f.write(f"\nKolumny przetworzone ({len(report['columns_processed'])}):\n")
            for col_info in report['columns_processed']:
                f.write(f"  - {col_info['column']}:\n")
                f.write(f"    * Braki początkowe: {col_info['original_missing']}\n")
                f.write(f"    * Wypełniono: {col_info['filled']}\n")
                f.write(f"    * Luki krótkie: {col_info['gaps_short']}\n")
                f.write(f"    * Luki średnie: {col_info['gaps_medium']}\n")
                f.write(f"    * Luki długie: {col_info['gaps_long']}\n")

            f.write(f"\nKolumny usunięte ({len(report['columns_removed'])}):\n")
            for col_info in report['columns_removed']:
                f.write(f"  - {col_info['column']}: {col_info['reason']}\n")

            f.write("\n" + "=" * 60 + "\n")

    print(f" Szczegółowy raport zapisany: {detailed_report_path}")


def main():
    """Główna funkcja"""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING - OBSŁUGA BRAKÓW DANYCH")
    print("=" * 60)

    print("\n Konfiguracja:")
    print(f"   - Krótkie luki (≤{CONFIG['SHORT_GAP_HOURS']}h): interpolacja liniowa")
    print(f"   - Średnie luki (≤{CONFIG['MEDIUM_GAP_HOURS']}h): interpolacja + imputacja historyczna")
    print(f"   - Długie luki (≤{CONFIG['LONG_GAP_HOURS']}h): imputacja historyczna + hourly mean")
    print(f"     Parametry:")
    print(f"       * Okno wyszukiwania: ±{CONFIG['HISTORICAL_WINDOW_DAYS']} dni")
    print(f"       * Lata wstecz: ±{CONFIG['HISTORICAL_YEARS_BACK']} lata")
    print(f"       * Min. wartości historycznych: {CONFIG['MIN_HISTORICAL_VALUES']}")
    print(f"       * Min. wartości hourly mean: {CONFIG['MIN_HOURLY_MEAN_VALUES']}")
    print(f"   - Ekstremalne luki (>{CONFIG['EXTREME_GAP_HOURS']}h): usuń kolumnę")
    print(f"   - Min. próg danych: {CONFIG['MIN_DATA_THRESHOLD'] * 100}%")

    # Znajdź wszystkie pliki
    files = [f for f in os.listdir(MERGED_DATA_FOLDER) if f.endswith('.csv')]

    if not files:
        print(f"\n Brak plików CSV w folderze: {MERGED_DATA_FOLDER}")
        return

    print(f"\n Znaleziono {len(files)} plików do przetworzenia")

    # Przetwarzaj każdy plik
    all_reports = []

    for file in files:
        filepath = os.path.join(MERGED_DATA_FOLDER, file)
        df_processed, report = process_dataset(filepath, CONFIG)
        all_reports.append(report)

    # Utwórz raport zbiorczy
    create_summary_report(all_reports)

    print("\n" + "=" * 60)
    print("PRZETWARZANIE ZAKOŃCZONE")
    print(f"Przetworzone dane: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f" Raporty: {os.path.abspath(REPORTS_FOLDER)}")
    print("=" * 60)


if __name__ == "__main__":
    main()