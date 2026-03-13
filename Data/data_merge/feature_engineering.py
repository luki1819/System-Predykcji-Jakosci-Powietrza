from operator import index

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import holidays

interpolated_data = pd.read_csv('feature_interpolation_data/processed_Wrocław___wyb._Conrada_Korzeniowskiego.csv')
interpolated_data = interpolated_data.drop(columns=['station_id', 'station_name'])
interpolated_data['datetime'] = pd.to_datetime(interpolated_data['datetime'])

print(interpolated_data.head())
print(interpolated_data.shape)
print(interpolated_data.isnull().sum())
print(interpolated_data.columns)


def create_time_features(df):
    df = df.copy()

    df['czas_pomiaru'] = pd.to_datetime(df['datetime'])
    df = df.set_index('czas_pomiaru')  # ustawiam kolumnę jako indeks czasu

    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # przy wysokim ciśnieniu i braku wiatru, jest większe ryzyko smogu
    df['air_stagnation'] = (df['pressure_msl'] / (df['wind_speed_10m'] + 0.1)).round(2)
    # wysoka wilgotność przy niskiej temperaturze sprzyja kondensacji cząstek i mgłom
    df['fog_potential'] = (df['relative_humidity_2m'] * np.maximum(0, (10 - df['temperature_2m']))).round(2)

    # pory roku uproszczone
    def get_season(month):
        if month in [1, 2, 12]:
            return 3  # zima
        elif month in [3, 4, 5]:
            return 0  # wiosna
        elif month in [6, 7, 8]:
            return 1  # lato
        else:
            return 2  # jesień

    df['season'] = df['month'].apply(get_season)

    # generuję listę świąt dla lat z danych
    years = df.index.year.unique()
    polish_holidays = holidays.Poland(years=years)
    polish_holidays = pd.to_datetime(list(polish_holidays.keys()))

    # dodaję kolumnę binarną: czy dzień to dzień wolny od pracy
    df['is_holiday'] = df.index.normalize().isin(polish_holidays).astype(int)

    return df


def show_corelation(df):
    df_numeric = df.select_dtypes(include=[np.number])

    corr = df_numeric.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        center=0,
        linecolor='gray',  # kolor konturu między polami
        linewidths=1  # grubość linii konturu
    )
    plt.title("Macierz korelacji")

    plt.tight_layout()
    plt.show()


def show_mutual_information(df, targets=['dwutlenek azotu', 'pył zawieszony PM10', 'pył zawieszony PM2.5'], top_n=12):
    df = df.copy()
    mi_results = {}

    #bez targetow
    columns_to_drop = targets.copy()

    # bez datetime
    if 'datetime' in df.columns:
        columns_to_drop.append('datetime')


    features_df = df.drop(columns=columns_to_drop)

    features_df = features_df.select_dtypes(include=[np.number])

    print(f"\n Analiza MI dla {len(features_df.columns)} cech numerycznych")
    print(f"lista cech: {list(features_df.columns)}")

    for target in targets:
        if target not in df.columns:
            print(f"brakuje: '{target}' nie ma wiec pomijam")
            continue

        mi = mutual_info_regression(features_df, df[target], random_state=42)
        mi_series = pd.Series(mi, index=features_df.columns)
        mi_results[target] = mi_series

    if not mi_results:
        print("brak wyników coś nie tak")
        return

    top_features = set()
    for target in mi_results.keys():
        top_features.update(mi_results[target].sort_values(ascending=False).head(top_n).index)
    top_features = list(top_features)


    plot_df = pd.DataFrame({
        target: mi_results[target].reindex(top_features)
        for target in mi_results.keys()
    })

    plot_df = plot_df.sort_values(by=list(mi_results.keys())[0], ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    plot_df.plot(kind='barh', ax=ax)

    ax.set_xlabel("Mutual Information")
    ax.set_ylabel("Cechy")
    ax.set_title(f"Top {top_n} cech wg Mutual Information dla pomiarów jakości powietrza")
    ax.grid(True)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    rozszerzone_cechy = create_time_features(interpolated_data)
    print("\n" + "=" * 60)
    print("ROZSZERZONE CECHY")
    print("=" * 60)
    print(rozszerzone_cechy.head())

    show_corelation(rozszerzone_cechy)
    show_mutual_information(rozszerzone_cechy)

    # niech zostanie 12 najlepszych
    rozszerzone_cechy.drop(columns=['is_holiday', 'is_weekend'], inplace=True)

    print("\n" + "=" * 60)
    print("statystyki")
    print("=" * 60)
    print(rozszerzone_cechy.describe())
    print(rozszerzone_cechy.info())

    plt.figure(figsize=(10, 5))
    plt.plot(rozszerzone_cechy.index, rozszerzone_cechy['pył zawieszony PM10'], label='PM10', color='tab:blue')
    plt.xlabel('Czas pomiaru')
    plt.ylabel('Stężenie PM10 [µg/m³]')
    plt.title('Zmienność stężenia PM10 w czasie')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    rozszerzone_cechy_to_save = rozszerzone_cechy.drop(columns=['datetime'], errors='ignore')
    rozszerzone_cechy_to_save.to_csv('../data_for_model/rozszerzone_cechy.csv')
    print("\nnowe cechy zapisane do 'rozszerzone_cechy.csv'")