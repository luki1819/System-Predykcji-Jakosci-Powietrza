import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="AQPS - System Predykcji Jakości Powietrza",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_PATH = "../Data/AQPS.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def calculate_aqi(pollutant, value):
    if pd.isna(value) or value < 0:
        return None, "Brak danych"

    breakpoints = {
        'pył zawieszony PM2.5': [
            (0, 12, 'Dobra', 0, 50),
            (12, 35, 'Umiarkowana', 51, 100),
            (35, 55, 'Niezdrowa', 101, 150),
            (55, 150, 'Bardzo niezdrowa', 151, 200),
            (150, 500, 'Niebezpieczna', 201, 300)
        ],
        'pył zawieszony PM10': [
            (0, 25, 'Dobra', 0, 50),
            (25, 50, 'Umiarkowana', 51, 100),
            (50, 90, 'Niezdrowa', 101, 150),
            (90, 180, 'Bardzo niezdrowa', 151, 200),
            (180, 600, 'Niebezpieczna', 201, 300)
        ],
        'dwutlenek azotu': [
            (0, 40, 'Dobra', 0, 50),
            (40, 90, 'Umiarkowana', 51, 100),
            (90, 120, 'Niezdrowa', 101, 150),
            (120, 200, 'Bardzo niezdrowa', 151, 200),
            (200, 400, 'Niebezpieczna', 201, 300)
        ]
    }

    if pollutant not in breakpoints:
        return None, "Nieznany"

    for conc_low, conc_high, category, aqi_low, aqi_high in breakpoints[pollutant]:
        if conc_low <= value < conc_high:
            aqi = ((aqi_high - aqi_low) / (conc_high - conc_low)) * (value - conc_low) + aqi_low
            return int(aqi), category

    return 300, 'Niebezpieczna'


def get_activity_recommendations(aqi_value, category):
    """sekcja z rekomendacjami"""

    recommendations = {
        'Dobra': {
            'level': 'excellent',
            'icon': '🟢',
            'title': 'Doskonałe warunki',
            'activities': {
                'Bieganie': {'status': '✅', 'desc': 'Zalecane', 'color': '#00A000'},
                'Jazda na rowerze': {'status': '✅', 'desc': 'Zalecane', 'color': '#00A000'},
                'Spacery': {'status': '✅', 'desc': 'Zalecane', 'color': '#00A000'},
                'Intensywny trening': {'status': '✅', 'desc': 'Zalecane', 'color': '#00A000'}
            },
            'general_advice': 'Idealne warunki do wszystkich form aktywności fizycznej na zewnątrz.',
            'bg_color': '#A5D6A7'
        },
        'Umiarkowana': {
            'level': 'good',
            'icon': '🟡',
            'title': 'Dobre warunki',
            'activities': {
                'Bieganie': {'status': '✅', 'desc': 'Dozwolone', 'color': '#7CB342'},
                'Jazda na rowerze': {'status': '✅', 'desc': 'Dozwolone', 'color': '#7CB342'},
                'Spacery': {'status': '✅', 'desc': 'Zalecane', 'color': '#00A000'},
                'Intensywny trening': {'status': '⚠️', 'desc': 'Z umiarem', 'color': '#FFA000'}
            },
            'general_advice': 'Osoby wrażliwe mogą odczuwać lekki dyskomfort podczas intensywnego wysiłku.',
            'bg_color': '#FFF59D'
        },
        'Niezdrowa': {
            'level': 'moderate',
            'icon': '🟠',
            'title': 'Ograniczone warunki',
            'activities': {
                'Bieganie': {'status': '⚠️', 'desc': 'Ostrożnie', 'color': '#F57C00'},
                'Jazda na rowerze': {'status': '⚠️', 'desc': 'Ostrożnie', 'color': '#F57C00'},
                'Spacery': {'status': '⚠️', 'desc': 'Umiarkowane tempo', 'color': '#FFA000'},
                'Intensywny trening': {'status': '❌', 'desc': 'Niezalecane', 'color': '#CC0000'}
            },
            'general_advice': 'Ogranicz intensywność i czas trwania aktywności. Osoby wrażliwe powinny unikać wysiłku.',
            'bg_color': '#FFCC80'
        },
        'Bardzo niezdrowa': {
            'level': 'poor',
            'icon': '🔴',
            'title': 'Złe warunki',
            'activities': {
                'Bieganie': {'status': '❌', 'desc': 'Niezalecane', 'color': '#CC0000'},
                'Jazda na rowerze': {'status': '❌', 'desc': 'Niezalecane', 'color': '#CC0000'},
                'Spacery': {'status': '⚠️', 'desc': 'Krótkie, wolne tempo', 'color': '#F57C00'},
                'Intensywny trening': {'status': '❌', 'desc': 'Zdecydowanie nie', 'color': '#660066'}
            },
            'general_advice': 'Unikaj aktywności na zewnątrz. Rozważ trening w pomieszczeniach.',
            'bg_color': '#EF9A9A'
        },
        'Niebezpieczna': {
            'level': 'hazardous',
            'icon': '🟣',
            'title': 'Niebezpieczne warunki',
            'activities': {
                'Bieganie': {'status': '❌', 'desc': 'Zabronione', 'color': '#660066'},
                'Jazda na rowerze': {'status': '❌', 'desc': 'Zabronione', 'color': '#660066'},
                'Spacery': {'status': '❌', 'desc': 'Tylko w razie konieczności', 'color': '#CC0000'},
                'Intensywny trening': {'status': '❌', 'desc': 'Zabronione', 'color': '#660066'}
            },
            'general_advice': 'Pozostań w pomieszczeniach. Wszelka aktywność na zewnątrz jest niebezpieczna dla zdrowia.',
            'bg_color': '#CE93D8'
        },
        'Brak danych': {
            'level': 'unknown',
            'icon': '⚪',
            'title': 'Brak danych',
            'activities': {
                'Bieganie': {'status': '❓', 'desc': 'Brak informacji', 'color': '#666666'},
                'Jazda na rowerze': {'status': '❓', 'desc': 'Brak informacji', 'color': '#666666'},
                'Spacery': {'status': '❓', 'desc': 'Brak informacji', 'color': '#666666'},
                'Intensywny trening': {'status': '❓', 'desc': 'Brak informacji', 'color': '#666666'}
            },
            'general_advice': 'Brak danych o jakości powietrza. Zachowaj ostrożność.',
            'bg_color': '#BDBDBD'
        }
    }

    return recommendations.get(category, recommendations['Brak danych'])


def get_aqi_color(category):
    colors = {
        'Dobra': '#00A000',
        'Umiarkowana': '#D4D400',
        'Niezdrowa': '#CC6600',
        'Bardzo niezdrowa': '#CC0000',
        'Niebezpieczna': '#660066',
        'Brak danych': '#666666'
    }
    return colors.get(category, '#666666')


def get_dominant_pollutant_name(pollutant):
    """Zwraca nawę tego zanieczyszczenia które najbardziej wpływa na AQI"""
    names = {
        'pył zawieszony PM2.5': 'PM2.5',
        'pył zawieszony PM10': 'PM10',
        'dwutlenek azotu': 'NO₂'
    }
    return names.get(pollutant, pollutant)


def get_available_stations():
    conn = get_connection()
    query = """
            SELECT DISTINCT s.id_stacji, s.nazwa
            FROM stacje s
                     JOIN sensory sen ON s.id_stacji = sen.id_stacji
                     JOIN pomiary p ON sen.id_sensora = p.id_sensora
            ORDER BY s.nazwa
            """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def get_latest_measurements(station_id, hours_back=48):
    conn = get_connection()
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours_back)

    query = """
            SELECT p.data_pomiaru as timestamp,
               s.nazwa_sensora as pollutant,
               p.wartosc as value
            FROM pomiary p
                JOIN sensory s
            ON p.id_sensora = s.id_sensora
            WHERE s.id_stacji = ?
              AND p.data_pomiaru >= ?
              AND p.data_pomiaru <= ?
            ORDER BY p.data_pomiaru
            """
    df = pd.read_sql(query, conn, params=(station_id, start_time, end_time))
    conn.close()

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    return df


def get_predictions(station_id):
    conn = get_connection()

    query = """
            SELECT predykcja_utworzona,
                   czas_prognozowany,
                   cel_predykcji,
                   horyzont_czasowy,
                   nazwa_modelu,
                   wartosc_predykcji,
                   czas_danych_wejsciowych
            FROM prognozy
            WHERE id_stacji = ?
              AND predykcja_utworzona = (SELECT MAX(predykcja_utworzona)
                                         FROM prognozy
                                         WHERE id_stacji = ?)
            ORDER BY czas_prognozowany
            """
    df = pd.read_sql(query, conn, params=(station_id, station_id))
    conn.close()

    if not df.empty:
        df['predykcja_utworzona'] = pd.to_datetime(df['predykcja_utworzona'])
        df['czas_prognozowany'] = pd.to_datetime(df['czas_prognozowany'])
        df['czas_danych_wejsciowych'] = pd.to_datetime(df['czas_danych_wejsciowych'])

    return df


def get_all_prediction_timestamps(station_id):
    conn = get_connection()
    query = """
            SELECT DISTINCT predykcja_utworzona
            FROM prognozy
            WHERE id_stacji = ?
            ORDER BY predykcja_utworzona DESC LIMIT 168
            """
    df = pd.read_sql(query, conn, params=(station_id,))
    conn.close()

    if not df.empty:
        df['predykcja_utworzona'] = df['predykcja_utworzona'].astype(str)

    return df


def get_historical_predictions(station_id, prediction_timestamps):
    conn = get_connection()

    placeholders = ','.join(['?'] * len(prediction_timestamps))

    query = f"""
            SELECT predykcja_utworzona,
                   czas_prognozowany,
                   cel_predykcji,
                   horyzont_czasowy,
                   nazwa_modelu,
                   wartosc_predykcji
            FROM prognozy
            WHERE id_stacji = ?
              AND predykcja_utworzona IN ({placeholders})
            ORDER BY predykcja_utworzona, czas_prognozowany
            """

    params = [station_id] + prediction_timestamps
    df = pd.read_sql(query, conn, params=params)
    conn.close()

    if not df.empty:
        df['czas_prognozowany'] = pd.to_datetime(df['czas_prognozowany'])
        df['predykcja_utworzona'] = df['predykcja_utworzona'].astype(str)

    return df


def calculate_overall_aqi(measurements_dict):
    aqi_values = []
    worst_category = "Dobra"
    worst_pollutant = None
    worst_aqi = 0

    category_priority = {
        'Dobra': 0,
        'Umiarkowana': 1,
        'Niezdrowa': 2,
        'Bardzo niezdrowa': 3,
        'Niebezpieczna': 4
    }

    for pollutant, value in measurements_dict.items():
        if value is not None:
            aqi, category = calculate_aqi(pollutant, value)
            if aqi:
                aqi_values.append(aqi)
                if (category_priority.get(category, 0) > category_priority.get(worst_category, 0)) or \
                        (category_priority.get(category, 0) == category_priority.get(worst_category,0) and aqi > worst_aqi):
                    worst_category = category
                    worst_pollutant = pollutant
                    worst_aqi = aqi

    if aqi_values:
        return max(aqi_values), worst_category, worst_pollutant
    return None, "Brak danych", None


st.title("System Prognozowania Jakości Powietrza")
st.markdown("### Air Quality Prediction System (AQPS)")

st.sidebar.header("⚙️ Konfiguracja")

stations = get_available_stations()
if stations.empty:
    st.error("❌ Brak dostępnych stacji")
    st.stop()

station_options = {
    f"{row['nazwa']} (ID: {row['id_stacji']})": row['id_stacji']
    for _, row in stations.iterrows()
}

selected_station_name = st.sidebar.selectbox(
    "Wybierz stację:",
    options=list(station_options.keys())
)

selected_station_id = station_options[selected_station_name]

st.sidebar.markdown("---")
st.sidebar.markdown("**Zakres czasowy wykresów:**")
hours_back = st.sidebar.slider(
    "Liczba godzin wstecz:",
    min_value=12,
    max_value=168,
    value=48,
    step=12,
    help="Wybierz zakres danych historycznych do wyświetlenia"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Legenda AQI:**")
st.sidebar.markdown("🟢 **Dobra** (0-50)")
st.sidebar.markdown("🟡 **Umiarkowana** (51-100)")
st.sidebar.markdown("🟠 **Niezdrowa** (101-150)")
st.sidebar.markdown("🔴 **Bardzo niezdrowa** (151-200)")
st.sidebar.markdown("🟣 **Niebezpieczna** (201-300)")

st.sidebar.markdown("---")
st.sidebar.markdown("**Rekomendacje aktywności:**")
st.sidebar.markdown("✅ - Zalecane")
st.sidebar.markdown("⚠️ - Z ostrożnością")
st.sidebar.markdown("❌ - Niezalecane")

df_measurements = get_latest_measurements(selected_station_id, hours_back=hours_back)
df_predictions = get_predictions(selected_station_id)

pollutants_map = {
    'pył zawieszony PM2.5': 'PM2.5',
    'pył zawieszony PM10': 'PM10',
    'dwutlenek azotu': 'NO₂'
}

st.markdown("---")
st.header("Aktualny Stan Jakości Powietrza")

if df_measurements.empty:
    st.warning("⚠️ Brak danych pomiarowych")
    current_values = {}
    overall_aqi = None
    overall_category = "Brak danych"
else:
    latest_data = df_measurements.groupby('pollutant').last()
    current_values = {
        poll: latest_data.loc[poll, 'value'] if poll in latest_data.index else None
        for poll in pollutants_map.keys()
    }
    overall_aqi, overall_category, worst_pollutant = calculate_overall_aqi(current_values)

col1, col2 = st.columns([1, 3])

with col1:
    if overall_aqi:
        color = get_aqi_color(overall_category)
        dominant_name = get_dominant_pollutant_name(worst_pollutant) if worst_pollutant else "N/A"
        st.markdown(
            f"""
            <div style="padding: 20px; background-color: {color}; border-radius: 12px; text-align: center;">
                <p style="color: white; margin: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">Ogólny AQI</p>
                <h1 style="color: white; margin: 0; font-size: 3em; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">{overall_aqi}</h1>
                <h3 style="color: white; margin: 10px 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{overall_category}</h3>
                <hr style="border-color: rgba(255,255,255,0.3); margin: 5px 0;">
                <p style="color: white; margin: 0; font-size: 1.5em; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">Dominujące zanieczyszczenie:</p>
                <p style="color: white; margin: 5px 0 0 0; font-size: 1.5em; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);"><b>{dominant_name}</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("Brak danych AQI")

with col2:
    st.markdown("### Aktualne Stężenia")
    cols = st.columns(3)

    for idx, (pollutant, display_name) in enumerate(pollutants_map.items()):
        value = current_values.get(pollutant)
        if value is not None:
            aqi, category = calculate_aqi(pollutant, value)
            color = get_aqi_color(category)

            with cols[idx]:
                st.markdown(
                    f"""
                    <div style="padding: 15px; background-color: {color}; border-radius: 8px; text-align: center;">
                        <h4 style="color: white; margin: 0;text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{display_name}</h4>
                        <h2 style="color: white; margin: 5px 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{value:.1f} µg/m³</h2>
                        <p style="color: white; margin: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); font-size: 1.5em;">{category}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


st.markdown("---")
st.header("Rekomendacje Aktywności Fizycznej")


recommendations = get_activity_recommendations(overall_aqi, overall_category)

# Główny banner z ogólną rekomendacją
st.markdown(
    f"""
    <div style="padding: 20px; background-color: {recommendations['bg_color']};text-shadow: 1px 1px 2px rgba(0,0,0,0.5); border-radius: 12px; border-left: 5px solid {get_aqi_color(overall_category)};">
        <h3 style="margin: 0 0 10px 0;">{recommendations['icon']} {recommendations['title']}</h3>
        <p style="margin: 0; font-size: 1.5em;text-shadow: 1px 1px 2px rgba(0,0,0,0.5);"><b>{recommendations['general_advice']}</b></p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("")

# szczegółowe rekomendacje dla poszczególnych aktywności - te biegi, spacer, rower i intensywny trening
activity_cols = st.columns(4)

for idx, (activity_name, activity_info) in enumerate(recommendations['activities'].items()):
    with activity_cols[idx]:

        bg_color = activity_info['color'] + '15'  #przezroczystość

        st.markdown(
            f"""
            <div style="padding: 15px; background-color: {bg_color}; border-radius: 8px; border: 3px solid {activity_info['color']}; text-align: center; height: 140px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                <h4 style="margin: 0 0 10px 0; color: white;">{activity_name}</h4>
                <p style="margin: 0; font-size: 2em; line-height: 1;">{activity_info['status']}</p>
                <p style="margin: 5px 0 0 0; color: {activity_info['color']}; font-size: 1.2em; font-weight: bold;">{activity_info['desc']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# doatkowe info żeby wiedzieli co oznaczają te wskazniki
with st.expander("ℹ️ Szczegółowe wskazówki dla aktywnych osób"):
    st.markdown("""
    **Jak interpretować rekomendacje:**

    - **✅ Zalecane**: Możesz swobodnie wykonywać aktywność bez żadnych ograniczeń
    - **⚠️ Z ostrożnością**: Ogranicz intensywność i czas trwania aktywności, monitoruj samopoczucie
    - **❌ Niezalecane**: Unikaj tej aktywności lub przenieś ją do pomieszczenia

    **Grupy szczególnie narażone:**
    - Osoby z chorobami układu oddechowego (astma, POChP)
    - Osoby z chorobami serca
    - Dzieci i osoby starsze
    - Kobiety w ciąży

    **Zalecenia dodatkowe:**
    - Monitoruj prognozy i planuj aktywności w godzinach o najlepszej jakości powietrza
    - Przy złej jakości powietrza rozważ trening w pomieszczeniach (siłownia, basen)
    - Pij dużo wody podczas aktywności
    - Jeśli odczuwasz dyskomfort (kaszel, duszności, pieczenie oczu), przerwij aktywność
    """)

st.markdown("---")
st.header("Prognozy")
if not df_predictions.empty:
    latest_prediction_time = df_predictions['predykcja_utworzona'].iloc[0]
    input_data_time = df_predictions['czas_danych_wejsciowych'].iloc[0]

    st.info(f"📅 **Prognozy wygenerowane:** {latest_prediction_time.strftime('%Y-%m-%d %H:%M')}") # info z której pochodzą te predykcje

if df_predictions.empty:
    st.warning("⚠️ Brak prognoz")
else:
    df_pred_pm25 = df_predictions[df_predictions['cel_predykcji'] == 'pył zawieszony PM2.5']
    df_pred_pm10 = df_predictions[df_predictions['cel_predykcji'] == 'pył zawieszony PM10']
    df_pred_no2 = df_predictions[df_predictions['cel_predykcji'] == 'dwutlenek azotu']

    horizon_names = {
        '1h': '+1h',
        '3h': '+3h',
        '6h': '+6h',
        '12h': '+12h'
    }

    horizons_to_display = ['1h', '3h', '6h', '12h']
    cols = st.columns(len(horizons_to_display))

    for idx, horizon in enumerate(horizons_to_display):
        pred_values = {}

        for target in pollutants_map.keys():
            df_target = df_predictions[df_predictions['cel_predykcji'] == target]
            df_horizon = df_target[df_target['horyzont_czasowy'] == horizon]

            if not df_horizon.empty:
                pred_values[target] = df_horizon.iloc[0]['wartosc_predykcji']

        pred_aqi, pred_category, pred_worst = calculate_overall_aqi(pred_values)

        with cols[idx]:
            if pred_aqi:
                color = get_aqi_color(pred_category)
                dominant_pred_name = get_dominant_pollutant_name(pred_worst) if pred_worst else "N/A"

                # ikonki co wskazują na wzrost itp a ta w bok to nwm chyba zostawie
                pred_recommendations = get_activity_recommendations(pred_aqi, pred_category)
                activity_icon = pred_recommendations['icon']

                if overall_aqi:
                    diff = pred_aqi - overall_aqi
                    if diff > 2:
                        arrow = "↑"
                        arrow_color = "#FF6B6B"
                    elif diff < -2:
                        arrow = "↓"
                        arrow_color = "#51CF66"
                    else:
                        arrow = "→"
                        arrow_color = "#FFD93D"
                    diff_text = f"{arrow} {abs(diff)}"
                else:
                    diff_text = ""
                    arrow_color = "#999"

                st.markdown(
                    f"""
                    <div style="padding: 15px; background-color: {color}; border-radius: 8px; text-align: center;">
                        <h4 style="color: white; margin: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{horizon_names[horizon]}</h4>
                        <h2 style="color: white; margin: 8px 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); ">AQI: {pred_aqi}</h2>
                        <p style="color: white; margin: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); font-size: 1.5em;">{pred_category}</p>
                        <p style="color: {arrow_color}; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); margin: 5px 0; font-size: 1.5em; font-weight: bold;">{diff_text}</p>
                        <hr style="border-color: rgba(255,255,255,0.3); margin: 10px 0;">
                        <p style="color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); margin: 0; font-size: 1.5em;">Dominujące: {dominant_pred_name}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

st.markdown("---")
st.header("Szczegółowe Wykresy")

show_historical = st.checkbox("📊 Pokaż archiwalne prognozy", value=False)

historical_horizons = []
historical_predictions_df = pd.DataFrame()

if show_historical:
    st.markdown("**Wybierz horyzonty czasowe do porównania:**")
    col_h1, col_h2, col_h3, col_h4 = st.columns(4)

    with col_h1:
        show_1h = st.checkbox("+1h", value=True, key="hist_1h")
        if show_1h:
            historical_horizons.append('1h')
    with col_h2:
        show_3h = st.checkbox("+3h", value=False, key="hist_3h")
        if show_3h:
            historical_horizons.append('3h')
    with col_h3:
        show_6h = st.checkbox("+6h", value=False, key="hist_6h")
        if show_6h:
            historical_horizons.append('6h')
    with col_h4:
        show_12h = st.checkbox("+12h", value=False, key="hist_12h")
        if show_12h:
            historical_horizons.append('12h')

    if historical_horizons:
        df_timestamps = get_all_prediction_timestamps(selected_station_id)

        if not df_timestamps.empty:
            timestamps_raw = df_timestamps['predykcja_utworzona'].tolist()
            historical_predictions_df = get_historical_predictions(selected_station_id, timestamps_raw)

            if not historical_predictions_df.empty:
                cutoff_time = datetime.now() - timedelta(hours=hours_back)
                historical_predictions_df = historical_predictions_df[
                    historical_predictions_df['czas_prognozowany'] >= cutoff_time
                    ]

                if historical_predictions_df.empty:
                    st.warning("⚠️ Brak archiwalnych prognoz w wybranym zakresie czasowym.")
                else:
                    st.success(
                        f"✅ Wczytano {len(historical_predictions_df)} punktów archiwalnych prognoz (zakres {hours_back}h)")
        else:
            st.warning("⚠️ Brak historii prognoz w bazie")

tab1, tab2, tab3 = st.tabs(["PM2.5", "PM10", "NO₂"])


def create_pollutant_chart(df_measurements, df_predictions, pollutant, display_name,
                           show_historical=False, historical_df=pd.DataFrame(),
                           selected_horizons=[]):
    fig = go.Figure()

    hist_data = df_measurements[df_measurements['pollutant'] == pollutant].sort_values('timestamp')
    if not hist_data.empty:
        fig.add_trace(go.Scatter(
            x=hist_data['timestamp'],
            y=hist_data['value'],
            mode='lines',
            name='Pomiary',
            line=dict(color='#1f77b4', width=2)
        ))

    if not df_predictions.empty:
        pred_data = df_predictions.sort_values('czas_prognozowany').groupby('czas_prognozowany').first().reset_index()

        fig.add_trace(go.Scatter(
            x=pred_data['czas_prognozowany'],
            y=pred_data['wartosc_predykcji'],
            mode='lines+markers',
            name='Prognoza (Aktualna)',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))

        if not hist_data.empty:
            fig.add_trace(go.Scatter(
                x=[hist_data.iloc[-1]['timestamp'], pred_data.iloc[0]['czas_prognozowany']],
                y=[hist_data.iloc[-1]['value'], pred_data.iloc[0]['wartosc_predykcji']],
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))

    all_values_for_scale = []
    if not hist_data.empty: all_values_for_scale.extend(hist_data['value'].tolist())
    if not df_predictions.empty: all_values_for_scale.extend(pred_data['wartosc_predykcji'].tolist())

    if show_historical and not historical_df.empty and selected_horizons:
        hist_preds_pollutant = historical_df[historical_df['cel_predykcji'] == pollutant].copy()

        if not hist_preds_pollutant.empty:
            all_values_for_scale.extend(hist_preds_pollutant['wartosc_predykcji'].tolist())

            horizon_colors = {'1h': '#e377c2', '3h': '#bcbd22', '6h': '#17becf', '12h': '#8c564b'}

            hist_preds_pollutant['ts_str'] = hist_preds_pollutant['predykcja_utworzona'].astype(str)

            for horizon in selected_horizons:
                horizon_data = hist_preds_pollutant[hist_preds_pollutant['horyzont_czasowy'] == horizon].copy()
                horizon_data = horizon_data.sort_values('czas_prognozowany')

                if horizon_data.empty:
                    continue

                fig.add_trace(go.Scatter(
                    x=horizon_data['czas_prognozowany'],
                    y=horizon_data['wartosc_predykcji'],
                    mode='lines+markers',
                    line=dict(
                        dash='dash',
                        width=1.5,
                        color=horizon_colors.get(horizon, '#999')
                    ),
                    marker=dict(
                        size=6,
                        symbol='circle',
                        color=horizon_colors.get(horizon, '#999'),
                        line=dict(width=1, color='white'),
                        opacity=0.8
                    ),
                    connectgaps=True,
                    name=f'Arch. {horizon}',
                    customdata=horizon_data['ts_str'],
                    hovertemplate='<b>%{y:.1f} µg/m³</b><br>' +
                                  'Czas prog.: %{x|%Y-%m-%d %H:%M}<br>' +
                                  f'Horyzont: {horizon}<br>' +
                                  'Utworzono: %{customdata}<extra></extra>'
                ))

    if 'PM2.5' in pollutant:
        thresholds = [12, 35, 55, 150]
    elif 'PM10' in pollutant:
        thresholds = [25, 50, 90, 180] #zgodne musi byc
    else:
        thresholds = [40, 90, 120, 230] # z gioś normy

    colors = ['rgba(0, 160, 0, 0.1)', 'rgba(212, 212, 0, 0.1)',
              'rgba(204, 102, 0, 0.1)', 'rgba(204, 0, 0, 0.1)']
    labels = ['Dobra', 'Umiarkowana', 'Niezdrowa', 'Bardzo niezdrowa']

    for i, (thr, col, lab) in enumerate(zip(thresholds, colors, labels)):
        prev = 0 if i == 0 else thresholds[i - 1]
        fig.add_hrect(y0=prev, y1=thr, fillcolor=col, layer="below", line_width=0,
                      annotation_text=lab, annotation_position="top left")

    if all_values_for_scale:
        valid_values = [x for x in all_values_for_scale if pd.notnull(x)]
        if valid_values:
            y_max = max(valid_values) * 1.1
        else:
            y_max = 50
    else:
        y_max = 50

    fig.update_layout(
        title=f"{display_name}",
        xaxis_title="Czas",
        yaxis_title="Stężenie [µg/m³]",
        yaxis=dict(
            range=[0, y_max],
            fixedrange=False
        ),
        hovermode='closest',
        height=450,
        template='plotly_white', # motyw wykresu
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_predictions_table(df_predictions, pollutant):
    """Tworzy tabelkę z wartościami prognoz dla różnych horyzontów"""
    if df_predictions.empty:
        return pd.DataFrame()

    pollutant_preds = df_predictions[df_predictions['cel_predykcji'] == pollutant]

    if pollutant_preds.empty:
        return pd.DataFrame()

    horizons = ['1h', '3h', '6h', '12h']
    table_data = []

    for horizon in horizons:
        horizon_data = pollutant_preds[pollutant_preds['horyzont_czasowy'] == horizon]
        if not horizon_data.empty:
            value = horizon_data.iloc[0]['wartosc_predykcji']
            time = horizon_data.iloc[0]['czas_prognozowany']
            aqi, category = calculate_aqi(pollutant, value)

            table_data.append({
                'Horyzont': f'+{horizon}',
                'Czas': time.strftime('%Y-%m-%d %H:%M'),
                'Wartość [µg/m³]': f'{value:.2f}',
                'AQI': aqi if aqi else 'N/A',
                'Kategoria': category
            })

    return pd.DataFrame(table_data)

#tabelki jak by ktoś chciał dokładnie zobaczyc

with tab1:
    fig_pm25 = create_pollutant_chart(
        df_measurements, df_pred_pm25,
        'pył zawieszony PM2.5', 'PM2.5',
        show_historical, historical_predictions_df, historical_horizons
    )
    st.plotly_chart(fig_pm25, width='stretch')

    if not df_pred_pm25.empty:
        st.markdown("**📋 Szczegółowe wartości prognoz PM2.5:**")
        table_pm25 = create_predictions_table(df_pred_pm25, 'pył zawieszony PM2.5')
        if not table_pm25.empty:
            st.dataframe(table_pm25, width='stretch', hide_index=True)

with tab2:
    fig_pm10 = create_pollutant_chart(
        df_measurements, df_pred_pm10,
        'pył zawieszony PM10', 'PM10',
        show_historical, historical_predictions_df, historical_horizons
    )
    st.plotly_chart(fig_pm10, width='stretch')

    if not df_pred_pm10.empty:
        st.markdown("**📋 Szczegółowe wartości prognoz PM10:**")
        table_pm10 = create_predictions_table(df_pred_pm10, 'pył zawieszony PM10')
        if not table_pm10.empty:
            st.dataframe(table_pm10, width='stretch', hide_index=True)

with tab3:
    fig_no2 = create_pollutant_chart(
        df_measurements, df_pred_no2,
        'dwutlenek azotu', 'NO₂',
        show_historical, historical_predictions_df, historical_horizons
    )
    st.plotly_chart(fig_no2, width='stretch')

    if not df_pred_no2.empty:
        st.markdown("**📋 Szczegółowe wartości prognoz NO₂:**")
        table_no2 = create_predictions_table(df_pred_no2, 'dwutlenek azotu')
        if not table_no2.empty:
            st.dataframe(table_no2, width='stretch', hide_index=True)

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p><b>AQPS</b> - System Prognozowania Jakości Powietrza</p>
        <p>Dane: GIOŚ + OpenMeteo | Model: XGBoost</p>
    </div>
    """,
    unsafe_allow_html=True
)