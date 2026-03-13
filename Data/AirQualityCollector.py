import os
import sqlite3
import pandas as pd
import re
from datetime import datetime

# === KONFIGURACJA ===
FOLDER_Z_DANYMI = "folder_z_pomiarami"  # folder z plikami pomiarów można pobrać ze stronki gioś
PLIK_STACJE = "stacje.csv"  # opcjonalny plik z lokalizacjami
BAZA = "AQPS.db"


# === TWORZENIE BAZY ===
def utworz_baze():
    conn = sqlite3.connect(BAZA)
    cur = conn.cursor()

    # Tabela stacji
    cur.execute("""
                CREATE TABLE IF NOT EXISTS stacje
                (
                    id_stacji      INTEGER PRIMARY KEY AUTOINCREMENT,
                    nazwa          TEXT NOT NULL UNIQUE,
                    lokalizacja    TEXT,
                    szerokosc      REAL,
                    dlugosc        REAL,
                    id_stacji_GIOŚ INTEGER
                );
                """)

    # Tabela sensorów
    cur.execute("""
                CREATE TABLE IF NOT EXISTS sensory
                (
                    id_sensora      INTEGER PRIMARY KEY AUTOINCREMENT,
                    id_stacji       INTEGER NOT NULL,
                    nazwa_sensora   TEXT    NOT NULL,
                    jednostka       TEXT,
                    id_sensora_GIOŚ INTEGER,
                    UNIQUE (id_stacji, nazwa_sensora),
                    FOREIGN KEY (id_stacji) REFERENCES stacje (id_stacji)
                );
                """)

    # Tabela pomiarów - zaktualizowana struktura
    cur.execute("""
                CREATE TABLE IF NOT EXISTS pomiary
                (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    id_sensora    INTEGER NOT NULL,
                    data_pomiaru TEXT    NOT NULL,
                    wartosc      REAL,
                    dodano       TEXT    NOT NULL,
                    UNIQUE (id, data_pomiaru),
                    FOREIGN KEY (id_sensora) REFERENCES sensory (id_sensora)
                );
                """)
    conn.commit()
    conn.close()



def wczytaj_stacje_csv():
    if not os.path.exists(PLIK_STACJE):
        return {}
    df = pd.read_csv(PLIK_STACJE)
    df = df.fillna("")
    stacje_dict = {
        row["nazwa"]: {
            "lokalizacja": row.get("lokalizacja", ""),
            "szerokosc": row.get("szerokosc", None),
            "dlugosc": row.get("dlugosc", None),
        }
        for _, row in df.iterrows()
    }
    return stacje_dict


def normalizuj_date(data_str):
    data_str = str(data_str).strip()

    # różne możliwe formaty dat
    formaty = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%d.%m.%Y %H:%M:%S',
        '%d.%m.%Y %H:%M',
        '%Y/%m/%d %H:%M:%S',
        '%Y/%m/%d %H:%M',
    ]

    for fmt in formaty:
        try:
            dt = datetime.strptime(data_str, fmt)
            # zawsze w formacie z sekundami ustawionymi na :00
            return dt.strftime('%Y-%m-%d %H:%M:00')
        except ValueError:
            continue

    # jeśli żaden nie pasuje to niech będzie ten co jest
    print(f"     nie da sie sparsować daty: {data_str}")
    return data_str


def dodaj_stacje(conn, nazwa, dane_stacji=None):
    cur = conn.cursor()
    cur.execute("SELECT id_stacji FROM stacje WHERE nazwa = ?", (nazwa,))
    wynik = cur.fetchone()
    if wynik:
        return wynik[0]

    lokalizacja = dane_stacji.get("lokalizacja") if dane_stacji else None
    szerokosc = dane_stacji.get("szerokosc") if dane_stacji else None
    dlugosc = dane_stacji.get("dlugosc") if dane_stacji else None

    cur.execute("""
                INSERT INTO stacje (nazwa, lokalizacja, szerokosc, dlugosc)
                VALUES (?, ?, ?, ?)
                """, (nazwa, lokalizacja, szerokosc, dlugosc))
    conn.commit()
    return cur.lastrowid


def dodaj_sensor(conn, id_stacji, nazwa_sensora, jednostka):
    cur = conn.cursor()
    cur.execute("""
                SELECT id_sensora
                FROM sensory
                WHERE id_stacji = ?
                  AND nazwa_sensora = ?
                """, (id_stacji, nazwa_sensora))
    wynik = cur.fetchone()
    if wynik:
        return wynik[0]

    cur.execute("""
                INSERT INTO sensory (id_stacji, nazwa_sensora, jednostka)
                VALUES (?, ?, ?)
                """, (id_stacji, nazwa_sensora, jednostka))
    conn.commit()
    return cur.lastrowid


def dodaj_pomiary_bulk(conn, pomiary_lista):
    """
    Dodaje wiele pomiarów naraz, ale pomija duplikaty
    """
    if not pomiary_lista:
        return 0

    cur = conn.cursor()
    cur.executemany("""
                    INSERT OR IGNORE INTO pomiary (id_sensora, data_pomiaru, wartosc, dodano)
                    VALUES (?, ?, ?, ?)
                    """, pomiary_lista)
    conn.commit()

    # zwróć liczbę rzeczywiście dodanych wierszy
    return cur.rowcount


def parsuj_naglowek(kolumna):
    """
    oddziela te dziwnie sformatowane kolumny
    """
    # ma być Stacja (sensor [jednostka jednostka])
    dopasowanie = re.match(r"(.+?)\s*\((.+?)\s*\[jednostka\s+(.+?)\]\)", kolumna)
    if dopasowanie:
        nazwa_stacji = dopasowanie.group(1).strip()
        nazwa_sensora = dopasowanie.group(2).strip()
        jednostka = dopasowanie.group(3).strip()
        return nazwa_stacji, nazwa_sensora, jednostka

    # a jak nie ma jednostki
    dopasowanie2 = re.match(r"(.+?)\s*\((.+?)\s*\[(.+?)\]\)", kolumna)
    if dopasowanie2:
        nazwa_stacji = dopasowanie2.group(1).strip()
        nazwa_sensora = dopasowanie2.group(2).strip()
        jednostka = dopasowanie2.group(3).strip()
        return nazwa_stacji, nazwa_sensora, jednostka

    print(f"   Nie można sparsować kolumny: {kolumna}")
    return "Nieznana", kolumna, None


# przetwarzanie csv ===
def przetworz_csvy():
    utworz_baze()
    conn = sqlite3.connect(BAZA)
    stacje_z_csv = wczytaj_stacje_csv()

    pliki = [f for f in os.listdir(FOLDER_Z_DANYMI) if f.endswith(".csv")]
    if not pliki:
        print("Brak plików CSV w folderze:", FOLDER_Z_DANYMI)
        return

    for plik in pliki:
        sciezka = os.path.join(FOLDER_Z_DANYMI, plik)
        print(f"\nPrzetwarzanie pliku: {plik}")

        # Wczytanie nagłówka
        with open(sciezka, "r", encoding="utf-8") as f:
            naglowek = f.readline().strip()

        # podział nagłówka po znaku '*'
        kolumny = [x.strip() for x in naglowek.split("*") if x.strip()]

        print(f"   Znalezione kolumny: {len(kolumny)}")

        # Wczytanie danych
        try:
            df = pd.read_csv(
                sciezka,
                skiprows=1,  #pominiecie naglowka
                header=None,
                sep=",",  # separator jest przecinekiem
                engine="python"
            )

            # puste kolumny
            df = df.dropna(axis=1, how='all')

            # ile kolumn jest
            liczba_kolumn_danych = len(df.columns)
            liczba_kolumn_naglowek = len(kolumny)

            print(f"   Kolumny w nagłówku: {liczba_kolumn_naglowek}, w danych: {liczba_kolumn_danych}")

            # Dopasuj liczbę kolumn
            if liczba_kolumn_danych < liczba_kolumn_naglowek:
                print(f"  Mniej kolumn w danych niż w nagłówku używam tylko {liczba_kolumn_danych} kolumn")
                kolumny = kolumny[:liczba_kolumn_danych]
            elif liczba_kolumn_danych > liczba_kolumn_naglowek:
                print(
                    f"    Więcej kolumn w danych ({liczba_kolumn_danych}) niż w nagłówku ({liczba_kolumn_naglowek})")
                # Nie dodaje już ze "Nieznana" - po prostu obcinam nadmiarowe kolumny
                df = df.iloc[:, :liczba_kolumn_naglowek]

            # Przypisz nazwy kolumn
            df.columns = kolumny

        except Exception as e:
            print(f"    Błąd wczytywania pliku: {e}")
            import traceback
            traceback.print_exc()
            continue

        print(f"   Wczytano wierszy: {len(df)}")
        print(f"   Pierwsza kolumna to: '{kolumny[0]}'")
        print(f"   Przykładowa pierwsza data: {df[kolumny[0]].iloc[0] if len(df) > 0 else 'brak'}")

        # gromadzeni pomiarów
        wszystkie_pomiary = []
        dodano = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Przetwarzanie każdej kolumny z danymi (pomijam pierwszą - to data)
        for kolumna in kolumny[1:]:
            nazwa_stacji, nazwa_sensora, jednostka = parsuj_naglowek(kolumna)

            print(f"   ├─ Stacja: {nazwa_stacji}")
            print(f"   │  Sensor: {nazwa_sensora} [{jednostka}]")

            # Dodaj stację
            dane_stacji = stacje_z_csv.get(nazwa_stacji)
            id_stacji = dodaj_stacje(conn, nazwa_stacji, dane_stacji)

            # Dodaj sensor
            id_sensora = dodaj_sensor(conn, id_stacji, nazwa_sensora, jednostka)

            # Zbierz pomiary dla tego sensora
            licznik = 0
            for _, row in df.iterrows():
                data_raw = str(row[kolumny[0]]).strip()  # Data z pierwszej kolumny
                data = normalizuj_date(data_raw)  # NORMALIZACJA DATY
                wartosc = row[kolumna]

                # Sprawdź czy wartość nie jest pusta/NaN/spacja
                if pd.notnull(wartosc):
                    # Konwersja na string i usunięcie białych znaków
                    wartosc_str = str(wartosc).strip()
                    # Sprawdź czy po usunięciu spacji nie jest pusta
                    if wartosc_str and wartosc_str != '':
                        try:
                            wartosc_float = float(wartosc_str)
                            # Dodaj do listy zamiast bezpośrednio do bazy
                            wszystkie_pomiary.append((id_sensora, data, wartosc_float, dodano))
                            licznik += 1
                        except ValueError:
                            # Pomijamy wartości których nie da się skonwertować
                            continue

            print(f"     Przygotowano pomiarów: {licznik}")

        # wszystkie na raz
        if wszystkie_pomiary:
            print(f"\n    Zapisywanie {len(wszystkie_pomiary)} pomiarów do bazy")
            dodano_liczba = dodaj_pomiary_bulk(conn, wszystkie_pomiary)
            pominiete = len(wszystkie_pomiary) - dodano_liczba
            print(f"    Zapisano: {dodano_liczba} nowych")
            if pominiete > 0:
                print(f"     Pominięto: {pominiete} duplikatów")

    conn.close()
    print("\n zaimportowano pomyślnie!!!!!!!")


if __name__ == "__main__":
    os.makedirs(FOLDER_Z_DANYMI, exist_ok=True)
    przetworz_csvy()