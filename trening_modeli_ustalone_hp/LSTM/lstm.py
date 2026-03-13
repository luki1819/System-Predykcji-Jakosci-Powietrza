import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
import pickle
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class UnscaledMetricsCallback(Callback):
    """Wyświetla metryki w µg/m³ podczas treningu"""

    def __init__(self, X_val, y_val, scaler_y, horizons, print_every=5):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.scaler_y = scaler_y
        self.horizons = horizons
        self.print_every = print_every
        self.history_unscaled = {
            'epoch': [],
            'avg_mae_unscaled': [],
            'mae_per_horizon': {h: [] for h in horizons}
        }

    def on_epoch_end(self, epoch, logs=None):
        self.history_unscaled['epoch'].append(epoch + 1)
        y_pred_scaled = self.model.predict(self.X_val, verbose=0)
        y_val_original = self.scaler_y.inverse_transform(self.y_val)
        y_pred_original = self.scaler_y.inverse_transform(y_pred_scaled)

        mae_values = []
        for i, h in enumerate(self.horizons):
            mae = mean_absolute_error(y_val_original[:, i], y_pred_original[:, i])
            mae_values.append(mae)
            self.history_unscaled['mae_per_horizon'][h].append(mae)

        avg_mae = np.mean(mae_values)
        self.history_unscaled['avg_mae_unscaled'].append(avg_mae)

        if (epoch + 1) % self.print_every == 0 or epoch == 0:
            print(f"\n  → Epoch {epoch + 1}: Val MAE (µg/m³): {avg_mae:.2f} ", end="")
            print(f"[{', '.join([f'{m:.1f}' for m in mae_values])}]", end="")


class LSTMPredictor:
    def __init__(self, csv_path, params, target_name='pył zawieszony PM2.5',
                 horizons=[1, 3, 6, 12], train_split=0.7, val_split=0.15,
                 output_dir='results_lstm_manual'):
        self.csv_path = csv_path
        self.params = params
        self.target_name = target_name
        self.horizons = horizons
        self.train_split = train_split
        self.val_split = val_split
        self.output_dir = output_dir

        self.train_df = None
        self.val_df = None
        self.test_df = None

        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)

    def load_data(self):
        print("=" * 80)
        print("ŁADOWANIE DANYCH")
        print("=" * 80)

        df = pd.read_csv(self.csv_path, index_col=0, parse_dates=True)
        df = df.sort_index()

        total = len(df)
        train_end = int(total * self.train_split)
        val_end = int(total * (self.train_split + self.val_split))

        self.train_df = df.iloc[:train_end].copy()
        self.val_df = df.iloc[train_end:val_end].copy()
        self.test_df = df.iloc[val_end:].copy()

        # pokazywaniezakresów dat
        print(f"Train range: {self.train_df.index.min()}  ->  {self.train_df.index.max()} (n={len(self.train_df)})")
        print(f"Val range:   {self.val_df.index.min()}  ->  {self.val_df.index.max()} (n={len(self.val_df)})")
        print(f"Test range:  {self.test_df.index.min()}  ->  {self.test_df.index.max()} (n={len(self.test_df)})")

        print(f"\nTarget: {self.target_name}")
        print(f"Horyzonty: {self.horizons}h")
        print("=" * 80)

    def prepare_sequences(self, df, window_size):
        X_raw = df.values
        y_raw = df[self.target_name].values

        if not hasattr(self.scaler_X, 'n_features_in_'):
            train_X = self.train_df.values
            train_y = self.train_df[self.target_name].values
            self.scaler_X.fit(train_X)
            self.scaler_y.fit(train_y.reshape(-1, 1))

        X_scaled = self.scaler_X.transform(X_raw)
        y_scaled = self.scaler_y.transform(y_raw.reshape(-1, 1)).flatten()

        X = []
        y = []
        max_horizon = max(self.horizons)

        for i in range(len(X_scaled) - window_size - max_horizon + 1):
            X.append(X_scaled[i:(i + window_size)])
            y_horizons = [y_scaled[i + window_size + h - 1] for h in self.horizons]
            y.append(y_horizons)

        return np.array(X), np.array(y)

    def build_model(self, window_size, n_features):
        inputs = Input(shape=(window_size, n_features))

        x = LSTM(self.params['lstm_units'], return_sequences=False)(inputs)
        x = Dropout(self.params['dropout_rate'])(x)

        x = Dense(self.params['dense_units'], activation='relu')(x)
        x = Dropout(self.params['dropout_rate'] / 2)(x)

        outputs = Dense(len(self.horizons), name='multi_horizon_output')(x)

        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=self.params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

        return model

    def train(self, epochs=100):
        print("\n" + "=" * 80)
        print("TRENING MODELU (PARAMETRY SZTYWNE)")
        print("=" * 80)

        window_size = self.params['window_size']
        batch_size = self.params['batch_size']

        # Zapisz parametry do JSON
        with open(f"{self.output_dir}/parameters.json", 'w') as f:
            json.dump(self.params, f, indent=4)

        X_train, y_train = self.prepare_sequences(self.train_df, window_size)
        X_val, y_val = self.prepare_sequences(self.val_df, window_size)
        X_test, y_test = self.prepare_sequences(self.test_df, window_size)

        n_features = X_train.shape[2]
        model = self.build_model(window_size, n_features)
        model.summary()

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )

        unscaled_callback = UnscaledMetricsCallback(
            X_val=X_val,
            y_val=y_val,
            scaler_y=self.scaler_y,
            horizons=self.horizons,
            print_every=5
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, unscaled_callback],
            verbose=1
        )

        model.save(f"{self.output_dir}/final_model.keras")

        with open(f"{self.output_dir}/scalers.pkl", 'wb') as f:
            pickle.dump({'scaler_X': self.scaler_X, 'scaler_y': self.scaler_y}, f)

        self._plot_training_history(history, unscaled_callback.history_unscaled)
        return model, X_test, y_test

    def _plot_training_history(self, history, history_unscaled):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_title('Proces Uczenia (Scaled)', fontsize=13)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        epochs = history_unscaled['epoch']
        mae_unscaled = history_unscaled['avg_mae_unscaled']
        axes[1].plot(epochs, mae_unscaled, linewidth=2, color='orange')
        axes[1].axhline(y=min(mae_unscaled), color='green', linestyle='--', label=f'Best: {min(mae_unscaled):.2f}')
        axes[1].set_title('Walidacja (Unscaled MAE)', fontsize=13)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/training_history.png", dpi=300)
        plt.close()

    def evaluate(self, model, X_test, y_test):
        print("\n" + "=" * 80)
        print("EWALUACJA NA TEST SET")
        print("=" * 80)

        y_pred_scaled = model.predict(X_test, verbose=0)
        y_test_original = self.scaler_y.inverse_transform(y_test)
        y_pred_original = self.scaler_y.inverse_transform(y_pred_scaled)

        metrics = {}
        metrics_json = {}

        for i, h in enumerate(self.horizons):
            y_true = y_test_original[:, i]
            y_pred = y_pred_original[:, i]

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100

            # do wykresów
            metrics[h] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                'y_true': y_true,
                'y_pred': y_pred
            }

            # Do zapisu JSON (tylko liczby)
            metrics_json[f'{h}h'] = {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'R2': float(r2),
                'MAPE': float(mape)
            }

        # liczenie średnich dla horyzontów
        avg_mae = np.mean([metrics[h]['MAE'] for h in self.horizons])
        avg_rmse = np.mean([metrics[h]['RMSE'] for h in self.horizons])
        avg_r2 = np.mean([metrics[h]['R2'] for h in self.horizons])
        avg_mape = np.mean([metrics[h]['MAPE'] for h in self.horizons])

        metrics_json['average'] = {
            'MAE': float(avg_mae),
            'RMSE': float(avg_rmse),
            'R2': float(avg_r2),
            'MAPE': float(avg_mape)
        }

        # Zapis do JSON
        with open(f"{self.output_dir}/test_metrics.json", 'w') as f:
            json.dump(metrics_json, f, indent=4)

        # TABELKI W KONSOLI (Pandas)
        print("\nPODSUMOWANIE PARAMETRÓW MODELU:")
        params_df = pd.DataFrame(list(self.params.items()), columns=['Parametr', 'Wartość'])
        print(params_df.to_string(index=False))

        print("\nWYNIKI METRYK NA ZBIORZE TESTOWYM:")
        results_list = []
        for h in self.horizons:
            row = {'Horyzont': f"{h}h"}
            row.update({k: round(v, 4) for k, v in metrics[h].items() if k not in ['y_true', 'y_pred']})
            results_list.append(row)

        # Dodaj wiersz średniej
        avg_row = {'Horyzont': 'Średnia'}
        avg_row.update(metrics_json['average'])
        # Zaokrąglenie średniej dla ładnego wyświetlania
        for k in avg_row:
            if k != 'Horyzont': avg_row[k] = round(avg_row[k], 4)
        results_list.append(avg_row)

        results_df = pd.DataFrame(results_list)
        # Ustaw horyzont jako pierwszą kolumnę
        cols = ['Horyzont', 'MAE', 'RMSE', 'R2', 'MAPE']
        results_df = results_df[cols]
        print(results_df.to_string(index=False))

        print(f"\n✓ Metryki zapisane: {self.output_dir}/test_metrics.json")
        print(f"✓ Parametry zapisane: {self.output_dir}/parameters.json")

        self._plot_metrics(metrics)
        self._plot_predictions(metrics)
        return metrics

    def _plot_metrics(self, metrics):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        horizons_list = list(self.horizons)

        # Helper function for bar plots
        def plot_bar(ax, key, title, color):
            values = [metrics[h][key] for h in horizons_list]
            ax.bar(range(len(horizons_list)), values, color=color, alpha=0.7)
            ax.set_xticks(range(len(horizons_list)))
            ax.set_xticklabels([f'{h}h' for h in horizons_list])
            ax.set_title(title)
            for i, v in enumerate(values):
                ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')

        plot_bar(axes[0, 0], 'MAE', 'MAE (µg/m³)', 'skyblue')
        plot_bar(axes[0, 1], 'RMSE', 'RMSE (µg/m³)', 'lightcoral')
        plot_bar(axes[1, 0], 'R2', 'R² Score', 'lightgreen')
        plot_bar(axes[1, 1], 'MAPE', 'MAPE (%)', 'plum')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/metrics_comparison.png", dpi=300)
        plt.close()

    def _plot_predictions(self, metrics):
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()

        for i, h in enumerate(self.horizons):
            y_true = metrics[h]['y_true']
            y_pred = metrics[h]['y_pred']
            n_samples = min(500, len(y_true))

            axes[i].plot(y_true[:n_samples], label='Rzeczywiste', linewidth=2, alpha=0.7, color='navy')
            axes[i].plot(y_pred[:n_samples], label='Przewidywane', linewidth=2, alpha=0.7, color='orange')
            axes[i].set_title(f'Horyzont {h}h (MAE: {metrics[h]["MAE"]:.2f})')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/predictions_vs_actual.png", dpi=300)
        plt.close()


def main():

    CSV_PATH = '../../Data/data_for_model/rozszerzone_cechy.csv'
    TARGET = 'dwutlenek azotu'  # tutaj można wybierać cel predykcji
    OUTPUT_DIR = 'results_lstm_dwutlenek_azotu' # tu można zmieniać nazwę folderu z modelem

    # PARAMETRY NA SZTYWNO
    FIXED_PARAMS = {
        'window_size': 48,
        'batch_size': 64,
        'lstm_units': 32,
        'dense_units': 48,
        'dropout_rate': 0.1,
        'learning_rate': 0.0005
    }

# ========================================================================

    predictor = LSTMPredictor(
        csv_path=CSV_PATH,
        params=FIXED_PARAMS,
        target_name=TARGET,
        output_dir=OUTPUT_DIR
    )

    predictor.load_data()
    model, X_test, y_test = predictor.train(epochs=40)  # można dać więcej epok
    metrics = predictor.evaluate(model, X_test, y_test)

    print(f"\nGotowe! Wyniki zapisane w: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()