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
import optuna
import pickle
import json
import os

"""
TEN SKRYPT ZOSTAŁ UŻYTY DO WYBORU HIPERPARAMETRÓW modelu z jedną WARSTWĄ LSTM
"""

# żeby nie spamowało warningami
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

#style wykresów
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class UnscaledMetricsCallback(Callback):
    """żeby widoczne były błędy w normalnej skali, a nie jako wartosci z przediało od 0 do1"""

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



#klasa do optymalizacji hiperparametrów
class LSTMOptymalizator:
    """

    """

    def __init__(self,
                 csv_path,
                 target_name='pył zawieszony PM10',
                 horizons=[1, 3, 6, 12],
                 train_split=0.7,
                 val_split=0.15,
                 output_dir='results_lstm_1layer'):

        self.csv_path = csv_path
        self.target_name = target_name
        self.horizons = horizons
        self.train_split = train_split
        self.val_split = val_split
        self.output_dir = output_dir

        self.train_df = None
        self.val_df = None
        self.test_df = None

        self.scaler_X = MinMaxScaler(feature_range=(0, 1))# wymagane skalowanie dla LSTM
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))# wymagane skalowanie dla LSTM

        self.best_params = None

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)

    def load_data(self):
        """Wczytaj dane"""
        print("Wczytywanie danych")

        df = pd.read_csv(self.csv_path, index_col=0, parse_dates=True)
        df = df.sort_index() # ponowne sortowanie żeby była pewność że chronologicznie

        print(f"Rekordów: {len(df)}")
        print(f"Cech: {len(df.columns)}")
        print(f"Target: {self.target_name}")
        print(f"Horyzonty: {self.horizons}h")

        total = len(df)
        train_end = int(total * self.train_split)
        val_end = int(total * (self.train_split + self.val_split))

        self.train_df = df.iloc[:train_end].copy()
        self.val_df = df.iloc[train_end:val_end].copy()
        self.test_df = df.iloc[val_end:].copy()

        print(f"\nTrain: {len(self.train_df)}")
        print(f"Val:   {len(self.val_df)}")
        print(f"Test:  {len(self.test_df)}")
        print("Testowy tylko do końcowej oceny")
        print("----------------------------------------------")

    def prepare_sequences(self, df, window_size):
        """przygotowywanie sekwencji zgodnie z wybranym oknem czasowym"""
        X_raw = df.values
        y_raw = df[self.target_name].values

        if not hasattr(self.scaler_X, 'n_features_in_'):
            train_X = self.train_df.values
            train_y = self.train_df[self.target_name].values
            self.scaler_X.fit(train_X)
            self.scaler_y.fit(train_y.reshape(-1, 1))

        X_scaled = self.scaler_X.transform(X_raw) #skalowanie
        y_scaled = self.scaler_y.transform(y_raw.reshape(-1, 1)).flatten()#skalowanie

        X = []
        y = []
        max_horizon = max(self.horizons)

        for i in range(len(X_scaled) - window_size - max_horizon + 1):
            X.append(X_scaled[i:(i + window_size)])
            y_horizons = [y_scaled[i + window_size + h - 1] for h in self.horizons]
            y.append(y_horizons)

        return np.array(X), np.array(y)

    def build_model(self, trial, window_size, n_features):
        """tutaj są określone wartości które optuna może testować"""
        lstm_units = trial.suggest_categorical('lstm_units', [32, 48, 64, 96, 128, 160])

        dropout_rate = trial.suggest_categorical('dropout_rate', [0.1, 0.2, 0.3])

        dense_units = trial.suggest_categorical('dense_units', [16, 32, 48, 64])

        learning_rate = trial.suggest_categorical('learning_rate', [0.0003, 0.0005, 0.001, 0.002])

        inputs = Input(shape=(window_size, n_features))
        x = LSTM(lstm_units, return_sequences=False)(inputs)
        x = Dropout(dropout_rate)(x)
        x = Dense(dense_units, activation='relu')(x)
        x = Dropout(dropout_rate / 2)(x)

        outputs = Dense(len(self.horizons), name='multi_horizon_output')(x)

        model = Model(inputs=inputs, outputs=outputs)

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

        return model

    def objective(self, trial):
        """Funkcja celu dla Optuna"""

        # w lietraturze było od 1 do 3 dni
        window_size = trial.suggest_categorical('window_size', [24, 48])
        batch_size = trial.suggest_categorical('batch_size', [32, 64])

        try:
            X_train, y_train = self.prepare_sequences(self.train_df, window_size)
            X_val, y_val = self.prepare_sequences(self.val_df, window_size)
        except:
            return float('inf')

        if len(X_train) < 100:
            return float('inf')

        n_features = X_train.shape[2]
        model = self.build_model(trial, window_size, n_features)


        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=40, #dam na 40 niech się uczy na każdym do końca
            restore_best_weights=True, #zwraca najlepsze wagi
            verbose=0
        )

        try:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )

            # MAE ale w normalnych jednostkach nie skalowane
            y_pred_scaled = model.predict(X_val, verbose=0)
            y_val_original = self.scaler_y.inverse_transform(y_val)
            y_pred_original = self.scaler_y.inverse_transform(y_pred_scaled)

            mae_unscaled = np.mean([
                mean_absolute_error(y_val_original[:, i], y_pred_original[:, i])
                for i in range(len(self.horizons))
            ])

            return mae_unscaled

        except Exception as e:
            print(f"próba nieudana failed: {e}")
            return float('inf')

    def run_optimization(self, n_trials=20):
        #jak niżej nie podane to bierz 20 prob
        """Start pocesu optymalizcaji"""

        study = optuna.create_study(
            direction='minimize',
            study_name='lstm_fixed_optimization'
        )

        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        print("---------------------------------")
        print("KONIEC optymalizacji")

        print(f"\nLiczba trials: {len(study.trials)}")
        print(f"Najlepszy MAE: {study.best_value:.2f} µg/m³")

        print(f"\nNajlepsze hiperparametry:")
        for key, value in study.best_params.items():
            print(f"  {key:20s}: {value}")

        self.best_params = study.best_params

        # Zapisz
        with open(f"{self.output_dir}/best_params.json", 'w') as f:
            json.dump(study.best_params, f, indent=4)

        with open(f"{self.output_dir}/study.pkl", 'wb') as f:
            pickle.dump(study, f)

        print(f"Wyniki zapisane w {self.output_dir}/")
        print('--------------------------------------------')
        self._plot_optimization(study)

        return study

    def _plot_optimization(self, study):
        # historia optymalizacji
        fig, ax = plt.subplots(figsize=(12, 6))

        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        values = [t.value for t in trials]

        # pokazanie najlepszych do tej pory
        best_so_far = []
        current_best = float('inf')
        for v in values:
            if v < current_best:
                current_best = v
            best_so_far.append(current_best)

        ax.plot(values, marker='o', linestyle='-', linewidth=1.5, markersize=4,
                alpha=0.6, label='Trial value')
        ax.plot(best_so_far, linewidth=2.5, color='red', label='najlepsze')
        ax.axhline(y=study.best_value, color='green', linestyle='--',
                   label=f'Best: {study.best_value:.2f} µg/m³')

        ax.set_xlabel('Próba', fontsize=12)
        ax.set_ylabel('MAE (µg/m³)', fontsize=12)
        ax.set_title('LSTM - Historia Optymalizacji', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/optimization_history.png", dpi=300)
        print(f"✓ Wykres: optimization_history.png")
        plt.close()

        # wykres ważności hiperparametrów
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            importances = optuna.importance.get_param_importances(study)
            params = list(importances.keys())
            values = list(importances.values())

            ax.barh(params, values, color='skyblue', edgecolor='navy')
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title('LSTM - Ważność Hiperparametrów', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/plots/param_importances.png", dpi=300)
            print(f"✓ Wykres: param_importances.png")
            plt.close()
        except:
            print("nie udało się zrobić wykresu ważności hiperparametrów")

    def train_final_model(self, epochs=100):

        print("\n" + "=" * 80)
        print("TRENING FINALNEGO MODELU (POPRAWIONY)")
        print("=" * 80)

        window_size = self.best_params['window_size']
        batch_size = self.best_params['batch_size']

        print(f"\nWindow size: {window_size}")
        print(f"Batch size: {batch_size}")

        print(f"zbiory wyglądają następująco:")
        print(f"   - Trening na: train_df ({len(self.train_df)} próbek)")
        print(f"   - Walidacja na: val_df ({len(self.val_df)} próbek)")
        print(f"   - Test (TYLKO ewaluacja): test_df ({len(self.test_df)} próbek)")

        X_train, y_train = self.prepare_sequences(self.train_df, window_size)
        X_val, y_val = self.prepare_sequences(self.val_df, window_size)
        X_test, y_test = self.prepare_sequences(self.test_df, window_size)

        print(f"\nSekwencje:")
        print(f"  Train: {X_train.shape}")
        print(f"  Val:   {X_val.shape}")
        print(f"  Test:  {X_test.shape}")

        n_features = X_train.shape[2]

        # udawany trial optuna żeby nie trzeba było drugi raz budować modelu, tylko można zbudować model na podstawie najlepszych parametrów
        class FakowyTrial:
            def __init__(self, params):
                self.params = params

            def suggest_categorical(self, name, *args, **kwargs):
                return self.params[name]

        fake_trial = FakowyTrial(self.best_params)
        model = self.build_model(fake_trial, window_size, n_features)

        print("architektura modelu:")
        model.summary()

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15, # tu niech będzie żeby się niepotrzebnie nie przeuczał
            restore_best_weights=True,
            verbose=1
        )

        # żeby było w normalnej skali
        unscaled_callback = UnscaledMetricsCallback(
            X_val=X_val,
            y_val=y_val,
            scaler_y=self.scaler_y,
            horizons=self.horizons,
            print_every=5
        )

        print(f"Rozpoczynam trening (max epochs={epochs})...")

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, unscaled_callback],
            verbose=1
        )

        # Zapisz
        model.save(f"{self.output_dir}/final_model.keras")

        with open(f"{self.output_dir}/scalers.pkl", 'wb') as f:
            pickle.dump({
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y
            }, f)

        with open(f"{self.output_dir}/training_history_unscaled.pkl", 'wb') as f:
            pickle.dump(unscaled_callback.history_unscaled, f)

        print(f"Model zapisany: final_model.keras")

        self._plot_training_history(history, unscaled_callback.history_unscaled)

        return model, X_test, y_test

    def _plot_training_history(self, history, history_unscaled):
        """Wykres treningu"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # wykres zeskalowany
        axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MAE, skala 0-1)', fontsize=12)
        axes[0].set_title('Proces Uczenia (Scaled)', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # wykrs w normalnej skali
        epochs = history_unscaled['epoch']
        mae_unscaled = history_unscaled['avg_mae_unscaled']

        axes[1].plot(epochs, mae_unscaled, linewidth=2, color='orange')
        axes[1].axhline(y=min(mae_unscaled), color='green', linestyle='--',
                        label=f'Best: {min(mae_unscaled):.2f} µg/m³')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE (µg/m³)', fontsize=12)
        axes[1].set_title('Walidacja (Unscaled)', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/training_history.png", dpi=300)
        print(f"✓ Wykres: training_history.png")
        plt.close()

    def evaluate(self, model, X_test, y_test):
        """Ewaluacja na testowym zbiorze"""
        print("Testowanie na zbiorze testowym")

        y_pred_scaled = model.predict(X_test, verbose=0)
        y_test_original = self.scaler_y.inverse_transform(y_test)
        y_pred_original = self.scaler_y.inverse_transform(y_pred_scaled)

        metrics = {}

        for i, h in enumerate(self.horizons):
            y_true = y_test_original[:, i]
            y_pred = y_pred_original[:, i]

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100

            metrics[h] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                'y_true': y_true,
                'y_pred': y_pred
            }

            print(f"\nHoryzont {h}h:")
            print(f"  MAE:  {mae:.2f} µg/m³")
            print(f"  RMSE: {rmse:.2f} µg/m³")
            print(f"  R²:   {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")

        avg_mae = np.mean([metrics[h]['MAE'] for h in self.horizons])
        avg_rmse = np.mean([metrics[h]['RMSE'] for h in self.horizons])
        avg_r2 = np.mean([metrics[h]['R2'] for h in self.horizons])

        print(f"\nŚrednie:")
        print(f"  MAE:  {avg_mae:.2f} µg/m³")
        print(f"  RMSE: {avg_rmse:.2f} µg/m³")
        print(f"  R²:   {avg_r2:.4f}")

        # Zapisz
        metrics_save = {
            f'{h}h': {
                'MAE': float(metrics[h]['MAE']),
                'RMSE': float(metrics[h]['RMSE']),
                'R2': float(metrics[h]['R2']),
                'MAPE': float(metrics[h]['MAPE'])
            }
            for h in self.horizons
        }
        metrics_save['average'] = {
            'MAE': float(avg_mae),
            'RMSE': float(avg_rmse),
            'R2': float(avg_r2)
        }

        with open(f"{self.output_dir}/test_metrics.json", 'w') as f:
            json.dump(metrics_save, f, indent=4)

        print(f"Metryki zapisane: test_metrics.json")

        self._plot_metrics(metrics)
        self._plot_predictions(metrics)

        return metrics

    def _plot_metrics(self, metrics):
        """Wykres metryk"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        horizons_list = list(self.horizons)

        # MAE
        mae_values = [metrics[h]['MAE'] for h in horizons_list]
        axes[0, 0].bar(range(len(horizons_list)), mae_values,
                       color='skyblue', edgecolor='navy', linewidth=1.5)
        axes[0, 0].set_xticks(range(len(horizons_list)))
        axes[0, 0].set_xticklabels([f'{h}h' for h in horizons_list])
        axes[0, 0].set_ylabel('MAE (µg/m³)', fontsize=12)
        axes[0, 0].set_title('MAE per horyzont', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(mae_values):
            axes[0, 0].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

        # RMSE
        rmse_values = [metrics[h]['RMSE'] for h in horizons_list]
        axes[0, 1].bar(range(len(horizons_list)), rmse_values,
                       color='lightcoral', edgecolor='darkred', linewidth=1.5)
        axes[0, 1].set_xticks(range(len(horizons_list)))
        axes[0, 1].set_xticklabels([f'{h}h' for h in horizons_list])
        axes[0, 1].set_ylabel('RMSE (µg/m³)', fontsize=12)
        axes[0, 1].set_title('RMSE per horyzont', fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(rmse_values):
            axes[0, 1].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

        # R²
        r2_values = [metrics[h]['R2'] for h in horizons_list]
        axes[1, 0].bar(range(len(horizons_list)), r2_values,
                       color='lightgreen', edgecolor='darkgreen', linewidth=1.5)
        axes[1, 0].set_xticks(range(len(horizons_list)))
        axes[1, 0].set_xticklabels([f'{h}h' for h in horizons_list])
        axes[1, 0].set_ylabel('R²', fontsize=12)
        axes[1, 0].set_title('R² per horyzont', fontsize=13, fontweight='bold')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(r2_values):
            axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

        # MAPE
        mape_values = [metrics[h]['MAPE'] for h in horizons_list]
        axes[1, 1].bar(range(len(horizons_list)), mape_values,
                       color='plum', edgecolor='purple', linewidth=1.5)
        axes[1, 1].set_xticks(range(len(horizons_list)))
        axes[1, 1].set_xticklabels([f'{h}h' for h in horizons_list])
        axes[1, 1].set_ylabel('MAPE (%)', fontsize=12)
        axes[1, 1].set_title('MAPE per horyzont', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(mape_values):
            axes[1, 1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/metrics_comparison.png", dpi=300)
        print(f"✓ Wykres: metrics_comparison.png")
        plt.close()

    def _plot_predictions(self, metrics):
        """Wykres predykcji"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()

        for i, h in enumerate(self.horizons):
            y_true = metrics[h]['y_true']
            y_pred = metrics[h]['y_pred']

            n_samples = min(500, len(y_true))

            axes[i].plot(y_true[:n_samples], label='Rzeczywiste',
                         linewidth=2, alpha=0.7, color='navy')
            axes[i].plot(y_pred[:n_samples], label='Przewidywane',
                         linewidth=2, alpha=0.7, color='orange')
            axes[i].set_xlabel('Próbka', fontsize=11)
            axes[i].set_ylabel(f'{self.target_name} (µg/m³)', fontsize=11)
            axes[i].set_title(
                f'Horyzont {h}h\nMAE: {metrics[h]["MAE"]:.2f}, R²: {metrics[h]["R2"]:.4f}',
                fontsize=12, fontweight='bold'
            )
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/predictions_vs_actual.png", dpi=300)
        print(f"✓ Wykres: predictions_vs_actual.png")
        plt.close()


def main():

    CSV_PATH = '../Data/data_for_model/rozszerzone_cechy.csv' #dane na których się uczy model
    TARGET = 'pył zawieszony PM10' #cel predykcji
    HORIZONS = [1, 3, 6, 12] #horyzonty predykcji
    N_TRIALS = 50 #liczba prób optuna

    optimizer = LSTMOptymalizator(
        csv_path=CSV_PATH,
        target_name=TARGET,
        horizons=HORIZONS,
        output_dir='results_lstm_fixed' #nazwa filderu gdzie mają trafić modele, wykresy, skalery itp.
    )

    optimizer.load_data()
    study = optimizer.run_optimization(n_trials=N_TRIALS)
    model, X_test, y_test = optimizer.train_final_model(epochs=40)
    metrics = optimizer.evaluate(model, X_test, y_test)

    print("koniec optymalizacji")


if __name__ == "__main__":
    main()