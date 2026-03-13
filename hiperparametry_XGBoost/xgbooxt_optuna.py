import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import optuna
import pickle
import json
import os

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class XGBoostMultiOutputOptimizer:

    def __init__(self,
                 csv_path,
                 target_name='pył zawieszony PM10',
                 horizons=[1, 3, 6, 12],
                 train_split=0.7,
                 val_split=0.15,
                 output_dir='results_xgboost_nowe'):

        self.csv_path = csv_path
        self.target_name = target_name
        self.horizons = horizons
        self.train_split = train_split
        self.val_split = val_split
        self.output_dir = output_dir

        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.feature_names = None
        self.best_params = None
        self.model = None

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)

    def create_lag_features(self, df):
        df_features = df.copy()

        print(f"\n{'=' * 80}")
        print("FEATURE ENGINEERING")
        print(f"{'=' * 80}")

        temporal_features = ['hour', 'day_of_week', 'month', 'year', 'season']
        all_pollutants = ['dwutlenek azotu', 'pył zawieszony PM10', 'pył zawieszony PM2.5']
        dynamic_weather = ['wind_speed_10m', 'wind_direction_10m', 'precipitation']
        all_weather = [col for col in df.columns
                       if col not in all_pollutants
                       and col not in temporal_features
                       and col not in ['air_stagnation', 'fog_potential']]

        print(f"\n1. Zanieczyszczenia (lagi, rolling, różnice):")
        for pollutant in all_pollutants:
            if pollutant in df.columns:
                for lag in [1, 3, 6, 12, 24]:
                    df_features[f'{pollutant}_lag{lag}'] = df[pollutant].shift(lag)

                df_features[f'{pollutant}_rolling_min_3h'] = df[pollutant].rolling(3).min()
                df_features[f'{pollutant}_rolling_max_3h'] = df[pollutant].rolling(3).max()
                df_features[f'{pollutant}_rolling_mean_24h'] = df[pollutant].rolling(24).mean()
                df_features[f'{pollutant}_rolling_min_24h'] = df[pollutant].rolling(24).min()
                df_features[f'{pollutant}_rolling_max_24h'] = df[pollutant].rolling(24).max()

                df_features[f'{pollutant}_diff_1h'] = df[pollutant].diff(1)
                df_features[f'{pollutant}_diff_3h'] = df[pollutant].diff(3)

                print(f"   ✓ {pollutant}: 13 cech")

        print(f"\n2. Zmienne pogodowe dynamiczne (lagi):")
        for col in dynamic_weather:
            if col in df.columns:
                df_features[f'{col}_lag1'] = df[col].shift(1)
                df_features[f'{col}_lag3'] = df[col].shift(3)
                print(f"   ✓ {col}: 2 lagi")

        print(f"\n3. Zmienne pogodowe (rolling mean 24h):")
        for col in all_weather:
            if col in df.columns:
                df_features[f'{col}_rolling_mean_24h'] = df[col].rolling(24).mean()
        print(f"   ✓ {len(all_weather)} zmiennych")

        if 'temperature_2m' in df.columns:
            df_features['temp_diff_1h'] = df['temperature_2m'].diff(1)
            df_features['temp_diff_24h'] = df['temperature_2m'].diff(24)
            print(f"\n4. Temperatura (różnice): 2 cechy")

        initial_len = len(df_features)
        df_features = df_features.dropna()

        print(f"\n{'=' * 80}")
        print(f"Kolumny: {len(df.columns)} → {len(df_features.columns)}")
        print(f"Wiersze: {initial_len} → {len(df_features)} (usunięto {initial_len - len(df_features)} NaN)")
        print(f"{'=' * 80}\n")

        return df_features

    def load_data(self):
        print("=" * 80)
        print("ŁADOWANIE DANYCH")
        print("=" * 80)

        df = pd.read_csv(self.csv_path, index_col=0, parse_dates=True)
        df = df.sort_index()

        print(f"Rekordów (raw): {len(df)}")
        print(f"Cech (raw): {len(df.columns)}")
        print(f"Target: {self.target_name}")
        print(f"Horyzonty: {self.horizons}h")

        df = self.create_lag_features(df)

        self.feature_names = [col for col in df.columns if col != self.target_name]

        print(f"\nCech po feature engineering: {len(self.feature_names)}")

        total = len(df)
        train_end = int(total * self.train_split)
        val_end = int(total * (self.train_split + self.val_split))

        self.train_df = df.iloc[:train_end].copy()
        self.val_df = df.iloc[train_end:val_end].copy()
        self.test_df = df.iloc[val_end:].copy()

        print(f"\nPodział danych:")
        print(f"Train: {len(self.train_df)} ({self.train_split * 100:.0f}%)")
        print(f"Val:   {len(self.val_df)} ({self.val_split * 100:.0f}%)")
        print(f"Test:  {len(self.test_df)} ({(1 - self.train_split - self.val_split) * 100:.0f}%)")
        print("=" * 80)

    def prepare_data(self, df):
        df_clean = df.dropna()
        max_horizon = max(self.horizons)

        X = df_clean.drop(columns=[self.target_name]).values
        X = X[:-max_horizon]

        y_list = []
        for h in self.horizons:
            y_shifted = df_clean[self.target_name].shift(-h).values
            y_list.append(y_shifted[:-max_horizon])

        y = np.column_stack(y_list)

        return X, y

    def objective(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 20, 300, step=20)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        subsample = trial.suggest_float('subsample', 0.6, 1.0, step=0.1)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        gamma = trial.suggest_float('gamma', 0.0, 0.5, step=0.1)

        hyperparams = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'n_jobs': -1,
            'objective': 'reg:absoluteerror',
            'random_state': 42
        }

        try:
            X_train, y_train = self.prepare_data(self.train_df)
            X_val, y_val = self.prepare_data(self.val_df)
        except:
            return float('inf')

        try:
            base_model = xgb.XGBRegressor(**hyperparams)
            model = MultiOutputRegressor(base_model)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            mae_per_horizon = [mean_absolute_error(y_val[:, i], y_pred[:, i])
                               for i in range(len(self.horizons))]
            avg_mae = np.mean(mae_per_horizon)

            return avg_mae

        except Exception as e:
            print(f"  ⚠️  Trial failed: {e}")
            return float('inf')

    def run_optimization(self, n_trials=50):
        print("\n" + "=" * 80)
        print("OPTYMALIZACJA HIPERPARAMETRÓW XGBOOST MULTIOUTPUT")
        print("=" * 80)
        print(f"Liczba prób: {n_trials}")
        print(f"Strategia: 1 model MultiOutputRegressor dla {len(self.horizons)} horyzontów")
        print(f"Funkcja celu: średnia MAE")
        print(f"\nParametry do tuningu:")
        print(f"  - n_estimators: [100-500]")
        print(f"  - max_depth: [3-10]")
        print(f"  - learning_rate: [0.01-0.3]")
        print(f"  - subsample: [0.6-1.0]")
        print(f"  - colsample_bytree: [0.6-1.0]")
        print(f"  - min_child_weight: [1-10]")
        print(f"  - gamma: [0.0-0.5]")
        print("=" * 80 + "\n")

        study = optuna.create_study(
            direction='minimize',
            study_name='xgboost_multioutput_optimization',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )

        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        print("\n" + "=" * 80)
        print("OPTYMALIZACJA ZAKOŃCZONA")
        print("=" * 80)
        print(f"\nLiczba zakończonych trials: {len(study.trials)}")
        print(f"Najlepszy MAE (średnia z {len(self.horizons)} horyzontów): {study.best_value:.2f} µg/m³")

        print(f"\nNajlepsze hiperparametry:")
        for key, value in study.best_params.items():
            print(f"  {key:20s}: {value}")

        self.best_params = study.best_params

        with open(f"{self.output_dir}/best_params.json", 'w') as f:
            json.dump(study.best_params, f, indent=4)

        with open(f"{self.output_dir}/study.pkl", 'wb') as f:
            pickle.dump(study, f)

        print(f"\n✓ Wyniki zapisane w {self.output_dir}/")

        self._plot_optimization(study)

        return study

    def _plot_optimization(self, study):
        fig, ax = plt.subplots(figsize=(12, 6))

        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        values = [t.value for t in trials]

        ax.plot(values, marker='o', linestyle='-', linewidth=1.5, markersize=4)
        ax.axhline(y=study.best_value, color='r', linestyle='--',
                   label=f'Best: {study.best_value:.2f} µg/m³')
        ax.set_xlabel('Trial', fontsize=12)
        ax.set_ylabel('Średnia MAE (µg/m³)', fontsize=12)
        ax.set_title('XGBoost MultiOutput - Historia Optymalizacji', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/optimization_history.png", dpi=300)
        print(f"✓ Wykres: optimization_history.png")
        plt.close()

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            importances = optuna.importance.get_param_importances(study)
            params = list(importances.keys())
            values = list(importances.values())

            ax.barh(params, values, color='lightcoral', edgecolor='darkred')
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title('XGBoost - Ważność Hiperparametrów', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/plots/param_importances.png", dpi=300)
            print(f"✓ Wykres: param_importances.png")
            plt.close()
        except:
            print("Nie udało się wygenerować wykresu ważności parametrów")

    def train_final_model(self):
        print("\n" + "=" * 80)
        print("TRENING FINALNEGO MODELU")
        print("=" * 80)

        train_val_df = pd.concat([self.train_df, self.val_df])

        print(f"\nDataset: train + val = {len(train_val_df)} próbek")
        print(f"Model: MultiOutputRegressor z {len(self.horizons)} wyjściami")

        X_trainval, y_trainval = self.prepare_data(train_val_df)
        X_test, y_test = self.prepare_data(self.test_df)

        print(f"\nDane train+val: {X_trainval.shape}")
        print(f"Dane test: {X_test.shape}")
        print(f"Targets shape: {y_trainval.shape}")

        hyperparams = {
            **self.best_params,
            'objective': 'reg:absoluteerror',
            'random_state': 42,
            'n_jobs': -1
        }

        print(f"\nRozpoczynam trening modelu XGBoost MultiOutput...")
        print(f"  - n_estimators: {self.best_params['n_estimators']}")
        print(f"  - max_depth: {self.best_params['max_depth']}")
        print(f"  - learning_rate: {self.best_params['learning_rate']:.4f}")

        base_model = xgb.XGBRegressor(**hyperparams)
        self.model = MultiOutputRegressor(base_model)

        self.model.fit(X_trainval, y_trainval)

        print(f"\n✓ Model wytrenowany")

        with open(f"{self.output_dir}/model.pkl", 'wb') as f:
            pickle.dump(self.model, f)

        print(f"✓ Model zapisany: model.pkl")

        with open(f"{self.output_dir}/model_config.json", 'w') as f:
            json.dump({
                'horizons': self.horizons,
                'hyperparams': self.best_params,
                'n_features': len(self.feature_names),
                'feature_names': self.feature_names
            }, f, indent=4)

        return X_test, y_test

    def evaluate(self, X_test, y_test):
        print("\n" + "=" * 80)
        print("EWALUACJA NA TEST SET")
        print("=" * 80)

        y_pred = self.model.predict(X_test)

        metrics = {}

        for i, h in enumerate(self.horizons):
            y_true = y_test[:, i]
            y_pred_h = y_pred[:, i]

            mae = mean_absolute_error(y_true, y_pred_h)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred_h))
            r2 = r2_score(y_true, y_pred_h)
            mape = mean_absolute_percentage_error(y_true, y_pred_h) * 100

            metrics[h] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                'y_true': y_true,
                'y_pred': y_pred_h
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

        print(f"\n✓ Metryki zapisane: test_metrics.json")

        self._plot_metrics(metrics)
        self._plot_predictions(metrics)
        self._plot_feature_importance(top_n=10)

        return metrics

    def _plot_metrics(self, metrics):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        horizons_list = list(self.horizons)

        mae_values = [metrics[h]['MAE'] for h in horizons_list]
        axes[0, 0].bar(range(len(horizons_list)), mae_values,
                       color='skyblue', edgecolor='navy', linewidth=1.5)
        axes[0, 0].set_xticks(range(len(horizons_list)))
        axes[0, 0].set_xticklabels([f'{h}h' for h in horizons_list])
        axes[0, 0].set_ylabel('MAE (µg/m³)', fontsize=12)
        axes[0, 0].set_title('MAE dla różnych horyzontów', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(mae_values):
            axes[0, 0].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

        rmse_values = [metrics[h]['RMSE'] for h in horizons_list]
        axes[0, 1].bar(range(len(horizons_list)), rmse_values,
                       color='lightcoral', edgecolor='darkred', linewidth=1.5)
        axes[0, 1].set_xticks(range(len(horizons_list)))
        axes[0, 1].set_xticklabels([f'{h}h' for h in horizons_list])
        axes[0, 1].set_ylabel('RMSE (µg/m³)', fontsize=12)
        axes[0, 1].set_title('RMSE dla różnych horyzontów', fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(rmse_values):
            axes[0, 1].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

        r2_values = [metrics[h]['R2'] for h in horizons_list]
        axes[1, 0].bar(range(len(horizons_list)), r2_values,
                       color='lightgreen', edgecolor='darkgreen', linewidth=1.5)
        axes[1, 0].set_xticks(range(len(horizons_list)))
        axes[1, 0].set_xticklabels([f'{h}h' for h in horizons_list])
        axes[1, 0].set_ylabel('R²', fontsize=12)
        axes[1, 0].set_title('R² dla różnych horyzontów', fontsize=13, fontweight='bold')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(r2_values):
            axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

        mape_values = [metrics[h]['MAPE'] for h in horizons_list]
        axes[1, 1].bar(range(len(horizons_list)), mape_values,
                       color='plum', edgecolor='purple', linewidth=1.5)
        axes[1, 1].set_xticks(range(len(horizons_list)))
        axes[1, 1].set_xticklabels([f'{h}h' for h in horizons_list])
        axes[1, 1].set_ylabel('MAPE (%)', fontsize=12)
        axes[1, 1].set_title('MAPE dla różnych horyzontów', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(mape_values):
            axes[1, 1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/metrics_comparison.png", dpi=300)
        print(f"✓ Wykres: metrics_comparison.png")
        plt.close()

    def _plot_predictions(self, metrics):
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
                f'Horyzont {h}h\nMAE: {metrics[h]["MAE"]:.2f} µg/m³, R²: {metrics[h]["R2"]:.4f}',
                fontsize=12, fontweight='bold'
            )
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/predictions_vs_actual.png", dpi=300)
        print(f"✓ Wykres: predictions_vs_actual.png")
        plt.close()

    def _plot_feature_importance(self, top_n=10):
        print(f"\nGenerowanie wykresów ważności cech (top {top_n})...")

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()

        for i, h in enumerate(self.horizons):
            estimator = self.model.estimators_[i]
            importance = estimator.feature_importances_

            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)

            y_pos = np.arange(len(importance_df))
            axes[i].barh(y_pos, importance_df['importance'].values,
                         color='steelblue', edgecolor='darkblue', linewidth=1.5)
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(importance_df['feature'].values, fontsize=9)
            axes[i].invert_yaxis()
            axes[i].set_xlabel('Ważność cechy', fontsize=11)
            axes[i].set_title(
                f'Top {top_n} najważniejszych cech - Horyzont {h}h',
                fontsize=12, fontweight='bold'
            )
            axes[i].grid(True, alpha=0.3, axis='x')

            for j, v in enumerate(importance_df['importance'].values):
                axes[i].text(v, j, f' {v:.4f}', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/feature_importance.png", dpi=300)
        print(f"✓ Wykres: feature_importance.png")
        plt.close()

        print(f"\nZapisywanie pełnego rankingu cech do CSV...")
        for i, h in enumerate(self.horizons):
            estimator = self.model.estimators_[i]
            importance = estimator.feature_importances_

            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

            importance_df.to_csv(
                f"{self.output_dir}/feature_importance_{h}h.csv",
                index=False
            )

        print(f"✓ Rankingi cech zapisane: feature_importance_1h.csv, ...")


def main():
    print("\n" + "=" * 80)
    print("OPTYMALIZACJA HIPERPARAMETRÓW - XGBOOST MULTIOUTPUT")
    print("=" * 80)

    CSV_PATH = '../Data/data_for_model/rozszerzone_cechy.csv'
    TARGET = 'pył zawieszony PM10'
    HORIZONS = [1, 3, 6, 12]
    N_TRIALS = 50

    optimizer = XGBoostMultiOutputOptimizer(
        csv_path=CSV_PATH,
        target_name=TARGET,
        horizons=HORIZONS,
        output_dir='finalna wwersja'
    )

    optimizer.load_data()
    study = optimizer.run_optimization(n_trials=N_TRIALS)
    X_test, y_test = optimizer.train_final_model()
    metrics = optimizer.evaluate(X_test, y_test)

    print("\n" + "=" * 80)
    print("OPTYMALIZACJA XGBOOST MULTIOUTPUT ZAKOŃCZONA!")
    print("=" * 80)



if __name__ == "__main__":
    main()