import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import pickle
import json
import os

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class XGBoostPredictor:

    def __init__(self,
                 csv_path,
                 params,
                 target_name='pył zawieszony PM10',
                 horizons=[1, 3, 6, 12],
                 train_split=0.7,
                 val_split=0.15,
                 output_dir='results_xgboost_manual'):

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
        self.feature_names = None
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
        print(f"Train: {self.train_df.index.min()} -> {self.train_df.index.max()} (n={len(self.train_df)})")
        print(f"Val:   {self.val_df.index.min()} -> {self.val_df.index.max()} (n={len(self.val_df)})")
        print(f"Test:  {self.test_df.index.min()} -> {self.test_df.index.max()} (n={len(self.test_df)})")
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

    def train(self):
        print("\n" + "=" * 80)
        print("TRENING MODELU (PARAMETRY SZTYWNE)")
        print("=" * 80)

        # Zapisz parametry do JSON
        with open(f"{self.output_dir}/parameters.json", 'w') as f:
            json.dump(self.params, f, indent=4)

        train_val_df = pd.concat([self.train_df]) #żeby było sprawiedliwie to na tym samym co LSTM, ale jak coś to można dodać po przecinku jeszcze validacyjny

        print(f"\nDataset: train = {len(train_val_df)} próbek")
        print(f"Model: MultiOutputRegressor z {len(self.horizons)} wyjściami")

        X_trainval, y_trainval = self.prepare_data(train_val_df)
        X_test, y_test = self.prepare_data(self.test_df)

        hyperparams = {
            **self.params,
            'objective': 'reg:absoluteerror',
            'random_state': 42,
            'n_jobs': -1
        }

        print(f"\nParametry XGBoost:")
        print(json.dumps(self.params, indent=2))

        base_model = xgb.XGBRegressor(**hyperparams)
        self.model = MultiOutputRegressor(base_model)

        print("\nRozpoczynam trening...")
        self.model.fit(X_trainval, y_trainval)

        print(f"\n✓ Model wytrenowany")

        with open(f"{self.output_dir}/model.pkl", 'wb') as f:
            pickle.dump(self.model, f)

        with open(f"{self.output_dir}/model_config.json", 'w') as f:
            json.dump({
                'horizons': self.horizons,
                'hyperparams': self.params,
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
        metrics_json = {}

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

            metrics_json[f'{h}h'] = {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'R2': float(r2),
                'MAPE': float(mape)
            }

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

        with open(f"{self.output_dir}/test_metrics.json", 'w') as f:
            json.dump(metrics_json, f, indent=4)

        # TABELKI W KONSOLI (Pandas)
        print("\n📊 PODSUMOWANIE PARAMETRÓW MODELU:")
        params_df = pd.DataFrame(list(self.params.items()), columns=['Parametr', 'Wartość'])
        print(params_df.to_string(index=False))

        print("\n📊 WYNIKI METRYK NA ZBIORZE TESTOWYM:")
        results_list = []
        for h in self.horizons:
            row = {'Horyzont': f"{h}h"}
            row.update({k: round(v, 4) for k, v in metrics[h].items() if k not in ['y_true', 'y_pred']})
            results_list.append(row)

        # Dodaj wiersz średniej
        avg_row = {'Horyzont': 'Średnia'}
        avg_row.update(metrics_json['average'])
        for k in avg_row:
             if k != 'Horyzont': avg_row[k] = round(avg_row[k], 4)
        results_list.append(avg_row)

        results_df = pd.DataFrame(results_list)
        cols = ['Horyzont', 'MAE', 'RMSE', 'R2', 'MAPE']
        results_df = results_df[cols]
        print(results_df.to_string(index=False))

        print(f"\n✓ Metryki zapisane: {self.output_dir}/test_metrics.json")
        print(f"✓ Parametry zapisane: {self.output_dir}/parameters.json")

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
        plt.close()

    def _plot_predictions(self, metrics):
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()

        for i, h in enumerate(self.horizons):
            y_true = metrics[h]['y_true']
            y_pred = metrics[h]['y_pred']

            n_samples = min(36, len(y_true))

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


def main():
    print("\n" + "=" * 80)
    print("XGBOOST MULTIOUTPUT - TRENING NA SZTYWNO")
    print("=" * 80)

    CSV_PATH = '../../Data/data_for_model/rozszerzone_cechy.csv'
    TARGET = 'pył zawieszony PM10'  # Zmień target tutaj
    OUTPUT_DIR = 'results_xgboost_pm10TEST'
    HORIZONS = [1, 3, 6, 12]

    # PARAMETRY NA SZTYWNO
    FIXED_PARAMS = {
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.0502,
        'subsample': 0.6,
        'colsample_bytree': 0.9,
        'min_child_weight': 10,
        'gamma': 0.0
    }

    predictor = XGBoostPredictor(
        csv_path=CSV_PATH,
        params=FIXED_PARAMS,
        target_name=TARGET,
        horizons=HORIZONS,
        output_dir=OUTPUT_DIR
    )

    predictor.load_data()
    X_test, y_test = predictor.train()
    metrics = predictor.evaluate(X_test, y_test)

    print("\n" + "=" * 80)
    print(f"✅ ZAKOŃCZONO! Wyniki w: {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()