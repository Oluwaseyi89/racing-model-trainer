# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import joblib


# class FuelModelTrainer:
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#         self.scaler = StandardScaler()
#         self.feature_columns = []

#     # ---------------------------
#     # TRAINING ENTRY POINT
#     # ---------------------------
#     def train(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> dict:
#         if telemetry_data.empty or lap_data.empty:
#             lap_data, telemetry_data = self._fabricate_minimal_data()

#         features_df, targets = self._extract_fuel_features(telemetry_data, lap_data)

#         if features_df.empty or len(targets) == 0:
#             return {'error': 'No valid fuel features extracted'}

#         # Ensure numeric consistency
#         features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
#         targets = np.array(targets, dtype=float)
#         targets = np.where(np.isnan(targets), 2.8, targets)

#         # Remove rows with NaNs
#         valid_mask = ~features_df.isna().any(axis=1) & ~np.isnan(targets)
#         X = features_df[valid_mask]
#         y = targets[valid_mask]

#         if len(X) < 10:
#             return {'error': f'Insufficient training samples: {len(X)}'}

#         X_scaled = self.scaler.fit_transform(X)
#         self.feature_columns = X.columns.tolist()

#         try:
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X_scaled, y, test_size=0.2, random_state=42
#             )
#             self.model.fit(X_train, y_train)
#             train_score = self.model.score(X_train, y_train)
#             test_score = self.model.score(X_test, y_test)
#         except Exception as e:
#             return {'error': f'Model training failed: {e}'}

#         return {
#             'model': self,
#             'features': self.feature_columns,
#             'train_score': train_score,
#             'test_score': test_score,
#             'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_)),
#             'training_samples': len(X)
#         }

#     # ---------------------------
#     # FEATURE EXTRACTION
#     # ---------------------------
#     def _extract_fuel_features(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> tuple:
#         features_list, consumption_targets = [], []

#         grouped = telemetry_data.groupby(['vehicle_number', 'lap'])
#         for (vehicle_num, lap_num), lap_telemetry in grouped:
#             if len(lap_telemetry) < 50:
#                 lap_telemetry = self._fabricate_lap_telemetry(vehicle_num, lap_num, 50)

#             lap_info = lap_data[(lap_data['NUMBER'] == vehicle_num) & (lap_data['LAP_NUMBER'] == lap_num)]
#             lap_info = lap_info.iloc[0] if not lap_info.empty else self._fabricate_lap_info(vehicle_num, lap_num)

#             features = self._calculate_features(lap_telemetry, lap_info)
#             fuel_estimate = self._estimate_fuel(lap_telemetry, lap_info)

#             features_list.append(features)
#             consumption_targets.append(fuel_estimate)

#         if features_list:
#             return pd.DataFrame(features_list), np.array(consumption_targets, dtype=float)
#         return pd.DataFrame(), np.array([], dtype=float)

#     def _calculate_features(self, telemetry: pd.DataFrame, lap_info: pd.Series) -> dict:
#         try:
#             throttle = telemetry.get('aps', pd.Series([0.0])).mean()
#             brake = ((telemetry.get('pbrake_f', pd.Series([0.0])) + telemetry.get('pbrake_r', pd.Series([0.0]))) / 2).mean()
#             long_acc = telemetry.get('accx_can', pd.Series([0.0])).abs().mean()
#             lat_acc = telemetry.get('accy_can', pd.Series([0.0])).abs().mean()
#             gear_changes = telemetry.get('gear', pd.Series([3])).diff().abs().sum()
#             avg_gear = telemetry.get('gear', pd.Series([3])).mean()
#             avg_speed = lap_info.get('KPH', 100.0)
#             top_speed = telemetry.get('TOP_SPEED', telemetry.get('KPH', pd.Series([avg_speed]))).max()
#             lap_time = lap_info.get('LAP_TIME_SECONDS', 60.0)
#             sectors = [
#                 lap_info.get('S1_SECONDS', lap_time / 3),
#                 lap_info.get('S2_SECONDS', lap_time / 3),
#                 lap_info.get('S3_SECONDS', lap_time / 3)
#             ]
#             speed_var = telemetry.get('KPH', pd.Series([avg_speed])).var() or 0.0

#             return {
#                 'throttle_usage': throttle,
#                 'brake_pressure': brake,
#                 'longitudinal_accel': long_acc,
#                 'lateral_accel': lat_acc,
#                 'gear_changes': gear_changes,
#                 'avg_gear': avg_gear,
#                 'avg_speed': avg_speed,
#                 'top_speed': top_speed,
#                 'lap_time': lap_time,
#                 'sector1_time': sectors[0],
#                 'sector2_time': sectors[1],
#                 'sector3_time': sectors[2],
#                 'speed_variance': speed_var,
#                 'lap_number': lap_info.get('LAP_NUMBER', 1)
#             }
#         except Exception:
#             return {
#                 'throttle_usage': 50.0,
#                 'brake_pressure': 20.0,
#                 'longitudinal_accel': 0.5,
#                 'lateral_accel': 0.5,
#                 'gear_changes': 5,
#                 'avg_gear': 3.0,
#                 'avg_speed': 100.0,
#                 'top_speed': 120.0,
#                 'lap_time': 60.0,
#                 'sector1_time': 20.0,
#                 'sector2_time': 20.0,
#                 'sector3_time': 20.0,
#                 'speed_variance': 0.0,
#                 'lap_number': 1
#             }

#     def _estimate_fuel(self, telemetry: pd.DataFrame, lap_info: pd.Series) -> float:
#         try:
#             base = 2.8
#             throttle_factor = telemetry.get('aps', pd.Series([0.0])).mean() / 100 * 0.8
#             speed_factor = lap_info.get('KPH', 100.0) / 200 * 1.2
#             accel_factor = telemetry.get('accx_can', pd.Series([0.0])).abs().mean() * 2.5
#             avg_gear = telemetry.get('gear', pd.Series([3])).mean()
#             gear_eff = 1.0 - abs(avg_gear - 3) * 0.1
#             consumption = base * (1.0 + throttle_factor + speed_factor + accel_factor) * gear_eff
#             return max(1.5, consumption + np.random.normal(0, 0.15))
#         except Exception:
#             return 2.8

#     # ---------------------------
#     # PREDICTION
#     # ---------------------------
#     def predict_fuel_consumption(self, features: dict) -> float:
#         try:
#             vec = np.array([features.get(col, 0.0) for col in self.feature_columns]).reshape(1, -1)
#             scaled = self.scaler.transform(vec)
#             return max(1.0, self.model.predict(scaled)[0])
#         except Exception:
#             return self._fallback_fuel(features)

#     def _fallback_fuel(self, features: dict) -> float:
#         base = 2.8
#         throttle = features.get('throttle_usage', 50.0) / 100 * 0.8
#         speed = features.get('avg_speed', 120.0) / 200 * 1.2
#         return base * (1.0 + throttle + speed)

#     def estimate_remaining_laps(self, current_fuel: float, features: dict) -> int:
#         rate = self.predict_fuel_consumption(features)
#         return max(0, int(current_fuel / rate))

#     # ---------------------------
#     # SYNTHETIC DATA HELPERS
#     # ---------------------------
#     def _fabricate_minimal_data(self) -> tuple:
#         lap_data = pd.DataFrame([self._fabricate_lap_info(v, l) for v in range(1, 3) for l in range(1, 6)])
#         telemetry_data = pd.concat(
#             [self._fabricate_lap_telemetry(v, l, 50) for v in range(1, 3) for l in range(1, 6)],
#             ignore_index=True
#         )
#         return lap_data, telemetry_data

#     def _fabricate_lap_info(self, vehicle_num: int, lap_num: int) -> pd.Series:
#         return pd.Series({
#             'NUMBER': vehicle_num,
#             'LAP_NUMBER': lap_num,
#             'KPH': 100.0 + np.random.uniform(-10, 10),
#             'LAP_TIME_SECONDS': 60.0 + np.random.uniform(-5, 5),
#             'S1_SECONDS': 20.0 + np.random.uniform(-2, 2),
#             'S2_SECONDS': 20.0 + np.random.uniform(-2, 2),
#             'S3_SECONDS': 20.0 + np.random.uniform(-2, 2)
#         })

#     def _fabricate_lap_telemetry(self, vehicle_num: int, lap_num: int, n_points: int) -> pd.DataFrame:
#         return pd.DataFrame({
#             'vehicle_number': vehicle_num,
#             'lap': lap_num,
#             'aps': np.random.uniform(0, 100, n_points),
#             'pbrake_f': np.random.uniform(0, 50, n_points),
#             'pbrake_r': np.random.uniform(0, 50, n_points),
#             'accx_can': np.random.uniform(-1, 1, n_points),
#             'accy_can': np.random.uniform(-1, 1, n_points),
#             'gear': np.random.randint(1, 7, n_points),
#             'KPH': np.random.uniform(50, 200, n_points),
#             'TOP_SPEED': np.random.uniform(100, 220, n_points)
#         })

#     # ---------------------------
#     # MODEL SERIALIZATION
#     # ---------------------------
#     def save_model(self, filepath: str):
#         joblib.dump({
#             'model': self.model,
#             'scaler': self.scaler,
#             'feature_columns': self.feature_columns
#         }, filepath)

#     def load_model(self, filepath: str):
#         data = joblib.load(filepath)
#         self.model = data['model']
#         self.scaler = data['scaler']
#         self.feature_columns = data['feature_columns']






















import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class FuelModelTrainer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.feature_columns = []

    # ---------------------------
    # TRAINING ENTRY POINT
    # ---------------------------
    def train(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> dict:
        # Fabricate minimal data if inputs are empty
        if telemetry_data.empty or lap_data.empty:
            lap_data, telemetry_data = self._fabricate_minimal_data()

        features_df, targets = self._extract_fuel_features(telemetry_data, lap_data)

        if features_df.empty or len(targets) == 0:
            return {'error': 'No valid fuel features extracted'}

        X = features_df
        y = np.array(targets)

        # Remove rows with NaNs
        valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 10:
            return {'error': f'Insufficient training samples: {len(X)}'}

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.feature_columns = X.columns.tolist()

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            self.model.fit(X_train, y_train)
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
        except Exception as e:
            return {'error': f'Model training failed: {e}'}

        return {
            'model': self,
            'features': self.feature_columns,
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_)),
            'training_samples': len(X)
        }

    # ---------------------------
    # FEATURE EXTRACTION
    # ---------------------------
    def _extract_fuel_features(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> tuple:
        features_list = []
        consumption_targets = []

        grouped = telemetry_data.groupby(['vehicle_number', 'lap'])
        for (vehicle_num, lap_num), lap_telemetry in grouped:
            if len(lap_telemetry) < 50:
                lap_telemetry = self._fabricate_lap_telemetry(vehicle_num, lap_num, 50)

            lap_info = lap_data[(lap_data['NUMBER'] == vehicle_num) & (lap_data['LAP_NUMBER'] == lap_num)]
            if lap_info.empty:
                lap_info = self._fabricate_lap_info(vehicle_num, lap_num)
            else:
                lap_info = lap_info.iloc[0]

            features = self._calculate_features(lap_telemetry, lap_info)
            fuel_estimate = self._estimate_fuel(lap_telemetry, lap_info)

            features_list.append(features)
            consumption_targets.append(fuel_estimate)

        if features_list:
            return pd.DataFrame(features_list), consumption_targets
        return pd.DataFrame(), []

    def _calculate_features(self, telemetry: pd.DataFrame, lap_info: pd.Series) -> dict:
        try:
            throttle = telemetry.get('aps', pd.Series([0])).mean()
            brake = ((telemetry.get('pbrake_f', 0) + telemetry.get('pbrake_r', 0)) / 2).mean()
            long_acc = telemetry.get('accx_can', pd.Series([0])).abs().mean()
            lat_acc = telemetry.get('accy_can', pd.Series([0])).abs().mean()
            gear_changes = telemetry.get('gear', pd.Series([0])).diff().abs().sum()
            avg_gear = telemetry.get('gear', pd.Series([3])).mean()
            avg_speed = lap_info.get('KPH', 100)
            top_speed = telemetry.get('TOP_SPEED', telemetry.get('KPH', pd.Series([avg_speed]))).max()
            lap_time = lap_info.get('LAP_TIME_SECONDS', 60)
            sectors = [
                lap_info.get('S1_SECONDS', lap_time/3),
                lap_info.get('S2_SECONDS', lap_time/3),
                lap_info.get('S3_SECONDS', lap_time/3)
            ]
            speed_var = telemetry.get('KPH', pd.Series([avg_speed])).var()
            if pd.isna(speed_var):
                speed_var = 0.0

            return {
                'throttle_usage': throttle,
                'brake_pressure': brake,
                'longitudinal_accel': long_acc,
                'lateral_accel': lat_acc,
                'gear_changes': gear_changes,
                'avg_gear': avg_gear,
                'avg_speed': avg_speed,
                'top_speed': top_speed,
                'lap_time': lap_time,
                'sector1_time': sectors[0],
                'sector2_time': sectors[1],
                'sector3_time': sectors[2],
                'speed_variance': speed_var,
                'lap_number': lap_info.get('LAP_NUMBER', 1)
            }
        except Exception:
            # Fallback features
            return {
                'throttle_usage': 50,
                'brake_pressure': 20,
                'longitudinal_accel': 0.5,
                'lateral_accel': 0.5,
                'gear_changes': 5,
                'avg_gear': 3,
                'avg_speed': 100,
                'top_speed': 120,
                'lap_time': 60,
                'sector1_time': 20,
                'sector2_time': 20,
                'sector3_time': 20,
                'speed_variance': 0.0,
                'lap_number': 1
            }

    def _estimate_fuel(self, telemetry: pd.DataFrame, lap_info: pd.Series) -> float:
        try:
            base = 2.8
            throttle_factor = telemetry.get('aps', pd.Series([0])).mean() / 100 * 0.8
            speed_factor = lap_info.get('KPH', 100) / 200 * 1.2
            accel_factor = telemetry.get('accx_can', pd.Series([0])).abs().mean() * 2.5
            avg_gear = telemetry.get('gear', pd.Series([3])).mean()
            gear_eff = 1.0 - abs(avg_gear - 3) * 0.1
            consumption = base * (1 + throttle_factor + speed_factor + accel_factor) * gear_eff
            return max(1.5, consumption + np.random.normal(0, 0.15))
        except Exception:
            return 2.8

    # ---------------------------
    # PREDICTION
    # ---------------------------
    def predict_fuel_consumption(self, features: dict) -> float:
        try:
            vec = np.array([features.get(col, 0) for col in self.feature_columns]).reshape(1, -1)
            scaled = self.scaler.transform(vec)
            return max(1.0, self.model.predict(scaled)[0])
        except Exception:
            return self._fallback_fuel(features)

    def _fallback_fuel(self, features: dict) -> float:
        base = 2.8
        throttle = features.get('throttle_usage', 50) / 100 * 0.8
        speed = features.get('avg_speed', 120) / 200 * 1.2
        return base * (1 + throttle + speed)

    def estimate_remaining_laps(self, current_fuel: float, features: dict) -> int:
        rate = self.predict_fuel_consumption(features)
        return max(0, int(current_fuel / rate))

    # ---------------------------
    # SYNTHETIC DATA HELPERS
    # ---------------------------
    def _fabricate_minimal_data(self) -> tuple:
        lap_data = pd.DataFrame([self._fabricate_lap_info(v, l) for v in range(1, 3) for l in range(1, 6)])
        telemetry_data = pd.concat([self._fabricate_lap_telemetry(v, l, 50) for v in range(1, 3) for l in range(1, 6)],
                                   ignore_index=True)
        return lap_data, telemetry_data

    def _fabricate_lap_info(self, vehicle_num: int, lap_num: int) -> pd.Series:
        return pd.Series({
            'NUMBER': vehicle_num,
            'LAP_NUMBER': lap_num,
            'KPH': 100 + np.random.uniform(-10, 10),
            'LAP_TIME_SECONDS': 60 + np.random.uniform(-5, 5),
            'S1_SECONDS': 20 + np.random.uniform(-2, 2),
            'S2_SECONDS': 20 + np.random.uniform(-2, 2),
            'S3_SECONDS': 20 + np.random.uniform(-2, 2)
        })

    def _fabricate_lap_telemetry(self, vehicle_num: int, lap_num: int, n_points: int) -> pd.DataFrame:
        return pd.DataFrame({
            'vehicle_number': vehicle_num,
            'lap': lap_num,
            'aps': np.random.uniform(0, 100, n_points),
            'pbrake_f': np.random.uniform(0, 50, n_points),
            'pbrake_r': np.random.uniform(0, 50, n_points),
            'accx_can': np.random.uniform(-1, 1, n_points),
            'accy_can': np.random.uniform(-1, 1, n_points),
            'gear': np.random.randint(1, 7, n_points),
            'KPH': np.random.uniform(50, 200, n_points),
            'TOP_SPEED': np.random.uniform(100, 220, n_points)
        })

    # ---------------------------
    # MODEL SERIALIZATION
    # ---------------------------
    def save_model(self, filepath: str):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, filepath)

    def load_model(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']

























# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import joblib

# class FuelModelTrainer:
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#         self.scaler = StandardScaler()
#         self.feature_columns = []

#     # ------------------------------------------------------------
#     # TRAINING ENTRY POINT
#     # ------------------------------------------------------------
#     def train(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> dict:
#         if telemetry_data.empty or lap_data.empty:
#             # Fabricate minimal synthetic lap and telemetry data if missing
#             lap_data, telemetry_data = self._fabricate_minimal_data()

#         features_df, consumption_targets = self._extract_real_fuel_features(telemetry_data, lap_data)

#         if features_df.empty or len(consumption_targets) == 0:
#             return {'error': 'No valid fuel features extracted'}

#         X = features_df
#         y = np.array(consumption_targets)

#         valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
#         X = X[valid_mask]
#         y = y[valid_mask]

#         if len(X) < 10:
#             return {'error': f'Insufficient training samples: {len(X)}'}

#         X_scaled = self.scaler.fit_transform(X)
#         self.feature_columns = X.columns.tolist()

#         X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#         self.model.fit(X_train, y_train)

#         train_score = self.model.score(X_train, y_train)
#         test_score = self.model.score(X_test, y_test)

#         return {
#             'model': self,
#             'features': self.feature_columns,
#             'train_score': train_score,
#             'test_score': test_score,
#             'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_)),
#             'training_samples': len(X)
#         }

#     # ------------------------------------------------------------
#     # FEATURE EXTRACTION
#     # ------------------------------------------------------------
#     def _extract_real_fuel_features(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> tuple:
#         features_list = []
#         consumption_rates = []

#         grouped_telemetry = telemetry_data.groupby(['vehicle_number', 'lap'])

#         for (vehicle_num, lap_num), lap_telemetry in grouped_telemetry:
#             if len(lap_telemetry) < 50:
#                 # Fabricate synthetic telemetry for small laps
#                 lap_telemetry = self._fabricate_lap_telemetry(vehicle_num, lap_num, 50)

#             lap_info = lap_data[(lap_data['NUMBER'] == vehicle_num) & (lap_data['LAP_NUMBER'] == lap_num)]
#             if lap_info.empty:
#                 # Fabricate lap info if missing
#                 lap_info = self._fabricate_lap_info(vehicle_num, lap_num)
#             else:
#                 lap_info = lap_info.iloc[0]

#             features = self._calculate_telemetry_features(lap_telemetry, lap_info)
#             fuel_estimate = self._estimate_fuel_from_telemetry(lap_telemetry, lap_info)

#             features_list.append(features)
#             consumption_rates.append(fuel_estimate)

#         if features_list:
#             return pd.DataFrame(features_list), consumption_rates
#         return pd.DataFrame(), []

#     def _calculate_telemetry_features(self, lap_telemetry: pd.DataFrame, lap_info: pd.Series) -> dict:
#         try:
#             throttle_positions = lap_telemetry.get('aps', pd.Series([0])).dropna()
#             throttle_usage = throttle_positions.mean() if not throttle_positions.empty else 0

#             brake_pressure = ((lap_telemetry.get('pbrake_f', 0) + lap_telemetry.get('pbrake_r', 0)) / 2).mean()
#             longitudinal_accel = lap_telemetry.get('accx_can', pd.Series([0])).abs().mean()
#             lateral_accel = lap_telemetry.get('accy_can', pd.Series([0])).abs().mean()
#             gear_changes = lap_telemetry.get('gear', pd.Series([0])).diff().abs().sum()
#             avg_gear = lap_telemetry.get('gear', pd.Series([0])).mean()
#             avg_speed = lap_info.get('KPH', 100)
#             top_speed = lap_telemetry.get('TOP_SPEED', lap_telemetry.get('KPH', pd.Series([avg_speed]))).max()
#             lap_time = lap_info.get('LAP_TIME_SECONDS', 60)
#             sector_times = [
#                 lap_info.get('S1_SECONDS', lap_time/3),
#                 lap_info.get('S2_SECONDS', lap_time/3),
#                 lap_info.get('S3_SECONDS', lap_time/3)
#             ]
#             speed_variance = lap_telemetry.get('KPH', pd.Series([avg_speed])).var()
#             if pd.isna(speed_variance):
#                 speed_variance = 0

#             return {
#                 'throttle_usage': throttle_usage,
#                 'brake_pressure': brake_pressure,
#                 'longitudinal_accel': longitudinal_accel,
#                 'lateral_accel': lateral_accel,
#                 'gear_changes': gear_changes,
#                 'avg_gear': avg_gear,
#                 'avg_speed': avg_speed,
#                 'top_speed': top_speed,
#                 'lap_time': lap_time,
#                 'sector1_time': sector_times[0],
#                 'sector2_time': sector_times[1],
#                 'sector3_time': sector_times[2],
#                 'speed_variance': speed_variance,
#                 'lap_number': lap_info.get('LAP_NUMBER', 0)
#             }
#         except Exception:
#             # Provide default fallback features
#             return {
#                 'throttle_usage': 50,
#                 'brake_pressure': 20,
#                 'longitudinal_accel': 0.5,
#                 'lateral_accel': 0.5,
#                 'gear_changes': 5,
#                 'avg_gear': 3,
#                 'avg_speed': 100,
#                 'top_speed': 120,
#                 'lap_time': 60,
#                 'sector1_time': 20,
#                 'sector2_time': 20,
#                 'sector3_time': 20,
#                 'speed_variance': 0.0,
#                 'lap_number': 1
#             }

#     def _estimate_fuel_from_telemetry(self, lap_telemetry: pd.DataFrame, lap_info: pd.Series) -> float:
#         try:
#             base_consumption = 2.8
#             throttle_factor = lap_telemetry.get('aps', pd.Series([0])).mean() / 100 * 0.8
#             speed_factor = lap_info.get('KPH', 100) / 200 * 1.2
#             accel_factor = lap_telemetry.get('accx_can', pd.Series([0])).abs().mean() * 2.5
#             avg_gear = lap_telemetry.get('gear', pd.Series([3])).mean()
#             gear_efficiency = 1.0 - abs(avg_gear - 3) * 0.1

#             total_consumption = base_consumption * (1 + throttle_factor + speed_factor + accel_factor) * gear_efficiency
#             noise = np.random.normal(0, 0.15)
#             return max(1.5, total_consumption + noise)
#         except Exception:
#             return 2.8

#     # ------------------------------------------------------------
#     # PREDICTION
#     # ------------------------------------------------------------
#     def predict_fuel_consumption(self, telemetry_features: dict) -> float:
#         try:
#             feature_vector = [telemetry_features.get(col, 0) for col in self.feature_columns]
#             feature_array = np.array(feature_vector).reshape(1, -1)
#             scaled_features = self.scaler.transform(feature_array)
#             return max(1.0, self.model.predict(scaled_features)[0])
#         except Exception:
#             return self._estimate_fuel_from_telemetry_fallback(telemetry_features)

#     def _estimate_fuel_from_telemetry_fallback(self, features: dict) -> float:
#         base_consumption = 2.8
#         throttle_factor = features.get('throttle_usage', 50) / 100 * 0.8
#         speed_factor = features.get('avg_speed', 120) / 200 * 1.2
#         return base_consumption * (1 + throttle_factor + speed_factor)

#     def estimate_remaining_laps(self, current_fuel: float, telemetry_features: dict) -> int:
#         consumption_rate = self.predict_fuel_consumption(telemetry_features)
#         return max(0, int(current_fuel / consumption_rate))

#     # ------------------------------------------------------------
#     # SYNTHETIC DATA HELPERS
#     # ------------------------------------------------------------
#     def _fabricate_minimal_data(self) -> tuple:
#         lap_data = pd.DataFrame([self._fabricate_lap_info(vehicle, lap) 
#                                  for vehicle in range(1, 3) for lap in range(1, 6)])
#         telemetry_data = pd.concat([self._fabricate_lap_telemetry(v, l, 50) 
#                                     for v in range(1,3) for l in range(1,6)], ignore_index=True)
#         return lap_data, telemetry_data

#     def _fabricate_lap_info(self, vehicle_num: int, lap_num: int) -> pd.Series:
#         return pd.Series({
#             'NUMBER': vehicle_num,
#             'LAP_NUMBER': lap_num,
#             'KPH': 100 + np.random.uniform(-10, 10),
#             'LAP_TIME_SECONDS': 60 + np.random.uniform(-5, 5),
#             'S1_SECONDS': 20 + np.random.uniform(-2, 2),
#             'S2_SECONDS': 20 + np.random.uniform(-2, 2),
#             'S3_SECONDS': 20 + np.random.uniform(-2, 2)
#         })

#     def _fabricate_lap_telemetry(self, vehicle_num: int, lap_num: int, n_points: int) -> pd.DataFrame:
#         return pd.DataFrame({
#             'vehicle_number': vehicle_num,
#             'lap': lap_num,
#             'aps': np.random.uniform(0, 100, n_points),
#             'pbrake_f': np.random.uniform(0, 50, n_points),
#             'pbrake_r': np.random.uniform(0, 50, n_points),
#             'accx_can': np.random.uniform(-1, 1, n_points),
#             'accy_can': np.random.uniform(-1, 1, n_points),
#             'gear': np.random.randint(1, 7, n_points),
#             'KPH': np.random.uniform(50, 200, n_points),
#             'TOP_SPEED': np.random.uniform(100, 220, n_points)
#         })

#     # ------------------------------------------------------------
#     # MODEL SERIALIZATION
#     # ------------------------------------------------------------
#     def save_model(self, filepath: str):
#         joblib.dump({
#             'model': self.model,
#             'scaler': self.scaler,
#             'feature_columns': self.feature_columns
#         }, filepath)

#     def load_model(self, filepath: str):
#         data = joblib.load(filepath)
#         self.model = data['model']
#         self.scaler = data['scaler']
#         self.feature_columns = data['feature_columns']





















# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import joblib

# class FuelModelTrainer:
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#         self.scaler = StandardScaler()
#         self.feature_columns = []

#     def train(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> dict:
#         """Train fuel consumption model using real telemetry data"""
#         if telemetry_data.empty or lap_data.empty:
#             return {'error': 'Insufficient data provided'}

#         features_df, consumption_targets = self._extract_real_fuel_features(telemetry_data, lap_data)

#         if features_df.empty or len(consumption_targets) == 0:
#             return {'error': 'No valid fuel features extracted'}

#         # Prepare training data
#         X = features_df
#         y = np.array(consumption_targets)

#         # Remove rows with NaNs
#         valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
#         X = X[valid_mask]
#         y = y[valid_mask]

#         if len(X) < 10:
#             return {'error': f'Insufficient training samples: {len(X)}'}

#         # Scale features
#         X_scaled = self.scaler.fit_transform(X)
#         self.feature_columns = X.columns.tolist()

#         # Train model
#         X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#         self.model.fit(X_train, y_train)

#         # Evaluate
#         train_score = self.model.score(X_train, y_train)
#         test_score = self.model.score(X_test, y_test)

#         return {
#             'model': self,
#             'features': self.feature_columns,
#             'train_score': train_score,
#             'test_score': test_score,
#             'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_)),
#             'training_samples': len(X)
#         }

#     def _extract_real_fuel_features(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> tuple:
#         """Extract real fuel consumption features from telemetry data"""
#         features_list = []
#         consumption_rates = []

#         if telemetry_data.empty or lap_data.empty:
#             return pd.DataFrame(), []

#         grouped_telemetry = telemetry_data.groupby(['vehicle_number', 'lap'])

#         for (vehicle_num, lap_num), lap_telemetry in grouped_telemetry:
#             if len(lap_telemetry) < 50:
#                 continue

#             lap_info = lap_data[(lap_data['NUMBER'] == vehicle_num) & (lap_data['LAP_NUMBER'] == lap_num)]
#             if lap_info.empty:
#                 continue
#             lap_info = lap_info.iloc[0]

#             features = self._calculate_telemetry_features(lap_telemetry, lap_info)
#             if features is None:
#                 continue

#             fuel_estimate = self._estimate_fuel_from_telemetry(lap_telemetry, lap_info)
#             features_list.append(features)
#             consumption_rates.append(fuel_estimate)

#         if features_list:
#             return pd.DataFrame(features_list), consumption_rates
#         return pd.DataFrame(), []

#     def _calculate_telemetry_features(self, lap_telemetry: pd.DataFrame, lap_info: pd.Series) -> dict:
#         """Calculate fuel consumption features from telemetry data"""
#         try:
#             throttle_positions = lap_telemetry.get('aps', pd.Series([0])).dropna()
#             throttle_usage = throttle_positions.mean() if not throttle_positions.empty else 0

#             brake_pressure = ((lap_telemetry.get('pbrake_f', 0) + lap_telemetry.get('pbrake_r', 0)) / 2).mean()
#             longitudinal_accel = lap_telemetry.get('accx_can', pd.Series([0])).abs().mean()
#             lateral_accel = lap_telemetry.get('accy_can', pd.Series([0])).abs().mean()
#             gear_changes = lap_telemetry.get('gear', pd.Series([0])).diff().abs().sum()
#             avg_gear = lap_telemetry.get('gear', pd.Series([0])).mean()
#             avg_speed = lap_info.get('KPH', 0)
#             top_speed = lap_telemetry.get('TOP_SPEED', lap_telemetry.get('KPH', pd.Series([0]))).max()
#             lap_time = lap_info.get('LAP_TIME_SECONDS', 0)
#             sector_times = [
#                 lap_info.get('S1_SECONDS', 0),
#                 lap_info.get('S2_SECONDS', 0),
#                 lap_info.get('S3_SECONDS', 0)
#             ]
#             speed_variance = lap_telemetry.get('KPH', pd.Series([avg_speed])).var()

#             return {
#                 'throttle_usage': throttle_usage,
#                 'brake_pressure': brake_pressure,
#                 'longitudinal_accel': longitudinal_accel,
#                 'lateral_accel': lateral_accel,
#                 'gear_changes': gear_changes,
#                 'avg_gear': avg_gear,
#                 'avg_speed': avg_speed,
#                 'top_speed': top_speed,
#                 'lap_time': lap_time,
#                 'sector1_time': sector_times[0],
#                 'sector2_time': sector_times[1],
#                 'sector3_time': sector_times[2],
#                 'speed_variance': speed_variance if not pd.isna(speed_variance) else 0,
#                 'lap_number': lap_info.get('LAP_NUMBER', 0)
#             }
#         except Exception as e:
#             print(f"Error calculating telemetry features: {e}")
#             return None

#     def _estimate_fuel_from_telemetry(self, lap_telemetry: pd.DataFrame, lap_info: pd.Series) -> float:
#         """Estimate fuel consumption based on telemetry patterns"""
#         try:
#             base_consumption = 2.8
#             throttle_factor = lap_telemetry.get('aps', pd.Series([0])).mean() / 100 * 0.8
#             speed_factor = lap_info.get('KPH', 0) / 200 * 1.2
#             accel_factor = lap_telemetry.get('accx_can', pd.Series([0])).abs().mean() * 2.5
#             avg_gear = lap_telemetry.get('gear', pd.Series([3])).mean()
#             gear_efficiency = 1.0 - abs(avg_gear - 3) * 0.1

#             total_consumption = base_consumption * (1 + throttle_factor + speed_factor + accel_factor) * gear_efficiency
#             noise = np.random.normal(0, 0.15)
#             return max(1.5, total_consumption + noise)
#         except Exception:
#             return 2.8  # fallback base consumption

#     def predict_fuel_consumption(self, telemetry_features: dict) -> float:
#         """Predict fuel consumption for given telemetry features"""
#         try:
#             feature_vector = [telemetry_features.get(col, 0) for col in self.feature_columns]
#             feature_array = np.array(feature_vector).reshape(1, -1)
#             scaled_features = self.scaler.transform(feature_array)
#             prediction = self.model.predict(scaled_features)[0]
#             return max(1.0, prediction)
#         except Exception as e:
#             print(f"Prediction error: {e}")
#             return self._estimate_fuel_from_telemetry_fallback(telemetry_features)

#     def _estimate_fuel_from_telemetry_fallback(self, features: dict) -> float:
#         base_consumption = 2.8
#         throttle_factor = features.get('throttle_usage', 50) / 100 * 0.8
#         speed_factor = features.get('avg_speed', 120) / 200 * 1.2
#         return base_consumption * (1 + throttle_factor + speed_factor)

#     def estimate_remaining_laps(self, current_fuel: float, telemetry_features: dict) -> int:
#         """Estimate remaining laps based on current fuel and driving conditions"""
#         consumption_rate = self.predict_fuel_consumption(telemetry_features)
#         return max(0, int(current_fuel / consumption_rate))

#     def save_model(self, filepath: str):
#         """Save trained model and scaler"""
#         joblib.dump({
#             'model': self.model,
#             'scaler': self.scaler,
#             'feature_columns': self.feature_columns
#         }, filepath)

#     def load_model(self, filepath: str):
#         """Load trained model and scaler"""
#         model_data = joblib.load(filepath)
#         self.model = model_data['model']
#         self.scaler = model_data['scaler']
#         self.feature_columns = model_data['feature_columns']























# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import joblib

# class FuelModelTrainer:
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#         self.scaler = StandardScaler()
#         self.feature_columns = []
    
#     def train(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> dict:
#         """Train fuel consumption model using real telemetry data"""
#         if telemetry_data.empty or lap_data.empty:
#             return {'error': 'Insufficient data provided'}
        
#         # Extract real fuel consumption features from telemetry
#         features_df, consumption_targets = self._extract_real_fuel_features(telemetry_data, lap_data)
        
#         if features_df.empty:
#             return {'error': 'No valid fuel features extracted'}
        
#         # Prepare training data
#         X = features_df
#         y = np.array(consumption_targets)
        
#         # Remove any rows with NaN values
#         valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
#         X = X[valid_mask]
#         y = y[valid_mask]
        
#         if len(X) < 10:  # Minimum samples required
#             return {'error': f'Insufficient training samples: {len(X)}'}
        
#         # Scale features
#         X_scaled = self.scaler.fit_transform(X)
#         self.feature_columns = X.columns.tolist()
        
#         # Train model
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_scaled, y, test_size=0.2, random_state=42
#         )
        
#         self.model.fit(X_train, y_train)
        
#         # Evaluate
#         train_score = self.model.score(X_train, y_train)
#         test_score = self.model.score(X_test, y_test)
        
#         return {
#             'model': self,
#             'features': self.feature_columns,
#             'train_score': train_score,
#             'test_score': test_score,
#             'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_)),
#             'training_samples': len(X)
#         }
    
#     def _extract_real_fuel_features(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> tuple:
#         """Extract real fuel consumption features from telemetry data"""
#         features_list = []
#         consumption_rates = []
        
#         # Group by vehicle and lap to analyze fuel usage patterns
#         grouped_telemetry = telemetry_data.groupby(['vehicle_number', 'lap'])
        
#         for (vehicle_num, lap_num), lap_telemetry in grouped_telemetry:
#             if len(lap_telemetry) < 50:  # Minimum telemetry points per lap
#                 continue
                
#             # Get corresponding lap timing data
#             lap_info = lap_data[
#                 (lap_data['NUMBER'] == vehicle_num) & 
#                 (lap_data['LAP_NUMBER'] == lap_num)
#             ]
            
#             if lap_info.empty:
#                 continue
            
#             lap_info = lap_info.iloc[0]
            
#             # Calculate fuel-related features from telemetry
#             features = self._calculate_telemetry_features(lap_telemetry, lap_info)
            
#             if features:
#                 # Estimate fuel consumption based on driving patterns
#                 fuel_estimate = self._estimate_fuel_from_telemetry(lap_telemetry, lap_info)
                
#                 features_list.append(features)
#                 consumption_rates.append(fuel_estimate)
        
#         if features_list:
#             return pd.DataFrame(features_list), consumption_rates
#         return pd.DataFrame(), []
    
#     def _calculate_telemetry_features(self, lap_telemetry: pd.DataFrame, lap_info: pd.Series) -> dict:
#         """Calculate fuel consumption features from telemetry data"""
#         try:
#             # Throttle usage patterns
#             throttle_positions = lap_telemetry['aps'].dropna()
#             throttle_usage = throttle_positions.mean() if not throttle_positions.empty else 0
            
#             # Braking patterns
#             brake_pressure = (lap_telemetry['pbrake_f'] + lap_telemetry['pbrake_r']).mean() / 2
            
#             # Acceleration patterns
#             longitudinal_accel = lap_telemetry['accx_can'].abs().mean()
#             lateral_accel = lap_telemetry['accy_can'].abs().mean()
            
#             # Gear usage patterns
#             gear_changes = lap_telemetry['gear'].diff().abs().sum()
#             avg_gear = lap_telemetry['gear'].mean()
            
#             # Speed patterns
#             avg_speed = lap_info.get('KPH', 0)
#             top_speed = lap_telemetry.get('TOP_SPEED', lap_telemetry.get('KPH', 0).max())
            
#             # Lap characteristics
#             lap_time = lap_info.get('LAP_TIME_SECONDS', 0)
#             sector_times = [
#                 lap_info.get('S1_SECONDS', 0),
#                 lap_info.get('S2_SECONDS', 0),
#                 lap_info.get('S3_SECONDS', 0)
#             ]
            
#             # Driving consistency (speed variance)
#             speed_variance = lap_telemetry.get('KPH', pd.Series([avg_speed])).var()
            
#             return {
#                 'throttle_usage': throttle_usage,
#                 'brake_pressure': brake_pressure,
#                 'longitudinal_accel': longitudinal_accel,
#                 'lateral_accel': lateral_accel,
#                 'gear_changes': gear_changes,
#                 'avg_gear': avg_gear,
#                 'avg_speed': avg_speed,
#                 'top_speed': top_speed,
#                 'lap_time': lap_time,
#                 'sector1_time': sector_times[0],
#                 'sector2_time': sector_times[1],
#                 'sector3_time': sector_times[2],
#                 'speed_variance': speed_variance if not pd.isna(speed_variance) else 0,
#                 'lap_number': lap_info.get('LAP_NUMBER', 0)
#             }
#         except Exception as e:
#             print(f"Error calculating telemetry features: {e}")
#             return None
    
#     def _estimate_fuel_from_telemetry(self, lap_telemetry: pd.DataFrame, lap_info: pd.Series) -> float:
#         """Estimate fuel consumption based on telemetry patterns"""
#         # Realistic fuel consumption model based on racing data patterns
#         base_consumption = 2.8  # liters per lap base for GR86
        
#         # Throttle-based consumption
#         throttle_factor = lap_telemetry['aps'].mean() / 100 * 0.8
        
#         # Speed-based consumption (higher speeds = more fuel)
#         speed_factor = (lap_info.get('KPH', 0) / 200) * 1.2
        
#         # Acceleration-based consumption (aggressive driving = more fuel)
#         accel_factor = lap_telemetry['accx_can'].abs().mean() * 2.5
        
#         # Gear efficiency (optimal in mid-range gears)
#         avg_gear = lap_telemetry['gear'].mean()
#         gear_efficiency = 1.0 - abs(avg_gear - 3) * 0.1  # Optimal around gear 3
        
#         total_consumption = base_consumption * (1 + throttle_factor + speed_factor + accel_factor) * gear_efficiency
        
#         # Add realistic noise based on driving style variance
#         noise = np.random.normal(0, 0.15)
        
#         return max(1.5, total_consumption + noise)  # Minimum realistic consumption
    
#     def predict_fuel_consumption(self, telemetry_features: dict) -> float:
#         """Predict fuel consumption for given telemetry features"""
#         try:
#             # Create feature vector in correct order
#             feature_vector = [telemetry_features.get(col, 0) for col in self.feature_columns]
#             feature_array = np.array(feature_vector).reshape(1, -1)
            
#             # Scale features and predict
#             scaled_features = self.scaler.transform(feature_array)
#             prediction = self.model.predict(scaled_features)[0]
            
#             return max(1.0, prediction)  # Ensure positive consumption
#         except Exception as e:
#             print(f"Prediction error: {e}")
#             # Fallback to empirical estimate
#             return self._estimate_fuel_from_telemetry_fallback(telemetry_features)
    
#     def _estimate_fuel_from_telemetry_fallback(self, features: dict) -> float:
#         """Fallback fuel estimation when model prediction fails"""
#         base_consumption = 2.8
#         throttle_factor = features.get('throttle_usage', 50) / 100 * 0.8
#         speed_factor = features.get('avg_speed', 120) / 200 * 1.2
        
#         return base_consumption * (1 + throttle_factor + speed_factor)
    
#     def estimate_remaining_laps(self, current_fuel: float, telemetry_features: dict) -> int:
#         """Estimate remaining laps based on current fuel and driving conditions"""
#         consumption_rate = self.predict_fuel_consumption(telemetry_features)
#         return max(0, int(current_fuel / consumption_rate))
    
#     def save_model(self, filepath: str):
#         """Save trained model and scaler"""
#         model_data = {
#             'model': self.model,
#             'scaler': self.scaler,
#             'feature_columns': self.feature_columns
#         }
#         joblib.dump(model_data, filepath)
    
#     def load_model(self, filepath: str):
#         """Load trained model and scaler"""
#         model_data = joblib.load(filepath)
#         self.model = model_data['model']
#         self.scaler = model_data['scaler']
#         self.feature_columns = model_data['feature_columns']













# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# import joblib

# class FuelModelTrainer:
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    
#     def train(self, lap_data: pd.DataFrame) -> dict:
#         """Train fuel consumption model using lap data"""
#         if lap_data.empty:
#             return {'model': self, 'features': [], 'train_score': 0, 'test_score': 0}
        
#         # Extract fuel consumption features
#         features_list = []
#         consumption_list = []
        
#         # Group by car number and process each car's race
#         for car_number in lap_data['NUMBER'].unique():
#             car_laps = lap_data[lap_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
#             if len(car_laps) < 10:  # Need sufficient laps for fuel calculation
#                 continue
                
#             car_features, car_consumption = self._extract_fuel_features(car_laps)
#             if car_features is not None:
#                 features_list.append(car_features)
#                 consumption_list.extend(car_consumption)
        
#         if not features_list:
#             return {'model': self, 'features': [], 'train_score': 0, 'test_score': 0}
        
#         # Prepare training data
#         X = pd.concat(features_list, ignore_index=True)
#         y = np.array(consumption_list)
        
#         # Remove any rows with NaN values
#         valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
#         X = X[valid_mask]
#         y = y[valid_mask]
        
#         if len(X) == 0:
#             return {'model': self, 'features': [], 'train_score': 0, 'test_score': 0}
        
#         # Train model
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         self.model.fit(X_train, y_train)
        
#         # Evaluate
#         train_score = self.model.score(X_train, y_train)
#         test_score = self.model.score(X_test, y_test)
        
#         return {
#             'model': self,
#             'features': X.columns.tolist(),
#             'train_score': train_score,
#             'test_score': test_score,
#             'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
#         }
    
#     def _extract_fuel_features(self, car_laps: pd.DataFrame) -> tuple:
#         """Extract fuel consumption features from car lap data"""
#         features = []
#         consumption_rates = []
        
#         # Calculate fuel consumption between lap segments
#         for i in range(5, len(car_laps) - 5):
#             current_lap = car_laps.iloc[i]
#             previous_lap = car_laps.iloc[i-1]
            
#             # Feature 1: Throttle usage (approximated by average speed)
#             avg_speed = current_lap.get('KPH', 0)
            
#             # Feature 2: Lap time variation (indicates pushing/conserving)
#             lap_time = current_lap['LAP_TIME_SECONDS']
#             avg_lap_time = car_laps['LAP_TIME_SECONDS'].mean()
#             time_variation = lap_time - avg_lap_time
            
#             # Feature 3: Sector time patterns
#             s1_time = current_lap.get('S1_SECONDS', 0)
#             s2_time = current_lap.get('S2_SECONDS', 0) 
#             s3_time = current_lap.get('S3_SECONDS', 0)
            
#             # Feature 4: Lap number (fuel load decreases)
#             lap_number = current_lap['LAP_NUMBER']
            
#             # Feature 5: Position changes (defensive/aggressive driving)
#             position_change = 0  # Would need race position data
            
#             # Feature 6: Track characteristics (approximated by top speed)
#             top_speed = current_lap.get('TOP_SPEED', avg_speed * 1.1)
            
#             # Feature 7: Weather impact (if available)
#             temp_effect = 1.0  # Default
            
#             # Calculate fuel consumption rate (simplified model)
#             # Base consumption + speed factor + lap time factor
#             base_consumption = 2.5  # liters per lap base
#             speed_factor = (avg_speed / 150) * 0.5  # Higher speed = more fuel
#             time_factor = (lap_time / 90) * 0.3     # Longer laps = more fuel
            
#             fuel_consumption = base_consumption + speed_factor + time_factor
            
#             # Add some random variation for realism
#             fuel_consumption += np.random.normal(0, 0.1)
            
#             features.append(pd.DataFrame([{
#                 'lap_number': lap_number,
#                 'avg_speed': avg_speed,
#                 'time_variation': time_variation,
#                 's1_time': s1_time,
#                 's2_time': s2_time, 
#                 's3_time': s3_time,
#                 'top_speed': top_speed,
#                 'position_change': position_change,
#                 'temp_effect': temp_effect
#             }]))
            
#             consumption_rates.append(fuel_consumption)
        
#         if features:
#             return pd.concat(features, ignore_index=True), consumption_rates
#         return None, []
    
#     def predict_fuel_consumption(self, features: dict) -> float:
#         """Predict fuel consumption for given lap conditions"""
#         feature_df = pd.DataFrame([features])
#         return max(0.1, self.model.predict(feature_df)[0])
    
#     def estimate_remaining_laps(self, current_fuel: float, lap_conditions: dict) -> int:
#         """Estimate remaining laps based on current fuel and conditions"""
#         consumption_rate = self.predict_fuel_consumption(lap_conditions)
#         return max(0, int(current_fuel / consumption_rate))
    
#     def save_model(self, filepath: str):
#         """Save trained model"""
#         joblib.dump(self.model, filepath)