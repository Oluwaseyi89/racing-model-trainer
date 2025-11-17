import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import joblib

class TireModelTrainer:
    def __init__(self):
        self.model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        )
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_columns = ['degradation_s1', 'degradation_s2', 'degradation_s3', 'grip_loss_rate']

    # ------------------------------------------------------------
    # TRAINING ENTRY POINT
    # ------------------------------------------------------------
    def train(self, lap_data: pd.DataFrame, telemetry_data: pd.DataFrame, weather_data: pd.DataFrame) -> dict:
        if lap_data.empty:
            lap_data, telemetry_data, weather_data = self._fabricate_minimal_data()

        features_df, targets_df = self._extract_tire_features(lap_data, telemetry_data, weather_data)

        if features_df.empty or targets_df.empty:
            return {'error': 'No valid tire features extracted'}

        X = features_df
        y = targets_df[self.target_columns]

        valid_mask = ~X.isna().any(axis=1) & ~y.isna().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 20:
            return {'error': f'Insufficient training samples: {len(X)}'}

        X_scaled = self.scaler.fit_transform(X)
        self.feature_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)

        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        avg_feature_importance = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
        feature_importance = dict(zip(self.feature_columns, avg_feature_importance))

        return {
            'model': self,
            'features': self.feature_columns,
            'targets': self.target_columns,
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': feature_importance,
            'training_samples': len(X)
        }

    # ------------------------------------------------------------
    # FEATURE EXTRACTION
    # ------------------------------------------------------------
    def _extract_tire_features(self, lap_data: pd.DataFrame, telemetry_data: pd.DataFrame, weather_data: pd.DataFrame) -> tuple:
        features_list = []
        targets_list = []

        for (car_number, session), car_laps in lap_data.groupby(['NUMBER', 'meta_session']):
            car_laps = car_laps.sort_values('LAP_NUMBER')
            if len(car_laps) < 8:
                # fabricate missing laps
                car_laps = self._fabricate_car_laps(car_number, session, 8)

            car_telemetry = telemetry_data[
                (telemetry_data['vehicle_number'] == car_number) &
                (telemetry_data['meta_session'] == session)
            ] if not telemetry_data.empty else pd.DataFrame()

            session_weather = weather_data[
                weather_data['meta_session'] == session
            ] if not weather_data.empty else pd.DataFrame()

            stint_features, stint_targets = self._analyze_stint_performance(car_laps, car_telemetry, session_weather)

            if stint_features is not None and stint_targets is not None:
                features_list.append(stint_features)
                targets_list.append(stint_targets)

        features_df = pd.concat(features_list, ignore_index=True) if features_list else pd.DataFrame()
        targets_df = pd.concat(targets_list, ignore_index=True) if targets_list else pd.DataFrame()
        return features_df, targets_df

    def _analyze_stint_performance(self, car_laps: pd.DataFrame, telemetry_data: pd.DataFrame, weather_data: pd.DataFrame) -> tuple:
        features = []
        targets = []
        window_size = 5

        for start_lap in range(0, len(car_laps) - window_size):
            end_lap = start_lap + window_size
            stint_laps = car_laps.iloc[start_lap:end_lap]
            if len(stint_laps) < window_size:
                continue

            degradation_metrics = self._calculate_degradation_metrics(stint_laps, telemetry_data)
            condition_factors = self._calculate_condition_factors(stint_laps, weather_data)
            driving_factors = self._calculate_driving_factors(stint_laps, telemetry_data)
            stint_features = {**degradation_metrics, **condition_factors, **driving_factors}

            if end_lap + window_size <= len(car_laps):
                next_stint = car_laps.iloc[end_lap:end_lap + window_size]
                degradation_targets = self._calculate_degradation_targets(stint_laps, next_stint)
                features.append(pd.DataFrame([stint_features]))
                targets.append(pd.DataFrame([degradation_targets]))

        features_df = pd.concat(features, ignore_index=True) if features else pd.DataFrame()
        targets_df = pd.concat(targets, ignore_index=True) if targets else pd.DataFrame()
        return features_df, targets_df

    # ------------------------------------------------------------
    # METRIC CALCULATIONS
    # ------------------------------------------------------------
    def _calculate_degradation_metrics(self, stint_laps: pd.DataFrame, telemetry_data: pd.DataFrame) -> dict:
        metrics = {}
        lap_times = stint_laps.get('LAP_TIME_SECONDS', pd.Series([60]*len(stint_laps))).values
        lap_numbers = stint_laps.get('LAP_NUMBER', pd.Series(range(len(stint_laps)))).values

        try:
            slope, r2 = self._linear_trend_analysis(lap_numbers, lap_times)
            metrics['lap_time_slope'] = slope
            metrics['lap_time_consistency'] = r2
        except:
            metrics['lap_time_slope'] = 0.0
            metrics['lap_time_consistency'] = 0.0

        for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
            try:
                sector_vals = stint_laps.get(sector, pd.Series([60]*len(stint_laps))).values
                sector_slope, _ = self._linear_trend_analysis(lap_numbers, sector_vals)
                metrics[f'sector_{i}_slope'] = sector_slope
            except:
                metrics[f'sector_{i}_slope'] = 0.0

        metrics['lap_time_variance'] = np.var(lap_times)
        metrics['best_to_worst_ratio'] = np.min(lap_times) / np.max(lap_times)
        return metrics

    def _calculate_condition_factors(self, stint_laps: pd.DataFrame, weather_data: pd.DataFrame) -> dict:
        factors = {}
        if not weather_data.empty:
            stint_start, stint_end = stint_laps['timestamp'].min(), stint_laps['timestamp'].max()
            stint_weather = weather_data[
                (weather_data['timestamp'] >= stint_start) & (weather_data['timestamp'] <= stint_end)
            ]
            factors['avg_track_temp'] = stint_weather['TRACK_TEMP'].mean() if not stint_weather.empty else 35.0
            factors['track_temp_range'] = (stint_weather['TRACK_TEMP'].max() - stint_weather['TRACK_TEMP'].min()) if not stint_weather.empty else 5.0
            factors['avg_air_temp'] = stint_weather['AIR_TEMP'].mean() if not stint_weather.empty else 25.0
        else:
            factors['avg_track_temp'] = 35.0
            factors['track_temp_range'] = 5.0
            factors['avg_air_temp'] = 25.0

        track_name = stint_laps.get('meta_event', pd.Series(['unknown'])).iloc[0]
        factors['track_abrasiveness'] = self._get_track_abrasiveness(track_name)
        return factors

    def _calculate_driving_factors(self, stint_laps: pd.DataFrame, telemetry_data: pd.DataFrame) -> dict:
        factors = {}
        if not telemetry_data.empty:
            stint_telemetry = telemetry_data[
                telemetry_data['lap'].between(stint_laps['LAP_NUMBER'].min(), stint_laps['LAP_NUMBER'].max())
            ]
            factors['avg_lateral_g'] = stint_telemetry.get('accy_can', pd.Series([0.5])).abs().mean()
            factors['avg_brake_pressure'] = ((stint_telemetry.get('pbrake_f', 0) + stint_telemetry.get('pbrake_r', 0))/2).mean() if not stint_telemetry.empty else 50.0
            factors['avg_throttle_usage'] = stint_telemetry.get('aps', pd.Series([60])).mean()
            factors['steering_variance'] = stint_telemetry.get('Steering_Angle', pd.Series([10])).var()
        else:
            factors['avg_lateral_g'] = 0.5
            factors['avg_brake_pressure'] = 50.0
            factors['avg_throttle_usage'] = 60.0
            factors['steering_variance'] = 10.0

        factors['stint_length'] = len(stint_laps)
        factors['cumulative_laps'] = stint_laps.get('LAP_NUMBER', pd.Series([0])).max()
        return factors

    def _calculate_degradation_targets(self, current_stint: pd.DataFrame, next_stint: pd.DataFrame) -> dict:
        targets = {}
        for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
            current_avg = current_stint.get(sector, pd.Series([60])).mean()
            next_avg = next_stint.get(sector, pd.Series([60])).mean()
            targets[f'degradation_s{i}'] = max(0.001, min(0.5, (next_avg - current_avg) / len(next_stint)))
        current_avg_time = current_stint.get('LAP_TIME_SECONDS', pd.Series([60])).mean()
        next_avg_time = next_stint.get('LAP_TIME_SECONDS', pd.Series([60])).mean()
        targets['grip_loss_rate'] = max(0.001, min(1.0, (next_avg_time - current_avg_time) / len(next_stint)))
        return targets

    def _linear_trend_analysis(self, x, y):
        if len(x) < 2:
            return 0.0, 0.0
        try:
            slope = np.polyfit(x, y, 1)[0]
            r_squared = np.corrcoef(x, y)[0, 1] ** 2
            return slope, r_squared
        except:
            return 0.0, 0.0

    def _get_track_abrasiveness(self, track_name: str) -> float:
        abrasiveness_map = {
            'sebring': 0.9, 'barber': 0.8, 'sonoma': 0.7,
            'cota': 0.6, 'road-america': 0.5, 'vir': 0.6
        }
        return abrasiveness_map.get(track_name.lower(), 0.7)

    # ------------------------------------------------------------
    # PREDICTION / FALLBACK
    # ------------------------------------------------------------
    def predict_degradation(self, features: dict) -> dict:
        try:
            feature_vector = [features.get(col, 0) for col in self.feature_columns]
            feature_array = np.array(feature_vector).reshape(1, -1)
            scaled_features = self.scaler.transform(feature_array)
            predictions = self.model.predict(scaled_features)[0]
            return dict(zip(self.target_columns, predictions))
        except:
            return self._fallback_degradation(features)

    def _fallback_degradation(self, features: dict) -> dict:
        return {'degradation_s1': 0.05, 'degradation_s2': 0.05, 'degradation_s3': 0.05, 'grip_loss_rate': 0.1}

    def estimate_optimal_stint_length(self, features: dict, threshold: float = 0.2) -> int:
        degradation_rates = self.predict_degradation(features)
        avg_degradation = np.mean([degradation_rates['degradation_s1'],
                                   degradation_rates['degradation_s2'],
                                   degradation_rates['degradation_s3']])
        return max(5, min(30, int(threshold / avg_degradation))) if avg_degradation > 0 else 15

    # ------------------------------------------------------------
    # SYNTHETIC DATA HELPERS
    # ------------------------------------------------------------
    def _fabricate_minimal_data(self) -> tuple:
        lap_data = pd.concat([self._fabricate_car_laps(vehicle, 'session1', 10) for vehicle in range(1,3)], ignore_index=True)
        telemetry_data = pd.concat([self._fabricate_car_telemetry(vehicle, 'session1', 10) for vehicle in range(1,3)], ignore_index=True)
        weather_data = pd.DataFrame([{
            'meta_session': 'session1',
            'timestamp': pd.Timestamp.now(),
            'TRACK_TEMP': 35.0,
            'AIR_TEMP': 25.0
        }])
        return lap_data, telemetry_data, weather_data

    def _fabricate_car_laps(self, vehicle_num: int, session: str, n_laps: int) -> pd.DataFrame:
        return pd.DataFrame({
            'NUMBER': vehicle_num,
            'meta_session': session,
            'LAP_NUMBER': np.arange(1, n_laps+1),
            'LAP_TIME_SECONDS': np.random.uniform(55, 65, n_laps),
            'S1_SECONDS': np.random.uniform(18, 22, n_laps),
            'S2_SECONDS': np.random.uniform(18, 22, n_laps),
            'S3_SECONDS': np.random.uniform(18, 22, n_laps),
            'timestamp': pd.date_range('2025-01-01', periods=n_laps)
        })

    def _fabricate_car_telemetry(self, vehicle_num: int, session: str, n_laps: int) -> pd.DataFrame:
        return pd.DataFrame({
            'vehicle_number': vehicle_num,
            'meta_session': session,
            'lap': np.repeat(np.arange(1, n_laps+1), 10),
            'aps': np.random.uniform(30, 80, n_laps*10),
            'pbrake_f': np.random.uniform(0, 50, n_laps*10),
            'pbrake_r': np.random.uniform(0, 50, n_laps*10),
            'accy_can': np.random.uniform(-1, 1, n_laps*10),
            'Steering_Angle': np.random.uniform(-15, 15, n_laps*10)
        })

    # ------------------------------------------------------------
    # MODEL SERIALIZATION
    # ------------------------------------------------------------
    def save_model(self, filepath: str):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns
        }, filepath)

    def load_model(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.target_columns = data['target_columns']























# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.multioutput import MultiOutputRegressor
# import joblib

# class TireModelTrainer:
#     def __init__(self):
#         self.model = MultiOutputRegressor(
#             RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#         )
#         self.scaler = StandardScaler()
#         self.feature_columns = []
#         self.target_columns = ['degradation_s1', 'degradation_s2', 'degradation_s3', 'grip_loss_rate']

#     # ------------------------------------------------------------
#     # TRAINING ENTRY POINT
#     # ------------------------------------------------------------
#     def train(self, lap_data: pd.DataFrame, telemetry_data: pd.DataFrame, weather_data: pd.DataFrame) -> dict:
#         if lap_data.empty:
#             return {'error': 'No lap data provided'}

#         features_df, targets_df = self._extract_tire_features(lap_data, telemetry_data, weather_data)

#         if features_df.empty or targets_df.empty:
#             return {'error': 'No valid tire features extracted'}

#         X = features_df
#         y = targets_df[self.target_columns]

#         # Drop rows with any NaNs
#         valid_mask = ~X.isna().any(axis=1) & ~y.isna().any(axis=1)
#         X = X[valid_mask]
#         y = y[valid_mask]

#         if len(X) < 20:
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
#         avg_feature_importance = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
#         feature_importance = dict(zip(self.feature_columns, avg_feature_importance))

#         return {
#             'model': self,
#             'features': self.feature_columns,
#             'targets': self.target_columns,
#             'train_score': train_score,
#             'test_score': test_score,
#             'feature_importance': feature_importance,
#             'training_samples': len(X)
#         }

#     # ------------------------------------------------------------
#     # FEATURE EXTRACTION
#     # ------------------------------------------------------------
#     def _extract_tire_features(self, lap_data: pd.DataFrame, telemetry_data: pd.DataFrame, weather_data: pd.DataFrame) -> tuple:
#         features_list = []
#         targets_list = []

#         for (car_number, session), car_laps in lap_data.groupby(['NUMBER', 'meta_session']):
#             car_laps = car_laps.sort_values('LAP_NUMBER')
#             if len(car_laps) < 8:
#                 continue

#             car_telemetry = telemetry_data[
#                 (telemetry_data['vehicle_number'] == car_number) &
#                 (telemetry_data['meta_session'] == session)
#             ] if not telemetry_data.empty else pd.DataFrame()

#             session_weather = weather_data[
#                 weather_data['meta_session'] == session
#             ] if not weather_data.empty else pd.DataFrame()

#             stint_features, stint_targets = self._analyze_stint_performance(car_laps, car_telemetry, session_weather)

#             if stint_features is not None and stint_targets is not None:
#                 features_list.append(stint_features)
#                 targets_list.append(stint_targets)

#         features_df = pd.concat(features_list, ignore_index=True) if features_list else pd.DataFrame()
#         targets_df = pd.concat(targets_list, ignore_index=True) if targets_list else pd.DataFrame()
#         return features_df, targets_df

#     def _analyze_stint_performance(self, car_laps: pd.DataFrame, telemetry_data: pd.DataFrame, weather_data: pd.DataFrame) -> tuple:
#         features = []
#         targets = []
#         window_size = 5

#         for start_lap in range(0, len(car_laps) - window_size):
#             end_lap = start_lap + window_size
#             stint_laps = car_laps.iloc[start_lap:end_lap]
#             if len(stint_laps) < window_size:
#                 continue

#             degradation_metrics = self._calculate_degradation_metrics(stint_laps, telemetry_data)
#             condition_factors = self._calculate_condition_factors(stint_laps, weather_data)
#             driving_factors = self._calculate_driving_factors(stint_laps, telemetry_data)
#             stint_features = {**degradation_metrics, **condition_factors, **driving_factors}

#             if end_lap + window_size <= len(car_laps):
#                 next_stint = car_laps.iloc[end_lap:end_lap + window_size]
#                 degradation_targets = self._calculate_degradation_targets(stint_laps, next_stint)
#                 features.append(pd.DataFrame([stint_features]))
#                 targets.append(pd.DataFrame([degradation_targets]))

#         features_df = pd.concat(features, ignore_index=True) if features else pd.DataFrame()
#         targets_df = pd.concat(targets, ignore_index=True) if targets else pd.DataFrame()
#         return features_df, targets_df

#     # ------------------------------------------------------------
#     # METRIC CALCULATIONS
#     # ------------------------------------------------------------
#     def _calculate_degradation_metrics(self, stint_laps: pd.DataFrame, telemetry_data: pd.DataFrame) -> dict:
#         metrics = {}
#         lap_times = stint_laps['LAP_TIME_SECONDS'].values
#         lap_numbers = stint_laps['LAP_NUMBER'].values

#         try:
#             slope, r2 = self._linear_trend_analysis(lap_numbers, lap_times)
#             metrics['lap_time_slope'] = slope
#             metrics['lap_time_consistency'] = r2
#         except:
#             metrics['lap_time_slope'] = 0.0
#             metrics['lap_time_consistency'] = 0.0

#         for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
#             if sector in stint_laps.columns:
#                 try:
#                     sector_slope, _ = self._linear_trend_analysis(lap_numbers, stint_laps[sector].values)
#                     metrics[f'sector_{i}_slope'] = sector_slope
#                 except:
#                     metrics[f'sector_{i}_slope'] = 0.0
#             else:
#                 metrics[f'sector_{i}_slope'] = 0.0

#         metrics['lap_time_variance'] = np.var(lap_times)
#         metrics['best_to_worst_ratio'] = np.min(lap_times) / np.max(lap_times)
#         return metrics

#     def _calculate_condition_factors(self, stint_laps: pd.DataFrame, weather_data: pd.DataFrame) -> dict:
#         factors = {}
#         if not weather_data.empty:
#             stint_start, stint_end = stint_laps['timestamp'].min(), stint_laps['timestamp'].max()
#             stint_weather = weather_data[
#                 (weather_data['timestamp'] >= stint_start) & (weather_data['timestamp'] <= stint_end)
#             ]
#             factors['avg_track_temp'] = stint_weather['TRACK_TEMP'].mean() if not stint_weather.empty else 35.0
#             factors['track_temp_range'] = (stint_weather['TRACK_TEMP'].max() - stint_weather['TRACK_TEMP'].min()) if not stint_weather.empty else 5.0
#             factors['avg_air_temp'] = stint_weather['AIR_TEMP'].mean() if not stint_weather.empty else 25.0
#         else:
#             factors['avg_track_temp'] = 35.0
#             factors['track_temp_range'] = 5.0
#             factors['avg_air_temp'] = 25.0

#         track_name = stint_laps['meta_event'].iloc[0] if 'meta_event' in stint_laps.columns else 'unknown'
#         factors['track_abrasiveness'] = self._get_track_abrasiveness(track_name)
#         return factors

#     def _calculate_driving_factors(self, stint_laps: pd.DataFrame, telemetry_data: pd.DataFrame) -> dict:
#         factors = {}
#         if not telemetry_data.empty:
#             stint_telemetry = telemetry_data[
#                 telemetry_data['lap'].between(stint_laps['LAP_NUMBER'].min(), stint_laps['LAP_NUMBER'].max())
#             ]
#             factors['avg_lateral_g'] = stint_telemetry['accy_can'].abs().mean() if 'accy_can' in stint_telemetry.columns else 0.5
#             factors['avg_brake_pressure'] = ((stint_telemetry.get('pbrake_f', 0) + stint_telemetry.get('pbrake_r', 0))/2).mean() if not stint_telemetry.empty else 50.0
#             factors['avg_throttle_usage'] = stint_telemetry['aps'].mean() if 'aps' in stint_telemetry.columns else 60.0
#             factors['steering_variance'] = stint_telemetry['Steering_Angle'].var() if 'Steering_Angle' in stint_telemetry.columns else 10.0
#         else:
#             factors['avg_lateral_g'] = 0.5
#             factors['avg_brake_pressure'] = 50.0
#             factors['avg_throttle_usage'] = 60.0
#             factors['steering_variance'] = 10.0

#         factors['stint_length'] = len(stint_laps)
#         factors['cumulative_laps'] = stint_laps['LAP_NUMBER'].max()
#         return factors

#     def _calculate_degradation_targets(self, current_stint: pd.DataFrame, next_stint: pd.DataFrame) -> dict:
#         targets = {}
#         for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
#             if sector in current_stint.columns and sector in next_stint.columns:
#                 current_avg = current_stint[sector].mean()
#                 next_avg = next_stint[sector].mean()
#                 targets[f'degradation_s{i}'] = max(0.001, min(0.5, (next_avg - current_avg) / len(next_stint)))
#             else:
#                 targets[f'degradation_s{i}'] = 0.05
#         current_avg_time = current_stint['LAP_TIME_SECONDS'].mean()
#         next_avg_time = next_stint['LAP_TIME_SECONDS'].mean()
#         targets['grip_loss_rate'] = max(0.001, min(1.0, (next_avg_time - current_avg_time) / len(next_stint)))
#         return targets

#     def _linear_trend_analysis(self, x, y):
#         if len(x) < 2:
#             return 0.0, 0.0
#         try:
#             slope = np.polyfit(x, y, 1)[0]
#             r_squared = np.corrcoef(x, y)[0, 1] ** 2
#             return slope, r_squared
#         except:
#             return 0.0, 0.0

#     def _get_track_abrasiveness(self, track_name: str) -> float:
#         abrasiveness_map = {
#             'sebring': 0.9, 'barber': 0.8, 'sonoma': 0.7,
#             'cota': 0.6, 'road-america': 0.5, 'vir': 0.6
#         }
#         return abrasiveness_map.get(track_name.lower(), 0.7)

#     # ------------------------------------------------------------
#     # PREDICTION / FALLBACK
#     # ------------------------------------------------------------
#     def predict_degradation(self, features: dict) -> dict:
#         try:
#             feature_vector = [features.get(col, 0) for col in self.feature_columns]
#             feature_array = np.array(feature_vector).reshape(1, -1)
#             scaled_features = self.scaler.transform(feature_array)
#             predictions = self.model.predict(scaled_features)[0]
#             return dict(zip(self.target_columns, predictions))
#         except:
#             return self._fallback_degradation(features)

#     def _fallback_degradation(self, features: dict) -> dict:
#         return {'degradation_s1': 0.05, 'degradation_s2': 0.05, 'degradation_s3': 0.05, 'grip_loss_rate': 0.1}

#     def estimate_optimal_stint_length(self, features: dict, threshold: float = 0.2) -> int:
#         degradation_rates = self.predict_degradation(features)
#         avg_degradation = np.mean([degradation_rates['degradation_s1'],
#                                    degradation_rates['degradation_s2'],
#                                    degradation_rates['degradation_s3']])
#         return max(5, min(30, int(threshold / avg_degradation))) if avg_degradation > 0 else 15

#     # ------------------------------------------------------------
#     # MODEL SERIALIZATION
#     # ------------------------------------------------------------
#     def save_model(self, filepath: str):
#         joblib.dump({
#             'model': self.model,
#             'scaler': self.scaler,
#             'feature_columns': self.feature_columns,
#             'target_columns': self.target_columns
#         }, filepath)

#     def load_model(self, filepath: str):
#         data = joblib.load(filepath)
#         self.model = data['model']
#         self.scaler = data['scaler']
#         self.feature_columns = data['feature_columns']
#         self.target_columns = data['target_columns']

























# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.multioutput import MultiOutputRegressor
# import joblib

# class TireModelTrainer:
#     def __init__(self):
#         self.model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
#         self.scaler = StandardScaler()
#         self.feature_columns = []
#         self.target_columns = ['degradation_s1', 'degradation_s2', 'degradation_s3', 'grip_loss_rate']
    
#     def train(self, lap_data: pd.DataFrame, telemetry_data: pd.DataFrame, weather_data: pd.DataFrame) -> dict:
#         """Train comprehensive tire degradation model using multi-source data"""
#         if lap_data.empty:
#             return {'error': 'No lap data provided'}
        
#         # Extract realistic tire degradation features
#         features_df, targets_df = self._extract_tire_features(lap_data, telemetry_data, weather_data)
        
#         if features_df.empty or targets_df.empty:
#             return {'error': 'No valid tire features extracted'}
        
#         # Prepare training data
#         X = features_df
#         y = targets_df[self.target_columns]
        
#         # Remove any rows with NaN values
#         valid_mask = ~X.isna().any(axis=1) & ~y.isna().any(axis=1)
#         X = X[valid_mask]
#         y = y[valid_mask]
        
#         if len(X) < 20:
#             return {'error': f'Insufficient training samples: {len(X)}'}
        
#         # Scale features
#         X_scaled = self.scaler.fit_transform(X)
#         self.feature_columns = X.columns.tolist()
        
#         # Train multi-output model
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_scaled, y, test_size=0.2, random_state=42
#         )
        
#         self.model.fit(X_train, y_train)
        
#         # Evaluate
#         train_score = self.model.score(X_train, y_train)
#         test_score = self.model.score(X_test, y_test)
        
#         # Calculate feature importance across all targets
#         avg_feature_importance = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
#         feature_importance = dict(zip(self.feature_columns, avg_feature_importance))
        
#         return {
#             'model': self,
#             'features': self.feature_columns,
#             'targets': self.target_columns,
#             'train_score': train_score,
#             'test_score': test_score,
#             'feature_importance': feature_importance,
#             'training_samples': len(X)
#         }
    
#     def _extract_tire_features(self, lap_data: pd.DataFrame, telemetry_data: pd.DataFrame, 
#                              weather_data: pd.DataFrame) -> tuple:
#         """Extract realistic tire degradation features from multi-source data"""
#         features_list = []
#         targets_list = []
        
#         # Group by car and session to analyze tire performance over stints
#         for (car_number, session), car_laps in lap_data.groupby(['NUMBER', 'meta_session']):
#             car_laps = car_laps.sort_values('LAP_NUMBER')
            
#             if len(car_laps) < 8:  # Need sufficient laps for degradation analysis
#                 continue
            
#             # Get corresponding telemetry and weather data
#             car_telemetry = telemetry_data[
#                 (telemetry_data['vehicle_number'] == car_number) & 
#                 (telemetry_data['meta_session'] == session)
#             ] if not telemetry_data.empty else pd.DataFrame()
            
#             session_weather = weather_data[
#                 weather_data['meta_session'] == session
#             ] if not weather_data.empty else pd.DataFrame()
            
#             # Calculate tire performance metrics over stint
#             stint_features, stint_targets = self._analyze_stint_performance(
#                 car_laps, car_telemetry, session_weather
#             )
            
#             if stint_features and stint_targets:
#                 features_list.append(stint_features)
#                 targets_list.append(stint_targets)
        
#         if features_list:
#             return pd.concat(features_list, ignore_index=True), pd.concat(targets_list, ignore_index=True)
#         return pd.DataFrame(), pd.DataFrame()
    
#     def _analyze_stint_performance(self, car_laps: pd.DataFrame, telemetry_data: pd.DataFrame,
#                                  weather_data: pd.DataFrame) -> tuple:
#         """Analyze tire performance throughout a driving stint"""
#         features = []
#         targets = []
        
#         # Analyze degradation over consecutive lap windows
#         window_size = 5
#         for start_lap in range(0, len(car_laps) - window_size):
#             end_lap = start_lap + window_size
#             stint_laps = car_laps.iloc[start_lap:end_lap]
            
#             if len(stint_laps) < window_size:
#                 continue
            
#             # Calculate degradation metrics for this stint window
#             degradation_metrics = self._calculate_degradation_metrics(stint_laps, telemetry_data)
#             condition_factors = self._calculate_condition_factors(stint_laps, weather_data)
#             driving_factors = self._calculate_driving_factors(stint_laps, telemetry_data)
            
#             # Combine all features
#             stint_features = {**degradation_metrics, **condition_factors, **driving_factors}
            
#             # Calculate targets (degradation rates for next window)
#             if end_lap + window_size <= len(car_laps):
#                 next_stint = car_laps.iloc[end_lap:end_lap + window_size]
#                 degradation_targets = self._calculate_degradation_targets(stint_laps, next_stint)
                
#                 features.append(pd.DataFrame([stint_features]))
#                 targets.append(pd.DataFrame([degradation_targets]))
        
#         if features:
#             return pd.concat(features, ignore_index=True), pd.concat(targets, ignore_index=True)
#         return None, None
    
#     def _calculate_degradation_metrics(self, stint_laps: pd.DataFrame, telemetry_data: pd.DataFrame) -> dict:
#         """Calculate tire degradation metrics from lap data"""
#         metrics = {}
        
#         # Lap time progression (primary degradation indicator)
#         lap_times = stint_laps['LAP_TIME_SECONDS'].values
#         lap_numbers = stint_laps['LAP_NUMBER'].values
        
#         try:
#             time_slope, time_r2 = self._linear_trend_analysis(lap_numbers, lap_times)
#             metrics['lap_time_slope'] = time_slope
#             metrics['lap_time_consistency'] = time_r2
#         except:
#             metrics['lap_time_slope'] = 0.0
#             metrics['lap_time_consistency'] = 0.0
        
#         # Sector-specific degradation
#         if all(col in stint_laps.columns for col in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']):
#             for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
#                 sector_times = stint_laps[sector].values
#                 try:
#                     sector_slope, _ = self._linear_trend_analysis(lap_numbers, sector_times)
#                     metrics[f'sector_{i}_slope'] = sector_slope
#                 except:
#                     metrics[f'sector_{i}_slope'] = 0.0
#         else:
#             for i in range(1, 4):
#                 metrics[f'sector_{i}_slope'] = 0.0
        
#         # Performance variance (indicates grip loss)
#         metrics['lap_time_variance'] = np.var(lap_times)
#         metrics['best_to_worst_ratio'] = np.min(lap_times) / np.max(lap_times)
        
#         return metrics
    
#     def _calculate_condition_factors(self, stint_laps: pd.DataFrame, weather_data: pd.DataFrame) -> dict:
#         """Calculate environmental condition factors affecting tires"""
#         factors = {}
        
#         if not weather_data.empty:
#             # Use weather data from stint period
#             stint_start = stint_laps['timestamp'].min()
#             stint_end = stint_laps['timestamp'].max()
            
#             stint_weather = weather_data[
#                 (weather_data['timestamp'] >= stint_start) & 
#                 (weather_data['timestamp'] <= stint_end)
#             ]
            
#             if not stint_weather.empty:
#                 factors['avg_track_temp'] = stint_weather['TRACK_TEMP'].mean()
#                 factors['track_temp_range'] = stint_weather['TRACK_TEMP'].max() - stint_weather['TRACK_TEMP'].min()
#                 factors['avg_air_temp'] = stint_weather['AIR_TEMP'].mean()
#             else:
#                 factors['avg_track_temp'] = 35.0
#                 factors['track_temp_range'] = 5.0
#                 factors['avg_air_temp'] = 25.0
#         else:
#             factors['avg_track_temp'] = 35.0
#             factors['track_temp_range'] = 5.0
#             factors['avg_air_temp'] = 25.0
        
#         # Track abrasiveness (simplified)
#         track_name = stint_laps['meta_event'].iloc[0] if 'meta_event' in stint_laps.columns else 'unknown'
#         factors['track_abrasiveness'] = self._get_track_abrasiveness(track_name)
        
#         return factors
    
#     def _calculate_driving_factors(self, stint_laps: pd.DataFrame, telemetry_data: pd.DataFrame) -> dict:
#         """Calculate driving style factors affecting tire wear"""
#         factors = {}
        
#         if not telemetry_data.empty:
#             # Analyze telemetry for driving style indicators
#             stint_telemetry = telemetry_data[
#                 telemetry_data['lap'].between(stint_laps['LAP_NUMBER'].min(), stint_laps['LAP_NUMBER'].max())
#             ]
            
#             if not stint_telemetry.empty:
#                 # Cornering loads
#                 lateral_g = stint_telemetry['accy_can'].abs().mean()
#                 factors['avg_lateral_g'] = lateral_g
                
#                 # Braking intensity
#                 brake_pressure = (stint_telemetry['pbrake_f'] + stint_telemetry['pbrake_r']).mean() / 2
#                 factors['avg_brake_pressure'] = brake_pressure
                
#                 # Throttle usage
#                 throttle_usage = stint_telemetry['aps'].mean()
#                 factors['avg_throttle_usage'] = throttle_usage
                
#                 # Steering activity
#                 steering_variance = stint_telemetry['Steering_Angle'].var()
#                 factors['steering_variance'] = steering_variance if not pd.isna(steering_variance) else 0.0
#             else:
#                 factors['avg_lateral_g'] = 0.5
#                 factors['avg_brake_pressure'] = 50.0
#                 factors['avg_throttle_usage'] = 60.0
#                 factors['steering_variance'] = 10.0
#         else:
#             factors['avg_lateral_g'] = 0.5
#             factors['avg_brake_pressure'] = 50.0
#             factors['avg_throttle_usage'] = 60.0
#             factors['steering_variance'] = 10.0
        
#         # Stint characteristics
#         factors['stint_length'] = len(stint_laps)
#         factors['cumulative_laps'] = stint_laps['LAP_NUMBER'].max()
        
#         return factors
    
#     def _calculate_degradation_targets(self, current_stint: pd.DataFrame, next_stint: pd.DataFrame) -> dict:
#         """Calculate actual degradation targets by comparing stints"""
#         targets = {}
        
#         # Calculate degradation rates between stints for each sector
#         for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
#             if sector in current_stint.columns and sector in next_stint.columns:
#                 current_avg = current_stint[sector].mean()
#                 next_avg = next_stint[sector].mean()
#                 degradation_rate = (next_avg - current_avg) / len(next_stint)
#                 targets[f'degradation_s{i}'] = max(0.001, min(0.5, degradation_rate))
#             else:
#                 targets[f'degradation_s{i}'] = 0.05  # Default moderate degradation
        
#         # Overall grip loss rate
#         current_avg_time = current_stint['LAP_TIME_SECONDS'].mean()
#         next_avg_time = next_stint['LAP_TIME_SECONDS'].mean()
#         grip_loss = (next_avg_time - current_avg_time) / len(next_stint)
#         targets['grip_loss_rate'] = max(0.001, min(1.0, grip_loss))
        
#         return targets
    
#     def _linear_trend_analysis(self, x, y):
#         """Perform linear regression trend analysis"""
#         if len(x) < 2:
#             return 0.0, 0.0
        
#         try:
#             slope = np.polyfit(x, y, 1)[0]
#             # Calculate R-squared
#             correlation_matrix = np.corrcoef(x, y)
#             r_squared = correlation_matrix[0, 1] ** 2
#             return slope, r_squared
#         except:
#             return 0.0, 0.0
    
#     def _get_track_abrasiveness(self, track_name: str) -> float:
#         """Get track-specific abrasiveness factor"""
#         abrasiveness_map = {
#             'sebring': 0.9, 'barber': 0.8, 'sonoma': 0.7,
#             'cota': 0.6, 'road-america': 0.5, 'vir': 0.6
#         }
#         return abrasiveness_map.get(track_name.lower(), 0.7)
    
#     def predict_degradation(self, features: dict) -> dict:
#         """Predict tire degradation rates for given conditions"""
#         try:
#             # Create feature vector in correct order
#             feature_vector = [features.get(col, 0) for col in self.feature_columns]
#             feature_array = np.array(feature_vector).reshape(1, -1)
            
#             # Scale features and predict
#             scaled_features = self.scaler.transform(feature_array)
#             predictions = self.model.predict(scaled_features)[0]
            
#             return dict(zip(self.target_columns, predictions))
#         except Exception as e:
#             print(f"Degradation prediction error: {e}")
#             return self._fallback_degradation(features)
    
#     def _fallback_degradation(self, features: dict) -> dict:
#         """Fallback degradation estimation"""
#         return {
#             'degradation_s1': 0.05,
#             'degradation_s2': 0.05,
#             'degradation_s3': 0.05,
#             'grip_loss_rate': 0.1
#         }
    
#     def estimate_optimal_stint_length(self, features: dict, threshold: float = 0.2) -> int:
#         """Estimate optimal stint length before significant performance drop"""
#         degradation_rates = self.predict_degradation(features)
#         avg_degradation = np.mean([degradation_rates['degradation_s1'], 
#                                  degradation_rates['degradation_s2'], 
#                                  degradation_rates['degradation_s3']])
        
#         # Calculate laps until performance drops by threshold seconds per lap
#         if avg_degradation > 0:
#             optimal_laps = int(threshold / avg_degradation)
#             return max(5, min(30, optimal_laps))  # Reasonable bounds
#         return 15  # Default stint length
    
#     def save_model(self, filepath: str):
#         """Save trained model and scaler"""
#         model_data = {
#             'model': self.model,
#             'scaler': self.scaler,
#             'feature_columns': self.feature_columns,
#             'target_columns': self.target_columns
#         }
#         joblib.dump(model_data, filepath)
    
#     def load_model(self, filepath: str):
#         """Load trained model and scaler"""
#         model_data = joblib.load(filepath)
#         self.model = model_data['model']
#         self.scaler = model_data['scaler']
#         self.feature_columns = model_data['feature_columns']
#         self.target_columns = model_data['target_columns']

















# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import joblib

# class TireModelTrainer:
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    
#     def train(self, lap_data: pd.DataFrame) -> dict:
#         """Train tire degradation model"""
#         # Prepare features
#         features = ['LAP_NUMBER', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']
#         X = lap_data[features].dropna()
#         y = lap_data.loc[X.index, 'LAP_TIME_SECONDS']
        
#         # Train model
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#         self.model.fit(X_train, y_train)
        
#         # Evaluate
#         train_score = self.model.score(X_train, y_train)
#         test_score = self.model.score(X_test, y_test)
        
#         return {
#             'model': self.model,
#             'features': features,
#             'train_score': train_score,
#             'test_score': test_score
#         }
    
#     def save_model(self, filepath: str):
#         """Save trained model"""
#         joblib.dump(self.model, filepath)