import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta

class WeatherModelTrainer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def train(self, processed_data: dict) -> dict:
        """Train weather impact model using integrated weather, lap, and telemetry data"""
        features_list = []
        impact_list = []
        
        for session_key, data in processed_data.items():
            if 'weather_data' in data and 'lap_data' in data:
                session_features, session_impacts = self._extract_weather_features(data, session_key)
                if not session_features.empty and len(session_impacts) > 0:
                    features_list.append(session_features)
                    impact_list.append(session_impacts)
        
        if not features_list:
            return {'error': 'No valid weather features extracted'}
        
        # Combine all session data safely
        X = pd.concat(features_list, ignore_index=True)
        y = np.concatenate(impact_list) if impact_list else np.array([])
        
        # Remove NaN values
        if len(y) > 0:
            valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
            X = X[valid_mask]
            y = y[valid_mask]
        
        if len(X) < 20:
            return {'error': f'Insufficient training samples: {len(X)}'}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.feature_columns = X.columns.tolist()
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        
        return {
            'model': self,
            'features': self.feature_columns,
            'train_score': self.model.score(X_train, y_train),
            'test_score': self.model.score(X_test, y_test),
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_)),
            'training_samples': len(X)
        }

    def update_model(self, existing_model_result: dict, new_processed_data: dict) -> dict:
        """Update existing weather model with new data - maintains backward compatibility"""
        try:
            # Extract the trainer from existing model result
            if isinstance(existing_model_result, dict) and 'model' in existing_model_result:
                existing_trainer = existing_model_result['model']
                if hasattr(existing_trainer, 'model') and hasattr(existing_trainer, 'feature_columns'):
                    # Use existing model as starting point
                    self.model = existing_trainer.model
                    self.feature_columns = existing_trainer.feature_columns
                    self.scaler = existing_trainer.scaler
                    
                    # Train with new data (incremental learning)
                    return self.train(new_processed_data)
            
            # Fallback: train new model if existing model structure is invalid
            self.logger.warning("Existing model structure invalid, training new model")
            return self.train(new_processed_data)
            
        except Exception as e:
            # Fallback to full retraining if incremental update fails
            print(f"⚠️ Incremental update failed, performing full retraining: {e}")
            return self.train(new_processed_data)
    
    def _extract_weather_features(self, data: dict, session_key: str) -> tuple:
        """Extract weather impact features with proper handling of missing data"""
        weather_data = data.get('weather_data', pd.DataFrame())
        lap_data = data.get('lap_data', pd.DataFrame())
        telemetry_data = data.get('telemetry_data', pd.DataFrame())
        
        features = []
        impacts = []
        
        if lap_data.empty or weather_data.empty:
            return pd.DataFrame(), np.array([])
        
        weather_data = self._prepare_weather_timestamps(weather_data)
        lap_data = self._prepare_lap_timestamps(lap_data)
        
        for car_number in lap_data['NUMBER'].unique():
            car_laps = lap_data[lap_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
            if len(car_laps) < 5:
                continue
            baseline_time = self._calculate_baseline_performance(car_laps)
            for _, lap in car_laps.iterrows():
                lap_weather = self._get_lap_weather_conditions(lap, weather_data)
                if lap_weather is None:
                    continue
                lap_telemetry = self._get_lap_telemetry(lap, telemetry_data, car_number)
                weather_impact = self._calculate_weather_impact(lap, baseline_time, lap_weather)
                feature_vector = self._create_weather_feature_vector(lap, lap_weather, lap_telemetry, session_key)
                features.append(pd.DataFrame([feature_vector]))
                impacts.append(weather_impact)
        
        return pd.concat(features, ignore_index=True) if features else pd.DataFrame(), np.array(impacts)
    
    def _prepare_weather_timestamps(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        if 'TIME_UTC_STR' in weather_data.columns:
            weather_data['timestamp'] = pd.to_datetime(weather_data['TIME_UTC_STR'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        elif 'TIME_UTC_SECONDS' in weather_data.columns:
            weather_data['timestamp'] = pd.to_datetime(weather_data['TIME_UTC_SECONDS'], unit='s', errors='coerce')
        else:
            start_time = datetime.now() - timedelta(hours=2)
            weather_data['timestamp'] = [start_time + timedelta(seconds=i*30) for i in range(len(weather_data))]
        return weather_data.sort_values('timestamp')
    
    def _prepare_lap_timestamps(self, lap_data: pd.DataFrame) -> pd.DataFrame:
        if 'HOUR' in lap_data.columns:
            lap_data['timestamp'] = pd.to_datetime(lap_data['HOUR'], format='%H:%M:%S.%f', errors='coerce')
        elif 'ELAPSED' in lap_data.columns:
            base_time = datetime.now().replace(hour=14, minute=0, second=0)
            lap_data['timestamp'] = lap_data['ELAPSED'].apply(lambda x: base_time + timedelta(seconds=x) if pd.notna(x) else base_time)
        else:
            base_time = datetime.now().replace(hour=14, minute=0, second=0)
            lap_data['timestamp'] = [base_time + timedelta(seconds=i*90) for i in range(len(lap_data))]
        return lap_data
    
    def _get_lap_weather_conditions(self, lap: pd.Series, weather_data: pd.DataFrame) -> pd.Series | None:
        lap_time = lap['timestamp']
        if weather_data.empty:
            return None
        time_diff = (weather_data['timestamp'] - lap_time).abs()
        closest_idx = time_diff.idxmin()
        if time_diff[closest_idx] > timedelta(minutes=5):
            return None
        return weather_data.loc[closest_idx]
    
    def _get_lap_telemetry(self, lap: pd.Series, telemetry_data: pd.DataFrame, car_number: int) -> dict:
        if telemetry_data.empty:
            return {}
        lap_telemetry = telemetry_data[(telemetry_data['vehicle_number'] == car_number) & 
                                       (telemetry_data['lap'] == lap['LAP_NUMBER'])]
        if lap_telemetry.empty:
            return {}
        return {
            'avg_throttle': lap_telemetry.get('aps', pd.Series([60.0])).mean(),
            'avg_brake': ((lap_telemetry.get('pbrake_f', pd.Series([50.0])) + lap_telemetry.get('pbrake_r', pd.Series([50.0])))/2).mean(),
            'avg_speed': lap_telemetry.get('KPH', lap_telemetry.get('speed', pd.Series([120.0]))).mean(),
            'steering_variance': lap_telemetry.get('Steering_Angle', pd.Series([10.0])).var()
        }
    
    def _calculate_baseline_performance(self, car_laps: pd.DataFrame) -> float:
        lap_times = car_laps['LAP_TIME_SECONDS'].dropna()
        if len(lap_times) < 3:
            return lap_times.mean() if not lap_times.empty else 100.0
        best_laps = lap_times.nsmallest(max(3, int(len(lap_times) * 0.3)))
        return best_laps.median()
    
    def _calculate_weather_impact(self, lap: pd.Series, baseline_time: float, weather: pd.Series) -> float:
        actual_time = lap['LAP_TIME_SECONDS']
        impact = actual_time - baseline_time
        normal_variation = 0.5
        adjusted_impact = impact if abs(impact) > normal_variation else 0
        return max(-5.0, min(5.0, adjusted_impact))
    
    def _create_weather_feature_vector(self, lap: pd.Series, weather: pd.Series, telemetry: dict, session_key: str) -> dict:
        features = {
            'air_temp': weather.get('AIR_TEMP', 25.0),
            'track_temp': weather.get('TRACK_TEMP', 30.0),
            'humidity': weather.get('HUMIDITY', 50.0),
            'pressure': weather.get('PRESSURE', 1013.0),
            'wind_speed': weather.get('WIND_SPEED', 0.0),
            'wind_direction': weather.get('WIND_DIRECTION', 0.0),
            'rain': weather.get('RAIN', 0.0),
        }
        features['temp_difference'] = features['track_temp'] - features['air_temp']
        features['air_density'] = self._calculate_air_density(features['air_temp'], features['pressure'], features['humidity'])
        features['wind_effect'] = self._calculate_wind_effect(features['wind_speed'], features['wind_direction'])
        track_name = session_key.split('_')[0] if '_' in session_key else session_key
        features['track_weather_sensitivity'] = self._get_track_weather_sensitivity(track_name)
        features['lap_number'] = lap['LAP_NUMBER']
        features['time_of_day'] = lap['timestamp'].hour + lap['timestamp'].minute / 60
        features.update({
            'throttle_usage': telemetry.get('avg_throttle', 60.0),
            'braking_intensity': telemetry.get('avg_brake', 50.0),
            'avg_speed': telemetry.get('avg_speed', 120.0),
            'steering_activity': telemetry.get('steering_variance', 10.0)
        })
        return features
    
    def _calculate_air_density(self, air_temp: float, pressure: float, humidity: float) -> float:
        R = 287.05
        temp_kelvin = air_temp + 273.15
        vapor_pressure = 0.611 * np.exp(17.27 * air_temp / (air_temp + 237.3)) * (humidity / 100)
        dry_air_pressure = pressure - vapor_pressure
        return (dry_air_pressure * 100) / (R * temp_kelvin)
    
    def _calculate_wind_effect(self, wind_speed: float, wind_direction: float) -> float:
        return wind_speed * 0.1
    
    def _get_track_weather_sensitivity(self, track_name: str) -> float:
        sensitivity_map = {
            'road-america': 0.9, 'sebring': 0.85, 'barber': 0.8,
            'sonoma': 0.75, 'vir': 0.7, 'cota': 0.65, 'indianapolis': 0.5
        }
        return sensitivity_map.get(track_name.lower(), 0.7)
    
    def predict_weather_impact(self, weather_conditions: dict, track_name: str, lap_context: dict) -> float:
        try:
            features = self._create_weather_feature_vector(lap_context.get('lap_info', {}), weather_conditions, lap_context.get('telemetry', {}), track_name)
            feature_vector = [features.get(col, 0) for col in self.feature_columns]
            feature_array = np.array(feature_vector).reshape(1, -1)
            scaled_features = self.scaler.transform(feature_array)
            impact = self.model.predict(scaled_features)[0]
            return max(-5.0, min(5.0, impact))
        except Exception as e:
            print(f"Weather impact prediction error: {e}")
            return self._fallback_weather_impact(weather_conditions, track_name)
    
    def _fallback_weather_impact(self, weather_conditions: dict, track_name: str) -> float:
        base_impact = 0.0
        temp_diff = abs(weather_conditions.get('air_temp', 25) - 25)
        base_impact += temp_diff * 0.05
        if weather_conditions.get('rain', 0) > 0:
            base_impact += 2.0
        sensitivity = self._get_track_weather_sensitivity(track_name)
        return base_impact * sensitivity
    
    def get_optimal_conditions(self, track_name: str) -> dict:
        return {'AIR_TEMP': 22.0, 'TRACK_TEMP': 30.0, 'HUMIDITY': 50.0, 'PRESSURE': 1013.0, 'WIND_SPEED': 2.0, 'RAIN': 0.0}
    
    def estimate_tire_temperature(self, weather_conditions: dict, track_name: str, lap_count: int) -> float:
        base_temp = weather_conditions.get('track_temp', 30.0)
        air_temp = weather_conditions.get('air_temp', 25.0)
        usage_heat = min(15.0, lap_count * 0.5)
        track_heat = self._get_track_weather_sensitivity(track_name) * 5.0
        estimated_temp = base_temp + usage_heat + track_heat
        if air_temp < base_temp:
            estimated_temp -= (base_temp - air_temp) * 0.1
        return max(air_temp, min(100.0, estimated_temp))
    
    def save_model(self, filepath: str):
        model_data = {'model': self.model, 'scaler': self.scaler, 'feature_columns': self.feature_columns}
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']



















# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import joblib
# from datetime import datetime, timedelta

# class WeatherModelTrainer:
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#         self.scaler = StandardScaler()
#         self.feature_columns = []
    
#     def train(self, processed_data: dict) -> dict:
#         """Train weather impact model using integrated weather, lap, and telemetry data"""
#         features_list = []
#         impact_list = []
        
#         for session_key, data in processed_data.items():
#             if 'weather_data' in data and 'lap_data' in data:
#                 session_features, session_impacts = self._extract_weather_features(data, session_key)
#                 if not session_features.empty and len(session_impacts) > 0:
#                     features_list.append(session_features)
#                     impact_list.append(session_impacts)
        
#         if not features_list:
#             return {'error': 'No valid weather features extracted'}
        
#         # Combine all session data safely
#         X = pd.concat(features_list, ignore_index=True)
#         y = np.concatenate(impact_list) if impact_list else np.array([])
        
#         # Remove NaN values
#         if len(y) > 0:
#             valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
#             X = X[valid_mask]
#             y = y[valid_mask]
        
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
        
#         return {
#             'model': self,
#             'features': self.feature_columns,
#             'train_score': self.model.score(X_train, y_train),
#             'test_score': self.model.score(X_test, y_test),
#             'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_)),
#             'training_samples': len(X)
#         }
    
#     def _extract_weather_features(self, data: dict, session_key: str) -> tuple:
#         """Extract weather impact features with proper handling of missing data"""
#         weather_data = data.get('weather_data', pd.DataFrame())
#         lap_data = data.get('lap_data', pd.DataFrame())
#         telemetry_data = data.get('telemetry_data', pd.DataFrame())
        
#         features = []
#         impacts = []
        
#         if lap_data.empty or weather_data.empty:
#             return pd.DataFrame(), np.array([])
        
#         weather_data = self._prepare_weather_timestamps(weather_data)
#         lap_data = self._prepare_lap_timestamps(lap_data)
        
#         for car_number in lap_data['NUMBER'].unique():
#             car_laps = lap_data[lap_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
#             if len(car_laps) < 5:
#                 continue
#             baseline_time = self._calculate_baseline_performance(car_laps)
#             for _, lap in car_laps.iterrows():
#                 lap_weather = self._get_lap_weather_conditions(lap, weather_data)
#                 if lap_weather is None:
#                     continue
#                 lap_telemetry = self._get_lap_telemetry(lap, telemetry_data, car_number)
#                 weather_impact = self._calculate_weather_impact(lap, baseline_time, lap_weather)
#                 feature_vector = self._create_weather_feature_vector(lap, lap_weather, lap_telemetry, session_key)
#                 features.append(pd.DataFrame([feature_vector]))
#                 impacts.append(weather_impact)
        
#         return pd.concat(features, ignore_index=True) if features else pd.DataFrame(), np.array(impacts)
    
#     def _prepare_weather_timestamps(self, weather_data: pd.DataFrame) -> pd.DataFrame:
#         if 'TIME_UTC_STR' in weather_data.columns:
#             weather_data['timestamp'] = pd.to_datetime(weather_data['TIME_UTC_STR'], errors='coerce')
#         elif 'TIME_UTC_SECONDS' in weather_data.columns:
#             weather_data['timestamp'] = pd.to_datetime(weather_data['TIME_UTC_SECONDS'], unit='s', errors='coerce')
#         else:
#             start_time = datetime.now() - timedelta(hours=2)
#             weather_data['timestamp'] = [start_time + timedelta(seconds=i*30) for i in range(len(weather_data))]
#         return weather_data.sort_values('timestamp')
    
#     def _prepare_lap_timestamps(self, lap_data: pd.DataFrame) -> pd.DataFrame:
#         if 'HOUR' in lap_data.columns:
#             lap_data['timestamp'] = pd.to_datetime(lap_data['HOUR'], errors='coerce')
#         elif 'ELAPSED' in lap_data.columns:
#             base_time = datetime.now().replace(hour=14, minute=0, second=0)
#             lap_data['timestamp'] = lap_data['ELAPSED'].apply(lambda x: base_time + timedelta(seconds=x) if pd.notna(x) else base_time)
#         else:
#             base_time = datetime.now().replace(hour=14, minute=0, second=0)
#             lap_data['timestamp'] = [base_time + timedelta(seconds=i*90) for i in range(len(lap_data))]
#         return lap_data
    
#     def _get_lap_weather_conditions(self, lap: pd.Series, weather_data: pd.DataFrame) -> pd.Series | None:
#         lap_time = lap['timestamp']
#         if weather_data.empty:
#             return None
#         time_diff = (weather_data['timestamp'] - lap_time).abs()
#         closest_idx = time_diff.idxmin()
#         if time_diff[closest_idx] > timedelta(minutes=5):
#             return None
#         return weather_data.loc[closest_idx]
    
#     def _get_lap_telemetry(self, lap: pd.Series, telemetry_data: pd.DataFrame, car_number: int) -> dict:
#         if telemetry_data.empty:
#             return {}
#         lap_telemetry = telemetry_data[(telemetry_data['vehicle_number'] == car_number) & 
#                                        (telemetry_data['lap'] == lap['LAP_NUMBER'])]
#         if lap_telemetry.empty:
#             return {}
#         return {
#             'avg_throttle': lap_telemetry.get('aps', pd.Series([60.0])).mean(),
#             'avg_brake': ((lap_telemetry.get('pbrake_f', pd.Series([50.0])) + lap_telemetry.get('pbrake_r', pd.Series([50.0])))/2).mean(),
#             'avg_speed': lap_telemetry.get('KPH', lap_telemetry.get('speed', pd.Series([120.0]))).mean(),
#             'steering_variance': lap_telemetry.get('Steering_Angle', pd.Series([10.0])).var()
#         }
    
#     def _calculate_baseline_performance(self, car_laps: pd.DataFrame) -> float:
#         lap_times = car_laps['LAP_TIME_SECONDS'].dropna()
#         if len(lap_times) < 3:
#             return lap_times.mean() if not lap_times.empty else 100.0
#         best_laps = lap_times.nsmallest(max(3, int(len(lap_times) * 0.3)))
#         return best_laps.median()
    
#     def _calculate_weather_impact(self, lap: pd.Series, baseline_time: float, weather: pd.Series) -> float:
#         actual_time = lap['LAP_TIME_SECONDS']
#         impact = actual_time - baseline_time
#         normal_variation = 0.5
#         adjusted_impact = impact if abs(impact) > normal_variation else 0
#         return max(-5.0, min(5.0, adjusted_impact))
    
#     def _create_weather_feature_vector(self, lap: pd.Series, weather: pd.Series, telemetry: dict, session_key: str) -> dict:
#         features = {
#             'air_temp': weather.get('AIR_TEMP', 25.0),
#             'track_temp': weather.get('TRACK_TEMP', 30.0),
#             'humidity': weather.get('HUMIDITY', 50.0),
#             'pressure': weather.get('PRESSURE', 1013.0),
#             'wind_speed': weather.get('WIND_SPEED', 0.0),
#             'wind_direction': weather.get('WIND_DIRECTION', 0.0),
#             'rain': weather.get('RAIN', 0.0),
#         }
#         features['temp_difference'] = features['track_temp'] - features['air_temp']
#         features['air_density'] = self._calculate_air_density(features['air_temp'], features['pressure'], features['humidity'])
#         features['wind_effect'] = self._calculate_wind_effect(features['wind_speed'], features['wind_direction'])
#         track_name = session_key.split('_')[0] if '_' in session_key else session_key
#         features['track_weather_sensitivity'] = self._get_track_weather_sensitivity(track_name)
#         features['lap_number'] = lap['LAP_NUMBER']
#         features['time_of_day'] = lap['timestamp'].hour + lap['timestamp'].minute / 60
#         features.update({
#             'throttle_usage': telemetry.get('avg_throttle', 60.0),
#             'braking_intensity': telemetry.get('avg_brake', 50.0),
#             'avg_speed': telemetry.get('avg_speed', 120.0),
#             'steering_activity': telemetry.get('steering_variance', 10.0)
#         })
#         return features
    
#     def _calculate_air_density(self, air_temp: float, pressure: float, humidity: float) -> float:
#         R = 287.05
#         temp_kelvin = air_temp + 273.15
#         vapor_pressure = 0.611 * np.exp(17.27 * air_temp / (air_temp + 237.3)) * (humidity / 100)
#         dry_air_pressure = pressure - vapor_pressure
#         return (dry_air_pressure * 100) / (R * temp_kelvin)
    
#     def _calculate_wind_effect(self, wind_speed: float, wind_direction: float) -> float:
#         return wind_speed * 0.1
    
#     def _get_track_weather_sensitivity(self, track_name: str) -> float:
#         sensitivity_map = {
#             'road-america': 0.9, 'sebring': 0.85, 'barber': 0.8,
#             'sonoma': 0.75, 'vir': 0.7, 'cota': 0.65, 'indianapolis': 0.5
#         }
#         return sensitivity_map.get(track_name.lower(), 0.7)
    
#     def predict_weather_impact(self, weather_conditions: dict, track_name: str, lap_context: dict) -> float:
#         try:
#             features = self._create_weather_feature_vector(lap_context.get('lap_info', {}), weather_conditions, lap_context.get('telemetry', {}), track_name)
#             feature_vector = [features.get(col, 0) for col in self.feature_columns]
#             feature_array = np.array(feature_vector).reshape(1, -1)
#             scaled_features = self.scaler.transform(feature_array)
#             impact = self.model.predict(scaled_features)[0]
#             return max(-5.0, min(5.0, impact))
#         except Exception as e:
#             print(f"Weather impact prediction error: {e}")
#             return self._fallback_weather_impact(weather_conditions, track_name)
    
#     def _fallback_weather_impact(self, weather_conditions: dict, track_name: str) -> float:
#         base_impact = 0.0
#         temp_diff = abs(weather_conditions.get('air_temp', 25) - 25)
#         base_impact += temp_diff * 0.05
#         if weather_conditions.get('rain', 0) > 0:
#             base_impact += 2.0
#         sensitivity = self._get_track_weather_sensitivity(track_name)
#         return base_impact * sensitivity
    
#     def get_optimal_conditions(self, track_name: str) -> dict:
#         return {'AIR_TEMP': 22.0, 'TRACK_TEMP': 30.0, 'HUMIDITY': 50.0, 'PRESSURE': 1013.0, 'WIND_SPEED': 2.0, 'RAIN': 0.0}
    
#     def estimate_tire_temperature(self, weather_conditions: dict, track_name: str, lap_count: int) -> float:
#         base_temp = weather_conditions.get('track_temp', 30.0)
#         air_temp = weather_conditions.get('air_temp', 25.0)
#         usage_heat = min(15.0, lap_count * 0.5)
#         track_heat = self._get_track_weather_sensitivity(track_name) * 5.0
#         estimated_temp = base_temp + usage_heat + track_heat
#         if air_temp < base_temp:
#             estimated_temp -= (base_temp - air_temp) * 0.1
#         return max(air_temp, min(100.0, estimated_temp))
    
#     def save_model(self, filepath: str):
#         model_data = {'model': self.model, 'scaler': self.scaler, 'feature_columns': self.feature_columns}
#         joblib.dump(model_data, filepath)
    
#     def load_model(self, filepath: str):
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
# from datetime import datetime, timedelta

# class WeatherModelTrainer:
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#         self.scaler = StandardScaler()
#         self.feature_columns = []
    
#     def train(self, processed_data: dict) -> dict:
#         """Train weather impact model using integrated weather, lap, and telemetry data"""
#         features_list = []
#         impact_list = []
        
#         for session_key, data in processed_data.items():
#             if not data['weather_data'].empty and not data['lap_data'].empty:
#                 session_features, session_impacts = self._extract_weather_features(data, session_key)
#                 if not session_features.empty:
#                     features_list.append(session_features)
#                     impact_list.append(session_impacts)
        
#         if not features_list:
#             return {'error': 'No valid weather features extracted'}
        
#         # Combine all session data
#         X = pd.concat(features_list, ignore_index=True)
#         y = np.concatenate(impact_list)
        
#         # Remove NaN values
#         valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
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
        
#         return {
#             'model': self,
#             'features': self.feature_columns,
#             'train_score': train_score,
#             'test_score': test_score,
#             'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_)),
#             'training_samples': len(X)
#         }
    
#     def _extract_weather_features(self, data: dict, session_key: str) -> tuple:
#         """Extract comprehensive weather impact features with proper time synchronization"""
#         weather_data = data['weather_data']
#         lap_data = data['lap_data']
#         telemetry_data = data.get('telemetry_data', pd.DataFrame())
        
#         features = []
#         impacts = []
        
#         # Convert timestamps to datetime objects
#         weather_data = self._prepare_weather_timestamps(weather_data)
#         lap_data = self._prepare_lap_timestamps(lap_data)
        
#         # Group by car to analyze individual performance
#         for car_number in lap_data['NUMBER'].unique():
#             car_laps = lap_data[lap_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
            
#             if len(car_laps) < 5:  # Need sufficient laps for baseline
#                 continue
            
#             # Calculate baseline performance for this car
#             baseline_time = self._calculate_baseline_performance(car_laps)
            
#             for _, lap in car_laps.iterrows():
#                 # Get precise weather conditions for this lap
#                 lap_weather = self._get_lap_weather_conditions(lap, weather_data)
#                 if lap_weather is None:
#                     continue
                
#                 # Get telemetry data for driving style context
#                 lap_telemetry = self._get_lap_telemetry(lap, telemetry_data, car_number)
                
#                 # Calculate actual weather impact
#                 weather_impact = self._calculate_weather_impact(lap, baseline_time, lap_weather)
                
#                 # Extract comprehensive features
#                 feature_vector = self._create_weather_feature_vector(lap, lap_weather, lap_telemetry, session_key)
                
#                 features.append(pd.DataFrame([feature_vector]))
#                 impacts.append(weather_impact)
        
#         if features:
#             return pd.concat(features, ignore_index=True), np.array(impacts)
#         return pd.DataFrame(), np.array([])
    
#     def _prepare_weather_timestamps(self, weather_data: pd.DataFrame) -> pd.DataFrame:
#         """Prepare weather data timestamps for precise matching"""
#         if 'TIME_UTC_STR' in weather_data.columns:
#             weather_data['timestamp'] = pd.to_datetime(weather_data['TIME_UTC_STR'])
#         elif 'TIME_UTC_SECONDS' in weather_data.columns:
#             weather_data['timestamp'] = pd.to_datetime(weather_data['TIME_UTC_SECONDS'], unit='s')
#         else:
#             # Create synthetic timestamps if not available
#             start_time = datetime.now() - timedelta(hours=2)
#             weather_data['timestamp'] = [start_time + timedelta(seconds=i*30) for i in range(len(weather_data))]
        
#         return weather_data.sort_values('timestamp')
    
#     def _prepare_lap_timestamps(self, lap_data: pd.DataFrame) -> pd.DataFrame:
#         """Prepare lap data timestamps for precise matching"""
#         if 'HOUR' in lap_data.columns:
#             lap_data['timestamp'] = pd.to_datetime(lap_data['HOUR'])
#         elif 'ELAPSED' in lap_data.columns:
#             # Convert elapsed time to timestamp (approximate)
#             base_time = datetime.now().replace(hour=14, minute=0, second=0)
#             lap_data['timestamp'] = lap_data['ELAPSED'].apply(
#                 lambda x: base_time + timedelta(seconds=x) if pd.notna(x) else base_time
#             )
#         else:
#             # Create synthetic timestamps
#             base_time = datetime.now().replace(hour=14, minute=0, second=0)
#             lap_data['timestamp'] = [base_time + timedelta(seconds=i*90) for i in range(len(lap_data))]
        
#         return lap_data
    
#     def _get_lap_weather_conditions(self, lap: pd.Series, weather_data: pd.DataFrame) -> pd.Series:
#         """Get precise weather conditions for a specific lap"""
#         lap_time = lap['timestamp']
        
#         # Find closest weather reading within reasonable time window
#         time_diff = (weather_data['timestamp'] - lap_time).abs()
#         closest_idx = time_diff.idxmin()
        
#         # Only use weather data within 5 minutes of lap time
#         if time_diff[closest_idx] > timedelta(minutes=5):
#             return None
        
#         return weather_data.loc[closest_idx]
    
#     def _get_lap_telemetry(self, lap: pd.Series, telemetry_data: pd.DataFrame, car_number: int) -> dict:
#         """Get telemetry data for driving style context"""
#         if telemetry_data.empty:
#             return {}
        
#         lap_telemetry = telemetry_data[
#             (telemetry_data['vehicle_number'] == car_number) & 
#             (telemetry_data['lap'] == lap['LAP_NUMBER'])
#         ]
        
#         if lap_telemetry.empty:
#             return {}
        
#         return {
#             'avg_throttle': lap_telemetry['aps'].mean(),
#             'avg_brake': (lap_telemetry['pbrake_f'] + lap_telemetry['pbrake_r']).mean() / 2,
#             'avg_speed': lap_telemetry.get('KPH', lap_telemetry.get('speed', 0)).mean(),
#             'steering_variance': lap_telemetry['Steering_Angle'].var()
#         }
    
#     def _calculate_baseline_performance(self, car_laps: pd.DataFrame) -> float:
#         """Calculate baseline performance for a car (optimal conditions)"""
#         # Use best laps as baseline, excluding outliers
#         lap_times = car_laps['LAP_TIME_SECONDS'].dropna()
#         if len(lap_times) < 3:
#             return lap_times.mean() if not lap_times.empty else 100.0
        
#         # Use median of best 30% laps as baseline
#         best_laps = lap_times.nsmallest(max(3, int(len(lap_times) * 0.3)))
#         return best_laps.median()
    
#     def _calculate_weather_impact(self, lap: pd.Series, baseline_time: float, weather: pd.Series) -> float:
#         """Calculate actual weather impact on lap time"""
#         actual_time = lap['LAP_TIME_SECONDS']
        
#         # Simple impact calculation (can be enhanced with more sophisticated models)
#         impact = actual_time - baseline_time
        
#         # Adjust for normal performance variation (non-weather related)
#         normal_variation = 0.5  # seconds of normal lap-to-lap variation
#         adjusted_impact = impact if abs(impact) > normal_variation else 0
        
#         return max(-5.0, min(5.0, adjusted_impact))  # Bound impact to reasonable range
    
#     def _create_weather_feature_vector(self, lap: pd.Series, weather: pd.Series, 
#                                      telemetry: dict, session_key: str) -> dict:
#         """Create comprehensive weather feature vector"""
#         # Basic weather conditions
#         features = {
#             'air_temp': weather.get('AIR_TEMP', 25.0),
#             'track_temp': weather.get('TRACK_TEMP', 30.0),
#             'humidity': weather.get('HUMIDITY', 50.0),
#             'pressure': weather.get('PRESSURE', 1013.0),
#             'wind_speed': weather.get('WIND_SPEED', 0.0),
#             'wind_direction': weather.get('WIND_DIRECTION', 0.0),
#             'rain': weather.get('RAIN', 0.0),
#         }
        
#         # Derived weather features
#         features['temp_difference'] = features['track_temp'] - features['air_temp']
#         features['air_density'] = self._calculate_air_density(features['air_temp'], 
#                                                             features['pressure'], 
#                                                             features['humidity'])
#         features['wind_effect'] = self._calculate_wind_effect(features['wind_speed'], 
#                                                             features['wind_direction'])
        
#         # Track and session context
#         track_name = session_key.split('_')[0] if '_' in session_key else session_key
#         features['track_weather_sensitivity'] = self._get_track_weather_sensitivity(track_name)
#         features['lap_number'] = lap['LAP_NUMBER']
#         features['time_of_day'] = lap['timestamp'].hour + lap['timestamp'].minute / 60
        
#         # Driving style context from telemetry
#         features.update({
#             'throttle_usage': telemetry.get('avg_throttle', 60.0),
#             'braking_intensity': telemetry.get('avg_brake', 50.0),
#             'avg_speed': telemetry.get('avg_speed', 120.0),
#             'steering_activity': telemetry.get('steering_variance', 10.0)
#         })
        
#         return features
    
#     def _calculate_air_density(self, air_temp: float, pressure: float, humidity: float) -> float:
#         """Calculate air density (affects engine performance and aerodynamics)"""
#         # Simplified air density calculation
#         R = 287.05  # Specific gas constant for dry air (J/kg·K)
#         temp_kelvin = air_temp + 273.15
        
#         # Adjust for humidity (simplified)
#         vapor_pressure = 0.611 * np.exp(17.27 * air_temp / (air_temp + 237.3)) * (humidity / 100)
#         dry_air_pressure = pressure - vapor_pressure
        
#         air_density = (dry_air_pressure * 100) / (R * temp_kelvin)  # Convert pressure to Pa
#         return air_density
    
#     def _calculate_wind_effect(self, wind_speed: float, wind_direction: float) -> float:
#         """Calculate wind effect on lap performance"""
#         # Simplified wind effect model
#         # Assuming headwind/tailwind component affects straightline speed
#         # This is a placeholder - real implementation would need track layout data
#         wind_effect = wind_speed * 0.1  # 0.1 seconds per m/s wind speed (approximate)
#         return wind_effect
    
#     def _get_track_weather_sensitivity(self, track_name: str) -> float:
#         """Get track-specific weather sensitivity based on actual characteristics"""
#         sensitivity_map = {
#             'road-america': 0.9,    # Long straights, elevation changes
#             'sebring': 0.85,        # Bumpy surface, temperature sensitive
#             'barber': 0.8,          # Technical, grip dependent
#             'sonoma': 0.75,         # Elevation changes, varied corners
#             'vir': 0.7,             # Balanced circuit
#             'cota': 0.65,           # Modern, smooth surface
#             'indianapolis': 0.5,    # Oval, less weather dependent
#         }
#         return sensitivity_map.get(track_name.lower(), 0.7)
    
#     def predict_weather_impact(self, weather_conditions: dict, track_name: str, 
#                              lap_context: dict) -> float:
#         """Predict weather impact on lap time for given conditions"""
#         try:
#             # Create comprehensive feature vector
#             features = self._create_weather_feature_vector(
#                 lap_context.get('lap_info', {}),
#                 weather_conditions,
#                 lap_context.get('telemetry', {}),
#                 track_name
#             )
            
#             # Ensure all expected features are present
#             feature_vector = [features.get(col, 0) for col in self.feature_columns]
#             feature_array = np.array(feature_vector).reshape(1, -1)
            
#             # Scale and predict
#             scaled_features = self.scaler.transform(feature_array)
#             impact = self.model.predict(scaled_features)[0]
            
#             return max(-5.0, min(5.0, impact))  # Bound prediction
#         except Exception as e:
#             print(f"Weather impact prediction error: {e}")
#             return self._fallback_weather_impact(weather_conditions, track_name)
    
#     def _fallback_weather_impact(self, weather_conditions: dict, track_name: str) -> float:
#         """Fallback weather impact estimation"""
#         # Simple rule-based fallback
#         base_impact = 0.0
        
#         # Temperature effect (optimal around 25°C)
#         temp_diff = abs(weather_conditions.get('air_temp', 25) - 25)
#         base_impact += temp_diff * 0.05
        
#         # Rain effect
#         if weather_conditions.get('rain', 0) > 0:
#             base_impact += 2.0
        
#         # Track sensitivity multiplier
#         sensitivity = self._get_track_weather_sensitivity(track_name)
        
#         return base_impact * sensitivity
    
#     def get_optimal_conditions(self, track_name: str) -> dict:
#         """Get optimal weather conditions for a track based on historical patterns"""
#         # These values represent typical optimal conditions for racing
#         return {
#             'AIR_TEMP': 22.0,      # Cool enough for engine performance, warm for tires
#             'TRACK_TEMP': 30.0,    # Optimal tire operating temperature
#             'HUMIDITY': 50.0,      # Moderate humidity
#             'PRESSURE': 1013.0,    # Standard atmospheric pressure
#             'WIND_SPEED': 2.0,     # Light wind
#             'RAIN': 0.0            # Dry conditions
#         }
    
#     def estimate_tire_temperature(self, weather_conditions: dict, track_name: str, 
#                                 lap_count: int) -> float:
#         """Estimate tire temperature based on weather and usage"""
#         base_temp = weather_conditions.get('track_temp', 30.0)
#         air_temp = weather_conditions.get('air_temp', 25.0)
        
#         # Tire heating from usage (simplified model)
#         usage_heat = min(15.0, lap_count * 0.5)  # Caps at 15°C above base
        
#         # Track abrasiveness effect
#         track_heat = self._get_track_weather_sensitivity(track_name) * 5.0
        
#         estimated_temp = base_temp + usage_heat + track_heat
        
#         # Cooling effect from air temperature difference
#         if air_temp < base_temp:
#             cooling = (base_temp - air_temp) * 0.1
#             estimated_temp -= cooling
        
#         return max(air_temp, min(100.0, estimated_temp))  # Reasonable bounds
    
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

# class WeatherModelTrainer:
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    
#     def train(self, processed_data: dict) -> dict:
#         """Train weather impact model using weather and lap data"""
#         features_list = []
#         impact_list = []
        
#         for track_name, data in processed_data.items():
#             if not data['weather_data'].empty and not data['lap_data'].empty:
#                 track_features, track_impacts = self._extract_weather_features(data, track_name)
#                 if track_features is not None:
#                     features_list.append(track_features)
#                     impact_list.append(track_impacts)
        
#         if not features_list:
#             return {'model': self, 'features': [], 'train_score': 0, 'test_score': 0}
        
#         # Combine all track data
#         X = pd.concat(features_list, ignore_index=True)
#         y = np.concatenate(impact_list)
        
#         # Remove NaN values
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
    
#     def _extract_weather_features(self, data: dict, track_name: str) -> tuple:
#         """Extract weather impact features by merging weather and lap data"""
#         weather_data = data['weather_data']
#         lap_data = data['lap_data']
        
#         features = []
#         impacts = []
        
#         # Convert weather timestamps to match lap data timing
#         weather_data['timestamp'] = pd.to_datetime(weather_data['TIME_UTC_STR'])
        
#         for _, lap in lap_data.iterrows():
#             # Find closest weather reading to this lap
#             lap_time = pd.to_datetime(lap.get('HOUR', weather_data['timestamp'].iloc[0]))
#             time_diff = (weather_data['timestamp'] - lap_time).abs()
#             closest_weather = weather_data.iloc[time_diff.argmin()]
            
#             # Calculate weather impact (lap time vs optimal conditions)
#             optimal_time = lap_data['LAP_TIME_SECONDS'].min()
#             weather_impact = lap['LAP_TIME_SECONDS'] - optimal_time
            
#             # Weather features
#             feature_vector = pd.DataFrame([{
#                 'air_temp': closest_weather.get('AIR_TEMP', 25),
#                 'track_temp': closest_weather.get('TRACK_TEMP', 30),
#                 'humidity': closest_weather.get('HUMIDITY', 50),
#                 'pressure': closest_weather.get('PRESSURE', 1013),
#                 'wind_speed': closest_weather.get('WIND_SPEED', 0),
#                 'wind_direction': closest_weather.get('WIND_DIRECTION', 0),
#                 'rain': closest_weather.get('RAIN', 0),
#                 'track_wear_factor': self._get_track_weather_factor(track_name),
#                 'lap_number': lap['LAP_NUMBER'],
#                 'time_of_day': lap_time.hour + lap_time.minute/60
#             }])
            
#             features.append(feature_vector)
#             impacts.append(weather_impact)
        
#         if features:
#             return pd.concat(features, ignore_index=True), np.array(impacts)
#         return None, []
    
#     def _get_track_weather_factor(self, track_name: str) -> float:
#         """Get track-specific weather sensitivity factor"""
#         weather_factors = {
#             'barber-motorsports-park': 0.8,    # Technical, weather sensitive
#             'circuit-of-the-americas': 0.7,    # Modern, less sensitive
#             'indianapolis': 0.6,               # Oval, less weather dependent
#             'road-america': 0.9,               # Long, weather sensitive
#             'sebring': 0.8,                    # Bumpy, weather affects grip
#             'sonoma': 0.7,                     # Hilly, moderate sensitivity
#             'virginia-international-raceway': 0.75
#         }
#         return weather_factors.get(track_name, 0.7)
    
#     def predict_weather_impact(self, weather_conditions: dict, track_name: str, lap_number: int) -> float:
#         """Predict lap time impact from weather conditions"""
#         features = {
#             'air_temp': weather_conditions.get('air_temp', 25),
#             'track_temp': weather_conditions.get('track_temp', 30),
#             'humidity': weather_conditions.get('humidity', 50),
#             'pressure': weather_conditions.get('pressure', 1013),
#             'wind_speed': weather_conditions.get('wind_speed', 0),
#             'wind_direction': weather_conditions.get('wind_direction', 0),
#             'rain': weather_conditions.get('rain', 0),
#             'track_wear_factor': self._get_track_weather_factor(track_name),
#             'lap_number': lap_number,
#             'time_of_day': 14.0  # Default afternoon
#         }
        
#         feature_df = pd.DataFrame([features])
#         return self.model.predict(feature_df)[0]
    
#     def get_optimal_conditions(self, track_name: str) -> dict:
#         """Get optimal weather conditions for a track"""
#         # Based on historical data patterns
#         return {
#             'air_temp': 25.0,
#             'track_temp': 30.0,
#             'humidity': 50.0,
#             'pressure': 1013.0,
#             'wind_speed': 2.0,
#             'rain': 0.0
#         }
    
#     def save_model(self, filepath: str):
#         """Save trained model"""
#         joblib.dump(self.model, filepath)