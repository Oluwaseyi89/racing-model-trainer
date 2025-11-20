import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class WeatherModelTrainer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def train(self, track_data: Dict[str, Dict[str, pd.DataFrame]]) -> dict:
        """Train weather impact model using structured data from Firebase loader across all tracks"""
        try:
            features_list = []
            impact_list = []
            
            # Process each track's data
            for track_name, data_dict in track_data.items():
                session_features, session_impacts = self._extract_weather_features(data_dict, track_name)
                if not session_features.empty and len(session_impacts) > 0:
                    features_list.append(session_features)
                    impact_list.append(session_impacts)
            
            if not features_list:
                return {'error': 'No valid weather features extracted from any track', 'status': 'error'}
            
            # Combine all track data
            X = pd.concat(features_list, ignore_index=True)
            y = np.concatenate(impact_list) if impact_list else np.array([])
            
            # Remove NaN values (should be minimal due to schema enforcement)
            if len(y) > 0:
                valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
                X = X[valid_mask]
                y = y[valid_mask]
            
            if len(X) < 20:
                return {'error': f'Insufficient training samples: {len(X)}', 'status': 'error'}
            
            # Scale features and train
            X_scaled = self.scaler.fit_transform(X)
            self.feature_columns = X.columns.tolist()
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            self.model.fit(X_train, y_train)
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            return {
                'model': self,
                'features': self.feature_columns,
                'train_score': train_score,
                'test_score': test_score,
                'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_)),
                'training_samples': len(X),
                'tracks_used': len(track_data),
                'status': 'success'
            }
            
        except Exception as e:
            return {'error': f'Training failed: {str(e)}', 'status': 'error'}
    
    def _extract_weather_features(self, data_dict: Dict[str, pd.DataFrame], track_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Extract weather impact features using EXACT column names from FirebaseDataLoader"""
        weather_data = data_dict.get('weather_data', pd.DataFrame())
        pit_data = data_dict.get('pit_data', pd.DataFrame())
        telemetry_data = data_dict.get('telemetry_data', pd.DataFrame())
        
        features = []
        impacts = []
        
        if pit_data.empty or weather_data.empty:
            return pd.DataFrame(), np.array([])
        
        # Prepare timestamps using EXACT column names
        weather_data = self._prepare_weather_timestamps(weather_data)
        pit_data = self._prepare_pit_timestamps(pit_data)
        
        # Process each car's laps
        for car_number in pit_data['NUMBER'].unique():
            car_laps = pit_data[pit_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
            if len(car_laps) < 5:
                continue
                
            baseline_time = self._calculate_baseline_performance(car_laps)
            
            for _, lap in car_laps.iterrows():
                lap_weather = self._get_lap_weather_conditions(lap, weather_data)
                if lap_weather is None:
                    continue
                    
                lap_telemetry = self._get_lap_telemetry(lap, telemetry_data, car_number)
                weather_impact = self._calculate_weather_impact(lap, baseline_time, lap_weather)
                feature_vector = self._create_weather_feature_vector(lap, lap_weather, lap_telemetry, track_name)
                
                features.append(pd.DataFrame([feature_vector]))
                impacts.append(weather_impact)
        
        if features:
            return pd.concat(features, ignore_index=True), np.array(impacts)
        return pd.DataFrame(), np.array([])
    
    def _prepare_weather_timestamps(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare weather timestamps using EXACT column names"""
        weather_data = weather_data.copy()
        
        if 'TIME_UTC_SECONDS' in weather_data.columns:
            weather_data['timestamp'] = pd.to_datetime(weather_data['TIME_UTC_SECONDS'], unit='s', errors='coerce')
        elif 'TIME_UTC_STR' in weather_data.columns:
            weather_data['timestamp'] = pd.to_datetime(weather_data['TIME_UTC_STR'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        else:
            # Fallback: create synthetic timestamps
            start_time = datetime.now() - timedelta(hours=2)
            weather_data['timestamp'] = [start_time + timedelta(seconds=i*30) for i in range(len(weather_data))]
        
        return weather_data.sort_values('timestamp').dropna(subset=['timestamp'])
    
    def _prepare_pit_timestamps(self, pit_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare pit data timestamps using EXACT column names"""
        pit_data = pit_data.copy()
        
        if 'HOUR' in pit_data.columns:
            # Convert HOUR column (e.g., "13:42:35.322") to datetime
            base_date = datetime.now().date()
            pit_data['timestamp'] = pd.to_datetime(
                base_date.strftime('%Y-%m-%d') + ' ' + pit_data['HOUR'].astype(str),
                format='%Y-%m-%d %H:%M:%S.%f',
                errors='coerce'
            )
        elif 'ELAPSED' in pit_data.columns:
            # Use ELAPSED time to create relative timestamps
            base_time = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)
            pit_data['timestamp'] = pit_data['ELAPSED'].apply(
                lambda x: base_time + timedelta(seconds=x) if pd.notna(x) else base_time
            )
        else:
            # Fallback: create synthetic timestamps based on lap number
            base_time = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)
            pit_data['timestamp'] = [base_time + timedelta(seconds=i*90) for i in range(len(pit_data))]
        
        return pit_data.sort_values('timestamp').dropna(subset=['timestamp'])
    
    def _get_lap_weather_conditions(self, lap: pd.Series, weather_data: pd.DataFrame) -> pd.Series | None:
        """Get weather conditions for a specific lap using EXACT column names"""
        lap_time = lap['timestamp']
        if weather_data.empty:
            return None
            
        # Find closest weather reading to lap time
        time_diff = (weather_data['timestamp'] - lap_time).abs()
        closest_idx = time_diff.idxmin()
        
        # Only use if within 5 minutes
        if time_diff[closest_idx] > timedelta(minutes=5):
            return None
            
        return weather_data.loc[closest_idx]
    
    def _get_lap_telemetry(self, lap: pd.Series, telemetry_data: pd.DataFrame, car_number: int) -> Dict[str, float]:
        """Get telemetry data for a specific lap using EXACT column names"""
        if telemetry_data.empty:
            return {}
            
        # Extract vehicle number from vehicle_id if needed
        lap_telemetry = telemetry_data[
            (telemetry_data['vehicle_id'].str.contains(str(car_number))) &
            (telemetry_data['lap'] == lap['LAP_NUMBER'])
        ]
        
        if lap_telemetry.empty:
            return {}
            
        return {
            'avg_speed': lap_telemetry['speed'].mean() if 'speed' in lap_telemetry.columns else 120.0,
            'avg_long_accel': lap_telemetry['accx_can'].abs().mean() if 'accx_can' in lap_telemetry.columns else 0.3,
            'avg_lat_accel': lap_telemetry['accy_can'].abs().mean() if 'accy_can' in lap_telemetry.columns else 0.4,
            'avg_gear': lap_telemetry['gear'].mean() if 'gear' in lap_telemetry.columns else 3.0
        }
    
    def _calculate_baseline_performance(self, car_laps: pd.DataFrame) -> float:
        """Calculate baseline performance for weather impact comparison"""
        # Convert lap times to seconds
        lap_times_seconds = car_laps['LAP_TIME'].apply(self._lap_time_to_seconds).dropna()
        
        if len(lap_times_seconds) < 3:
            return lap_times_seconds.mean() if not lap_times_seconds.empty else 100.0
            
        # Use best 30% of laps as baseline
        best_laps = lap_times_seconds.nsmallest(max(3, int(len(lap_times_seconds) * 0.3)))
        return best_laps.median()
    
    def _calculate_weather_impact(self, lap: pd.Series, baseline_time: float, weather: pd.Series) -> float:
        """Calculate weather impact on lap performance"""
        actual_time = self._lap_time_to_seconds(lap['LAP_TIME'])
        impact = actual_time - baseline_time
        
        # Only consider impacts beyond normal variation
        normal_variation = 0.5
        adjusted_impact = impact if abs(impact) > normal_variation else 0
        
        return max(-5.0, min(5.0, adjusted_impact))
    
    def _create_weather_feature_vector(self, lap: pd.Series, weather: pd.Series, telemetry: Dict[str, float], track_name: str) -> Dict[str, float]:
        """Create weather feature vector using EXACT column names"""
        features = {
            # Weather conditions using EXACT column names
            'air_temp': weather.get('AIR_TEMP', 25.0),
            'track_temp': weather.get('TRACK_TEMP', 30.0),
            'humidity': weather.get('HUMIDITY', 50.0),
            'pressure': weather.get('PRESSURE', 1013.0),
            'wind_speed': weather.get('WIND_SPEED', 0.0),
            'wind_direction': weather.get('WIND_DIRECTION', 0.0),
            'rain': weather.get('RAIN', 0.0),
            
            # Derived weather features
            'temp_difference': weather.get('TRACK_TEMP', 30.0) - weather.get('AIR_TEMP', 25.0),
            'air_density': self._calculate_air_density(
                weather.get('AIR_TEMP', 25.0), 
                weather.get('PRESSURE', 1013.0), 
                weather.get('HUMIDITY', 50.0)
            ),
            'wind_effect': self._calculate_wind_effect(
                weather.get('WIND_SPEED', 0.0), 
                weather.get('WIND_DIRECTION', 0.0)
            ),
            
            # Track and context features
            'track_weather_sensitivity': self._get_track_weather_sensitivity(track_name),
            'lap_number': lap['LAP_NUMBER'],
            'time_of_day': lap['timestamp'].hour + lap['timestamp'].minute / 60.0,
            
            # Telemetry-based features
            'avg_speed': telemetry.get('avg_speed', 120.0),
            'driving_aggressiveness': (telemetry.get('avg_long_accel', 0.3) + telemetry.get('avg_lat_accel', 0.4)) / 2,
            'gear_usage': telemetry.get('avg_gear', 3.0)
        }
        
        return features
    
    def _lap_time_to_seconds(self, lap_time: str) -> float:
        """Convert lap time string to seconds (consistent with FirebaseDataLoader)"""
        try:
            if pd.isna(lap_time) or lap_time == 0:
                return 60.0
                
            time_str = str(lap_time).strip()
            parts = time_str.split(':')
            
            if len(parts) == 3:  # HH:MM:SS.ms
                hours, minutes, seconds = parts
                return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
            elif len(parts) == 2:  # MM:SS.ms
                minutes, seconds = parts
                return float(minutes) * 60 + float(seconds)
            else:
                return float(time_str)
        except:
            return 60.0
    
    def _calculate_air_density(self, air_temp: float, pressure: float, humidity: float) -> float:
        """Calculate air density for aerodynamic effects"""
        R = 287.05  # Specific gas constant for dry air (J/kg·K)
        temp_kelvin = air_temp + 273.15
        
        # Calculate vapor pressure from humidity
        vapor_pressure = 0.611 * np.exp(17.27 * air_temp / (air_temp + 237.3)) * (humidity / 100)
        dry_air_pressure = pressure - vapor_pressure
        
        return (dry_air_pressure * 100) / (R * temp_kelvin)  # Convert hPa to Pa
    
    def _calculate_wind_effect(self, wind_speed: float, wind_direction: float) -> float:
        """Calculate wind effect on straight-line speed"""
        # Simplified wind effect model
        return wind_speed * 0.1  # 10% impact per m/s
    
    def _get_track_weather_sensitivity(self, track_name: str) -> float:
        """Get track-specific weather sensitivity"""
        sensitivity_map = {
            'barber': 0.8, 'cota': 0.65, 'indianapolis': 0.5,
            'road_america': 0.9, 'sebring': 0.85, 'sonoma': 0.75, 'virginia': 0.7
        }
        return sensitivity_map.get(track_name.lower(), 0.7)
    
    def predict_weather_impact(self, weather_conditions: Dict[str, float], track_name: str, lap_context: Dict[str, float]) -> float:
        """Predict weather impact on lap performance"""
        try:
            if not self.feature_columns:
                return self._fallback_weather_impact(weather_conditions, track_name)
                
            # Create feature vector
            feature_vector = self._create_weather_feature_vector(
                lap_context.get('lap_info', {}),
                weather_conditions,
                lap_context.get('telemetry', {}),
                track_name
            )
            
            # Ensure all features are present
            feature_array = np.array([feature_vector.get(col, 0.0) for col in self.feature_columns]).reshape(1, -1)
            scaled_features = self.scaler.transform(feature_array)
            
            impact = self.model.predict(scaled_features)[0]
            return max(-5.0, min(5.0, impact))
            
        except Exception as e:
            print(f"⚠️ Weather impact prediction error: {e}")
            return self._fallback_weather_impact(weather_conditions, track_name)
    
    def _fallback_weather_impact(self, weather_conditions: Dict[str, float], track_name: str) -> float:
        """Fallback weather impact calculation"""
        base_impact = 0.0
        
        # Temperature effect
        temp_diff = abs(weather_conditions.get('AIR_TEMP', 25) - 25)
        base_impact += temp_diff * 0.05
        
        # Rain effect
        if weather_conditions.get('RAIN', 0) > 0:
            base_impact += 2.0
            
        # Track sensitivity
        sensitivity = self._get_track_weather_sensitivity(track_name)
        
        return base_impact * sensitivity
    
    def get_optimal_conditions(self, track_name: str) -> Dict[str, float]:
        """Get optimal weather conditions for a track"""
        return {
            'AIR_TEMP': 22.0,
            'TRACK_TEMP': 30.0, 
            'HUMIDITY': 50.0,
            'PRESSURE': 1013.0,
            'WIND_SPEED': 2.0,
            'RAIN': 0.0
        }
    
    def estimate_tire_temperature(self, weather_conditions: Dict[str, float], track_name: str, lap_count: int) -> float:
        """Estimate tire temperature based on weather and usage"""
        base_temp = weather_conditions.get('TRACK_TEMP', 30.0)
        air_temp = weather_conditions.get('AIR_TEMP', 25.0)
        
        # Heat from usage
        usage_heat = min(15.0, lap_count * 0.5)
        
        # Track-specific heating
        track_heat = self._get_track_weather_sensitivity(track_name) * 5.0
        
        estimated_temp = base_temp + usage_heat + track_heat
        
        # Cooling effect if air is cooler than track
        if air_temp < base_temp:
            estimated_temp -= (base_temp - air_temp) * 0.1
            
        return max(air_temp, min(100.0, estimated_temp))
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']