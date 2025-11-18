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
        self.required_features = [
            'air_temp', 'track_temp', 'humidity', 'pressure', 'wind_speed', 'rain',
            'temp_difference', 'air_density', 'track_weather_sensitivity', 'lap_number'
        ]
    
    def train(self, processed_data: dict) -> dict:
        """Train weather impact model with comprehensive error handling"""
        try:
            # Validate input data structure
            if not self._validate_input_data(processed_data):
                return {'error': 'Invalid or insufficient weather training data'}
            
            # Extract features from all sessions
            features_list, impact_list = self._extract_all_weather_features(processed_data)
            
            if not features_list:
                return {'error': 'No valid weather features extracted from sessions'}

            # Combine and validate training data
            X, y = self._prepare_training_data(features_list, impact_list)
            if len(X) < 15:
                return {'error': f'Insufficient training samples: {len(X)}'}

            # Train model with validation
            training_result = self._train_model_with_validation(X, y)
            if 'error' in training_result:
                return training_result

            return {
                'model': self,
                'features': self.feature_columns,
                'train_score': training_result['train_score'],
                'test_score': training_result['test_score'],
                'feature_importance': training_result['feature_importance'],
                'training_samples': training_result['training_samples']
            }

        except Exception as e:
            return {'error': f'Weather model training failed: {str(e)}'}

    def _validate_input_data(self, processed_data: dict) -> bool:
        """Validate input data structure and content"""
        if not isinstance(processed_data, dict):
            return False
        
        valid_sessions = 0
        for session_key, data in processed_data.items():
            if (isinstance(data, dict) and 
                data.get('weather_data') is not None and 
                data.get('lap_data') is not None):
                valid_sessions += 1
        
        return valid_sessions >= 1  # Reduced minimum for flexibility

    def _extract_all_weather_features(self, processed_data: dict) -> tuple:
        """Extract weather features from all sessions with robust error handling"""
        features_list = []
        impact_list = []

        for session_key, data in processed_data.items():
            try:
                session_features, session_impacts = self._extract_session_weather_features(data, session_key)
                if (not session_features.empty and len(session_impacts) > 0 and
                    len(session_features) == len(session_impacts)):
                    features_list.append(session_features)
                    impact_list.append(session_impacts)
            except Exception as e:
                print(f"⚠️ Session {session_key} weather feature extraction failed: {e}")
                continue

        return features_list, impact_list

    def _extract_session_weather_features(self, data: dict, session_key: str) -> tuple:
        """Extract weather features for a single session"""
        features = []
        impacts = []

        try:
            weather_data = data.get('weather_data', pd.DataFrame())
            lap_data = data.get('lap_data', pd.DataFrame())
            telemetry_data = data.get('telemetry_data', pd.DataFrame())

            if lap_data.empty or weather_data.empty:
                return pd.DataFrame(), np.array([])

            # Prepare timestamps with error handling
            weather_data = self._prepare_weather_timestamps(weather_data)
            lap_data = self._prepare_lap_timestamps(lap_data)

            # Process each car in the session
            for car_number in lap_data['NUMBER'].unique():
                try:
                    car_features, car_impacts = self._extract_car_weather_features(
                        car_number, lap_data, weather_data, telemetry_data, session_key
                    )
                    if car_features is not None and car_impacts is not None:
                        features.extend(car_features)
                        impacts.extend(car_impacts)
                except Exception as e:
                    print(f"⚠️ Car {car_number} weather feature extraction failed: {e}")
                    continue

        except Exception as e:
            print(f"❌ Session weather feature extraction failed: {e}")
            return pd.DataFrame(), np.array([])

        # Convert to DataFrame and array
        if features and impacts:
            try:
                features_df = pd.DataFrame(features)
                features_df = self._ensure_required_features(features_df)
                return features_df, np.array(impacts, dtype=float)
            except Exception as e:
                print(f"❌ Feature combination failed: {e}")

        return pd.DataFrame(), np.array([])

    def _extract_car_weather_features(self, car_number: int, lap_data: pd.DataFrame, 
                                     weather_data: pd.DataFrame, telemetry_data: pd.DataFrame, 
                                     session_key: str) -> tuple:
        """Extract weather features for a single car"""
        car_features = []
        car_impacts = []

        try:
            car_laps = lap_data[lap_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
            
            if len(car_laps) < 3:  # Reduced minimum for more data
                return None, None

            baseline_time = self._calculate_baseline_performance(car_laps)

            for _, lap in car_laps.iterrows():
                try:
                    lap_weather = self._get_lap_weather_conditions(lap, weather_data)
                    if lap_weather is None:
                        continue

                    lap_telemetry = self._get_lap_telemetry(lap, telemetry_data, car_number)
                    weather_impact = self._calculate_weather_impact(lap, baseline_time, lap_weather)
                    feature_vector = self._create_weather_feature_vector(lap, lap_weather, lap_telemetry, session_key)

                    car_features.append(feature_vector)
                    car_impacts.append(weather_impact)

                except Exception as e:
                    print(f"⚠️ Lap {lap.get('LAP_NUMBER', 'unknown')} processing failed: {e}")
                    continue

        except Exception as e:
            print(f"❌ Car weather feature extraction failed: {e}")
            return None, None

        return car_features, car_impacts if car_features else None

    def _prepare_weather_timestamps(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare weather timestamps with robust error handling"""
        df = weather_data.copy()
        
        try:
            if 'TIME_UTC_STR' in df.columns:
                df['timestamp'] = pd.to_datetime(df['TIME_UTC_STR'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
            elif 'TIME_UTC_SECONDS' in df.columns:
                df['timestamp'] = pd.to_datetime(df['TIME_UTC_SECONDS'], unit='s', errors='coerce')
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            else:
                # Generate reasonable timestamps
                start_time = datetime.now() - timedelta(hours=2)
                df['timestamp'] = [start_time + timedelta(seconds=i*30) for i in range(len(df))]
            
            # Remove rows with invalid timestamps
            df = df[df['timestamp'].notna()]
            return df.sort_values('timestamp')
            
        except Exception as e:
            print(f"⚠️ Weather timestamp preparation failed: {e}")
            # Fallback: generate sequential timestamps
            start_time = datetime.now() - timedelta(hours=2)
            df['timestamp'] = [start_time + timedelta(seconds=i*30) for i in range(len(df))]
            return df

    def _prepare_lap_timestamps(self, lap_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare lap timestamps with robust error handling"""
        df = lap_data.copy()
        
        try:
            if 'HOUR' in df.columns:
                df['timestamp'] = pd.to_datetime(df['HOUR'], format='%H:%M:%S.%f', errors='coerce')
            elif 'ELAPSED' in df.columns:
                base_time = datetime.now().replace(hour=14, minute=0, second=0)
                df['timestamp'] = df['ELAPSED'].apply(
                    lambda x: base_time + timedelta(seconds=float(x)) if pd.notna(x) and str(x).strip() else base_time
                )
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            else:
                # Generate reasonable lap timestamps
                base_time = datetime.now().replace(hour=14, minute=0, second=0)
                df['timestamp'] = [base_time + timedelta(seconds=i*90) for i in range(len(df))]
            
            # Ensure LAP_NUMBER exists
            if 'LAP_NUMBER' not in df.columns:
                df['LAP_NUMBER'] = range(1, len(df) + 1)
                
            return df
            
        except Exception as e:
            print(f"⚠️ Lap timestamp preparation failed: {e}")
            # Fallback: generate sequential timestamps
            base_time = datetime.now().replace(hour=14, minute=0, second=0)
            df['timestamp'] = [base_time + timedelta(seconds=i*90) for i in range(len(df))]
            if 'LAP_NUMBER' not in df.columns:
                df['LAP_NUMBER'] = range(1, len(df) + 1)
            return df

    def _get_lap_weather_conditions(self, lap: pd.Series, weather_data: pd.DataFrame) -> pd.Series:
        """Get weather conditions for a specific lap with error handling"""
        try:
            lap_time = lap.get('timestamp')
            if lap_time is None or weather_data.empty:
                return None

            if 'timestamp' not in weather_data.columns:
                return None

            # Find closest weather reading within reasonable time window
            time_diff = (weather_data['timestamp'] - lap_time).abs()
            closest_idx = time_diff.idxmin()
            
            # Only use if within 10 minutes (increased tolerance)
            if time_diff[closest_idx] <= timedelta(minutes=10):
                return weather_data.loc[closest_idx]
            else:
                return None
                
        except Exception as e:
            print(f"⚠️ Lap weather condition retrieval failed: {e}")
            return None

    def _get_lap_telemetry(self, lap: pd.Series, telemetry_data: pd.DataFrame, car_number: int) -> dict:
        """Get telemetry data for a specific lap with error handling"""
        try:
            if telemetry_data.empty:
                return self._get_telemetry_fallback()

            lap_num = lap.get('LAP_NUMBER', 1)
            
            # Try different column name variations
            vehicle_col = 'vehicle_number' if 'vehicle_number' in telemetry_data.columns else 'NUMBER'
            lap_col = 'lap' if 'lap' in telemetry_data.columns else 'LAP_NUMBER'
            
            lap_telemetry = telemetry_data[
                (telemetry_data[vehicle_col] == car_number) & 
                (telemetry_data[lap_col] == lap_num)
            ]
            
            if lap_telemetry.empty:
                return self._get_telemetry_fallback()

            return {
                'avg_throttle': self._safe_column_mean(lap_telemetry, 'aps', 60.0),
                'avg_brake': self._safe_brake_pressure(lap_telemetry),
                'avg_speed': self._safe_column_mean(lap_telemetry, 'KPH', 120.0),
                'steering_variance': self._safe_column_variance(lap_telemetry, 'Steering_Angle', 10.0)
            }
            
        except Exception as e:
            print(f"⚠️ Lap telemetry retrieval failed: {e}")
            return self._get_telemetry_fallback()

    def _calculate_baseline_performance(self, car_laps: pd.DataFrame) -> float:
        """Calculate baseline performance with robust error handling"""
        try:
            lap_times = pd.to_numeric(car_laps['LAP_TIME_SECONDS'], errors='coerce').dropna()
            
            if len(lap_times) < 2:
                return float(lap_times.mean()) if not lap_times.empty else 100.0
            
            # Use best 30% of laps or minimum 3 laps
            n_best = max(2, int(len(lap_times) * 0.3))
            best_laps = lap_times.nsmallest(n_best)
            
            return float(best_laps.median())
            
        except Exception as e:
            print(f"⚠️ Baseline performance calculation failed: {e}")
            return 100.0

    def _calculate_weather_impact(self, lap: pd.Series, baseline_time: float, weather: pd.Series) -> float:
        """Calculate weather impact on lap performance"""
        try:
            actual_time = self._safe_get_value(lap, 'LAP_TIME_SECONDS', baseline_time)
            impact = actual_time - baseline_time
            
            # Only consider impacts beyond normal variation
            normal_variation = 1.0  # Increased tolerance
            adjusted_impact = impact if abs(impact) > normal_variation else 0
            
            return float(max(-8.0, min(8.0, adjusted_impact)))  # Increased range
            
        except Exception as e:
            print(f"⚠️ Weather impact calculation failed: {e}")
            return 0.0

    def _create_weather_feature_vector(self, lap: pd.Series, weather: pd.Series, 
                                     telemetry: dict, session_key: str) -> dict:
        """Create weather feature vector with comprehensive error handling"""
        features = {}
        
        try:
            # Basic weather features
            features['air_temp'] = float(self._safe_get_value(weather, 'AIR_TEMP', 25.0))
            features['track_temp'] = float(self._safe_get_value(weather, 'TRACK_TEMP', 30.0))
            features['humidity'] = float(self._safe_get_value(weather, 'HUMIDITY', 50.0))
            features['pressure'] = float(self._safe_get_value(weather, 'PRESSURE', 1013.0))
            features['wind_speed'] = float(self._safe_get_value(weather, 'WIND_SPEED', 0.0))
            features['wind_direction'] = float(self._safe_get_value(weather, 'WIND_DIRECTION', 0.0))
            features['rain'] = float(self._safe_get_value(weather, 'RAIN', 0.0))

            # Derived features
            features['temp_difference'] = features['track_temp'] - features['air_temp']
            features['air_density'] = self._calculate_air_density(
                features['air_temp'], features['pressure'], features['humidity']
            )
            features['wind_effect'] = self._calculate_wind_effect(
                features['wind_speed'], features['wind_direction']
            )

            # Track and context features
            track_name = session_key.split('_')[0] if '_' in session_key else session_key
            features['track_weather_sensitivity'] = self._get_track_weather_sensitivity(track_name)
            features['lap_number'] = int(self._safe_get_value(lap, 'LAP_NUMBER', 1))
            
            # Time of day feature
            lap_time = lap.get('timestamp')
            if lap_time and hasattr(lap_time, 'hour'):
                features['time_of_day'] = lap_time.hour + lap_time.minute / 60.0
            else:
                features['time_of_day'] = 14.0  # Default afternoon

            # Telemetry features
            features.update({
                'throttle_usage': telemetry.get('avg_throttle', 60.0),
                'braking_intensity': telemetry.get('avg_brake', 50.0),
                'avg_speed': telemetry.get('avg_speed', 120.0),
                'steering_activity': telemetry.get('steering_variance', 10.0)
            })

        except Exception as e:
            print(f"⚠️ Weather feature vector creation failed: {e}")
            # Fallback features
            features.update(self._get_weather_fallback_features())

        return features

    def _calculate_air_density(self, air_temp: float, pressure: float, humidity: float) -> float:
        """Calculate air density with error handling"""
        try:
            R = 287.05  # Specific gas constant for dry air (J/kg·K)
            temp_kelvin = air_temp + 273.15
            
            # Calculate vapor pressure (Buck equation)
            vapor_pressure = 0.611 * np.exp(17.27 * air_temp / (air_temp + 237.3)) * (humidity / 100.0)
            
            # Calculate dry air pressure (Pa)
            dry_air_pressure = (pressure - vapor_pressure) * 100.0
            
            # Calculate air density (kg/m³)
            air_density = dry_air_pressure / (R * temp_kelvin)
            
            return float(max(1.0, min(1.5, air_density)))  # Reasonable bounds
            
        except Exception:
            return 1.225  # Standard sea level density

    def _calculate_wind_effect(self, wind_speed: float, wind_direction: float) -> float:
        """Calculate wind effect on performance"""
        try:
            # Simple model: headwind/tailwind component affects straightline speed
            # Assume optimal wind direction is 0 (headwind) for simplicity
            wind_effect = wind_speed * 0.15  # 0.15 seconds per m/s wind speed
            return float(wind_effect)
        except Exception:
            return 0.0

    def _get_track_weather_sensitivity(self, track_name: str) -> float:
        """Get track weather sensitivity factor"""
        sensitivity_map = {
            'road-america': 0.9, 'sebring': 0.85, 'barber': 0.8,
            'sonoma': 0.75, 'vir': 0.7, 'cota': 0.65, 'indianapolis': 0.5,
            'circuit-of-the-americas': 0.65, 'sonoma': 0.75
        }
        return sensitivity_map.get(track_name.lower(), 0.7)

    # ---------------------------
    # UTILITY METHODS
    # ---------------------------
    def _safe_get_value(self, series: pd.Series, key: str, default: any) -> any:
        """Safely get value from pandas Series with fallback"""
        try:
            value = series.get(key, default)
            return value if not pd.isna(value) else default
        except (KeyError, TypeError, AttributeError):
            return default

    def _safe_column_mean(self, df: pd.DataFrame, column: str, default: float) -> float:
        """Safely calculate column mean with fallback"""
        try:
            if column not in df.columns:
                return default
            values = pd.to_numeric(df[column], errors='coerce').dropna()
            return float(values.mean()) if not values.empty else default
        except Exception:
            return default

    def _safe_column_variance(self, df: pd.DataFrame, column: str, default: float) -> float:
        """Safely calculate column variance with fallback"""
        try:
            if column not in df.columns:
                return default
            values = pd.to_numeric(df[column], errors='coerce').dropna()
            return float(values.var()) if len(values) > 1 else default
        except Exception:
            return default

    def _safe_brake_pressure(self, telemetry_data: pd.DataFrame) -> float:
        """Safely calculate average brake pressure"""
        try:
            front_brake = self._safe_column_mean(telemetry_data, 'pbrake_f', 25.0)
            rear_brake = self._safe_column_mean(telemetry_data, 'pbrake_r', 25.0)
            return (front_brake + rear_brake) / 2.0
        except Exception:
            return 25.0

    def _get_telemetry_fallback(self) -> dict:
        """Get telemetry fallback values"""
        return {
            'avg_throttle': 60.0,
            'avg_brake': 50.0,
            'avg_speed': 120.0,
            'steering_variance': 10.0
        }

    def _get_weather_fallback_features(self) -> dict:
        """Get comprehensive weather feature fallbacks"""
        return {
            'air_temp': 25.0,
            'track_temp': 30.0,
            'humidity': 50.0,
            'pressure': 1013.0,
            'wind_speed': 0.0,
            'wind_direction': 0.0,
            'rain': 0.0,
            'temp_difference': 5.0,
            'air_density': 1.225,
            'wind_effect': 0.0,
            'track_weather_sensitivity': 0.7,
            'lap_number': 1,
            'time_of_day': 14.0,
            'throttle_usage': 60.0,
            'braking_intensity': 50.0,
            'avg_speed': 120.0,
            'steering_activity': 10.0
        }

    def _ensure_required_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required features are present"""
        df = features_df.copy()
        for feature in self.required_features:
            if feature not in df.columns:
                df[feature] = 0.0
        return df

    def _prepare_training_data(self, features_list: list, impact_list: list) -> tuple:
        """Prepare training data with proper validation"""
        try:
            if not features_list:
                return pd.DataFrame(), np.array([])
                
            X = pd.concat(features_list, ignore_index=True)
            y = np.concatenate(impact_list) if impact_list else np.array([])
            
            # Remove rows with NaNs
            if len(y) > 0:
                valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
                X = X[valid_mask]
                y = y[valid_mask]
            
            # Ensure consistent data types
            X = self._enforce_feature_types(X)
            
            return X, y
            
        except Exception as e:
            print(f"❌ Training data preparation failed: {e}")
            return pd.DataFrame(), np.array([])

    def _train_model_with_validation(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        """Train model with comprehensive validation"""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            self.feature_columns = X.columns.tolist()

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Train model
            self.model.fit(X_train, y_train)
            
            # Calculate scores
            train_score = float(self.model.score(X_train, y_train))
            test_score = float(self.model.score(X_test, y_test))
            
            # Feature importance
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))

            return {
                'train_score': train_score,
                'test_score': test_score,
                'feature_importance': feature_importance,
                'training_samples': len(X)
            }

        except Exception as e:
            return {'error': f'Model training failed: {str(e)}'}

    def _enforce_feature_types(self, X: pd.DataFrame) -> pd.DataFrame:
        """Enforce consistent feature data types"""
        X = X.copy()
        for col in X.columns:
            try:
                if col == 'lap_number':
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(1).astype(int)
                else:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0).astype(float)
            except Exception:
                X[col] = 0.0
        return X

    # ---------------------------
    # PREDICTION METHODS
    # ---------------------------
    def predict_weather_impact(self, weather_conditions: dict, track_name: str, lap_context: dict = None) -> float:
        """Predict weather impact with comprehensive error handling"""
        try:
            if not self.feature_columns:
                return self._fallback_weather_impact(weather_conditions, track_name)
            
            # Create feature vector
            feature_vector = self._create_prediction_features(weather_conditions, track_name, lap_context)
            
            # Convert to array and predict
            feature_array = np.array([feature_vector.get(col, 0.0) for col in self.feature_columns]).reshape(1, -1)
            scaled_features = self.scaler.transform(feature_array)
            impact = self.model.predict(scaled_features)[0]
            
            return float(max(-8.0, min(8.0, impact)))
            
        except Exception as e:
            print(f"⚠️ Weather impact prediction failed: {e}")
            return self._fallback_weather_impact(weather_conditions, track_name)

    def _create_prediction_features(self, weather_conditions: dict, track_name: str, lap_context: dict = None) -> dict:
        """Create features for prediction"""
        if lap_context is None:
            lap_context = {}
            
        # Mock lap series for feature creation
        mock_lap = pd.Series({
            'LAP_NUMBER': lap_context.get('lap_number', 1),
            'timestamp': lap_context.get('timestamp', datetime.now())
        })
        
        # Mock telemetry
        mock_telemetry = {
            'avg_throttle': lap_context.get('throttle_usage', 60.0),
            'avg_brake': lap_context.get('braking_intensity', 50.0),
            'avg_speed': lap_context.get('avg_speed', 120.0),
            'steering_variance': lap_context.get('steering_activity', 10.0)
        }
        
        # Mock weather series
        mock_weather = pd.Series(weather_conditions)
        
        return self._create_weather_feature_vector(mock_lap, mock_weather, mock_telemetry, track_name)

    def _fallback_weather_impact(self, weather_conditions: dict, track_name: str) -> float:
        """Fallback weather impact calculation"""
        base_impact = 0.0
        
        # Temperature effect (optimal around 22°C)
        temp_diff = abs(weather_conditions.get('air_temp', 25) - 22.0)
        base_impact += temp_diff * 0.08
        
        # Rain effect
        if weather_conditions.get('rain', 0) > 0:
            base_impact += 2.5
        
        # Wind effect
        wind_speed = weather_conditions.get('wind_speed', 0)
        base_impact += wind_speed * 0.1
        
        # Track sensitivity
        sensitivity = self._get_track_weather_sensitivity(track_name)
        
        return base_impact * sensitivity

    def get_optimal_conditions(self, track_name: str) -> dict:
        """Get optimal weather conditions for a track"""
        return {
            'AIR_TEMP': 22.0,
            'TRACK_TEMP': 28.0,
            'HUMIDITY': 50.0,
            'PRESSURE': 1013.0,
            'WIND_SPEED': 2.0,
            'RAIN': 0.0
        }

    def estimate_tire_temperature(self, weather_conditions: dict, track_name: str, lap_count: int) -> float:
        """Estimate tire temperature based on conditions and usage"""
        try:
            base_temp = weather_conditions.get('track_temp', 30.0)
            air_temp = weather_conditions.get('air_temp', 25.0)
            
            # Heat from usage
            usage_heat = min(20.0, lap_count * 0.6)
            
            # Track-specific heating
            track_heat = self._get_track_weather_sensitivity(track_name) * 6.0
            
            # Cooling from air temperature difference
            cooling_effect = max(0, (base_temp - air_temp) * 0.15)
            
            estimated_temp = base_temp + usage_heat + track_heat - cooling_effect
            
            return float(max(air_temp, min(110.0, estimated_temp)))
            
        except Exception:
            return 35.0  # Reasonable fallback

    # ---------------------------
    # MODEL SERIALIZATION
    # ---------------------------
    def save_model(self, filepath: str):
        """Save model with error handling"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }
            joblib.dump(model_data, filepath)
        except Exception as e:
            print(f"❌ Model save failed: {e}")

    def load_model(self, filepath: str):
        """Load model with error handling"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            # Reinitialize with defaults
            self.__init__()

    # ---------------------------
    # BACKWARD COMPATIBILITY
    # ---------------------------
    def update_model(self, existing_model_result: dict, new_processed_data: dict) -> dict:
        """Update existing weather model with new data"""
        try:
            # Extract the trainer from existing model result
            if isinstance(existing_model_result, dict) and 'model' in existing_model_result:
                existing_trainer = existing_model_result['model']
                if hasattr(existing_trainer, 'model') and hasattr(existing_trainer, 'feature_columns'):
                    # Use existing model as starting point
                    self.model = existing_trainer.model
                    self.feature_columns = existing_trainer.feature_columns
                    self.scaler = existing_trainer.scaler
                    
                    # Train with new data
                    return self.train(new_processed_data)
            
            # Fallback: train new model
            print("⚠️ Existing model structure invalid, training new model")
            return self.train(new_processed_data)
            
        except Exception as e:
            print(f"⚠️ Incremental update failed: {e}")
            return self.train(new_processed_data)


















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

#     def update_model(self, existing_model_result: dict, new_processed_data: dict) -> dict:
#         """Update existing weather model with new data - maintains backward compatibility"""
#         try:
#             # Extract the trainer from existing model result
#             if isinstance(existing_model_result, dict) and 'model' in existing_model_result:
#                 existing_trainer = existing_model_result['model']
#                 if hasattr(existing_trainer, 'model') and hasattr(existing_trainer, 'feature_columns'):
#                     # Use existing model as starting point
#                     self.model = existing_trainer.model
#                     self.feature_columns = existing_trainer.feature_columns
#                     self.scaler = existing_trainer.scaler
                    
#                     # Train with new data (incremental learning)
#                     return self.train(new_processed_data)
            
#             # Fallback: train new model if existing model structure is invalid
#             self.logger.warning("Existing model structure invalid, training new model")
#             return self.train(new_processed_data)
            
#         except Exception as e:
#             # Fallback to full retraining if incremental update fails
#             print(f"⚠️ Incremental update failed, performing full retraining: {e}")
#             return self.train(new_processed_data)
    
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
#             weather_data['timestamp'] = pd.to_datetime(weather_data['TIME_UTC_STR'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
#         elif 'TIME_UTC_SECONDS' in weather_data.columns:
#             weather_data['timestamp'] = pd.to_datetime(weather_data['TIME_UTC_SECONDS'], unit='s', errors='coerce')
#         else:
#             start_time = datetime.now() - timedelta(hours=2)
#             weather_data['timestamp'] = [start_time + timedelta(seconds=i*30) for i in range(len(weather_data))]
#         return weather_data.sort_values('timestamp')
    
#     def _prepare_lap_timestamps(self, lap_data: pd.DataFrame) -> pd.DataFrame:
#         if 'HOUR' in lap_data.columns:
#             lap_data['timestamp'] = pd.to_datetime(lap_data['HOUR'], format='%H:%M:%S.%f', errors='coerce')
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