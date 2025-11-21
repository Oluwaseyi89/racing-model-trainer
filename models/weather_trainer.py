# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import joblib
# from datetime import datetime, timedelta
# from typing import Dict, List, Tuple
# import logging

# class WeatherModelTrainer:
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#         self.scaler = StandardScaler()
#         self.feature_columns = []
#         self.logger = logging.getLogger(__name__)
        
#         # Enhanced minimum requirements with validation
#         self.minimum_weather_cols = ['AIR_TEMP', 'TRACK_TEMP']
#         self.minimum_pit_cols = ['NUMBER', 'LAP_NUMBER', 'LAP_TIME']

#     def train(self, track_data: Dict[str, Dict[str, pd.DataFrame]]) -> dict:
#         """Train weather impact model with comprehensive error handling and data validation"""
#         try:
#             self.logger.info("🚀 Starting weather impact model training...")
            
#             features_list = []
#             impact_list = []
#             processed_tracks = []
            
#             # Process each track's data with enhanced validation
#             for track_name, data_dict in track_data.items():
#                 self.logger.info(f"📊 Processing track: {track_name}")
                
#                 if not self._validate_track_data(data_dict):
#                     self.logger.warning(f"⚠️ Skipping {track_name}: validation failed")
#                     continue
                    
#                 session_features, session_impacts = self._extract_weather_features(data_dict, track_name)
                
#                 if not session_features.empty and len(session_impacts) > 0:
#                     features_list.append(session_features)
#                     impact_list.extend(session_impacts)
#                     processed_tracks.append(track_name)
#                     self.logger.info(f"✅ {track_name}: extracted {len(session_features)} weather samples")
#                 else:
#                     self.logger.warning(f"⚠️ No weather features extracted from {track_name}")
            
#             if not features_list:
#                 return self._train_with_fallback("No valid weather features extracted from any track", processed_tracks)
            
#             # Combine all track data
#             X = pd.concat(features_list, ignore_index=True)
#             y = np.array(impact_list)
            
#             if X.empty or len(y) == 0:
#                 return self._train_with_fallback("Empty feature or target matrices after processing", processed_tracks)
            
#             self.logger.info(f"📈 Final dataset: {len(X)} samples, {len(X.columns)} features")

#             # Remove NaN values
#             valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
#             X_clean = X[valid_mask]
#             y_clean = y[valid_mask]
            
#             if len(X_clean) < 15:
#                 return self._train_with_fallback(f"Insufficient training samples: {len(X_clean)}", processed_tracks)
            
#             # Scale features and train
#             X_scaled = self.scaler.fit_transform(X_clean)
#             self.feature_columns = X_clean.columns.tolist()
            
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X_scaled, y_clean, test_size=0.2, random_state=42
#             )
            
#             self.logger.info("🏃 Training model...")
#             self.model.fit(X_train, y_train)
#             train_score = self.model.score(X_train, y_train)
#             test_score = self.model.score(X_test, y_test)
            
#             feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))

#             self.logger.info(f"✅ Weather impact model trained successfully - Test Score: {test_score:.3f}")

#             return {
#                 'model': self,
#                 'features': self.feature_columns,
#                 'train_score': train_score,
#                 'test_score': test_score,
#                 'feature_importance': feature_importance,
#                 'training_samples': len(X_clean),
#                 'tracks_used': len(processed_tracks),
#                 'processed_tracks': processed_tracks,
#                 'status': 'success'
#             }
            
#         except Exception as e:
#             error_msg = f'Training failed: {str(e)}'
#             self.logger.error(f"❌ {error_msg}")
#             return self._train_with_fallback(error_msg, [])

#     def _validate_track_data(self, data_dict: Dict[str, pd.DataFrame]) -> bool:
#         """Enhanced validation with comprehensive weather data checking"""
#         weather_data = data_dict.get('weather_data', pd.DataFrame())
#         pit_data = data_dict.get('pit_data', pd.DataFrame())
        
#         if weather_data.empty or pit_data.empty:
#             self.logger.debug("⚠️ Missing weather_data or pit_data")
#             return False
            
#         # Check for minimum required columns
#         missing_weather = [col for col in self.minimum_weather_cols if col not in weather_data.columns]
#         missing_pit = [col for col in self.minimum_pit_cols if col not in pit_data.columns]
        
#         if missing_weather or missing_pit:
#             self.logger.debug(f"⚠️ Missing required columns - weather: {missing_weather}, pit: {missing_pit}")
#             return False
        
#         # Enhanced weather data quality validation
#         if not self._validate_weather_data_quality(weather_data):
#             self.logger.debug("⚠️ Weather data quality validation failed")
#             return False
            
#         # Check for minimum data volume
#         if len(weather_data) < 3 or len(pit_data) < 5:
#             self.logger.debug(f"⚠️ Insufficient data - weather: {len(weather_data)}, pit: {len(pit_data)}")
#             return False
            
#         self.logger.debug(f"✅ Track data validated: {len(weather_data)} weather rows, {len(pit_data)} pit rows")
#         return True

#     def _validate_weather_data_quality(self, weather_data: pd.DataFrame) -> bool:
#         """Comprehensive weather data quality validation"""
#         try:
#             # Check for valid temperature ranges
#             if 'AIR_TEMP' in weather_data.columns:
#                 air_temp_valid = weather_data['AIR_TEMP'].between(-10, 50).any()
#                 if not air_temp_valid:
#                     self.logger.debug("⚠️ No valid AIR_TEMP values in range")
#                     return False
            
#             # Critical fix: Handle TRACK_TEMP = 0 values
#             if 'TRACK_TEMP' in weather_data.columns:
#                 track_temp_series = weather_data['TRACK_TEMP']
#                 valid_track_temps = track_temp_series.between(10, 80)  # Reasonable track temp range
                
#                 if not valid_track_temps.any():
#                     self.logger.debug("⚠️ No valid TRACK_TEMP values in range, will use fallbacks")
#                     # Don't fail validation - we'll handle this in processing
                
#             # Check for reasonable humidity values
#             if 'HUMIDITY' in weather_data.columns:
#                 humidity_valid = weather_data['HUMIDITY'].between(0, 100).any()
#                 if not humidity_valid:
#                     self.logger.debug("⚠️ No valid HUMIDITY values in range")
            
#             return True
            
#         except Exception as e:
#             self.logger.debug(f"⚠️ Weather data quality check failed: {e}")
#             return True  # Don't fail validation, let processing handle it

#     def _extract_weather_features(self, data_dict: Dict[str, Dict[str, pd.DataFrame]], track_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
#         """Extract weather impact features with comprehensive data repair and validation"""
#         weather_data = data_dict.get('weather_data', pd.DataFrame())
#         pit_data = data_dict.get('pit_data', pd.DataFrame())
#         telemetry_data = data_dict.get('telemetry_data', pd.DataFrame())
        
#         features = []
#         impacts = []
        
#         if pit_data.empty or weather_data.empty:
#             self.logger.debug("⚠️ Missing pit_data or weather_data")
#             return pd.DataFrame(), np.array([])
        
#         # Prepare timestamps with enhanced error handling and data repair
#         weather_data_clean = self._prepare_weather_data_with_repair(weather_data)
#         pit_data_clean = self._prepare_pit_timestamps(pit_data)
        
#         if weather_data_clean.empty or pit_data_clean.empty:
#             self.logger.debug("⚠️ Failed to prepare timestamps or repair data")
#             return pd.DataFrame(), np.array([])
        
#         self.logger.debug(f"🔧 Processing {len(pit_data_clean['NUMBER'].unique())} cars in {track_name}")

#         # Process each car's laps with enhanced baseline calculation
#         for car_number in pit_data_clean['NUMBER'].unique():
#             car_laps = pit_data_clean[pit_data_clean['NUMBER'] == car_number].sort_values('LAP_NUMBER')
#             if len(car_laps) < 3:
#                 self.logger.debug(f"⚠️ Car {car_number}: insufficient laps ({len(car_laps)})")
#                 continue
                
#             baseline_time = self._calculate_baseline_performance(car_laps)
            
#             for _, lap in car_laps.iterrows():
#                 lap_weather = self._get_lap_weather_conditions(lap, weather_data_clean)
#                 if lap_weather is None:
#                     continue
                    
#                 lap_telemetry = self._get_lap_telemetry(lap, telemetry_data, car_number)
#                 weather_impact = self._calculate_weather_impact(lap, baseline_time, lap_weather)
#                 feature_vector = self._create_weather_feature_vector(lap, lap_weather, lap_telemetry, track_name)
                
#                 features.append(pd.DataFrame([feature_vector]))
#                 impacts.append(weather_impact)
        
#         if features:
#             result_features = pd.concat(features, ignore_index=True)
#             self.logger.debug(f"✅ {track_name}: extracted {len(result_features)} weather impact samples")
#             return result_features, np.array(impacts)
        
#         self.logger.debug(f"❌ {track_name}: no weather features extracted")
#         return pd.DataFrame(), np.array([])
    
#     def _prepare_weather_data_with_repair(self, weather_data: pd.DataFrame) -> pd.DataFrame:
#         """Prepare weather data with comprehensive repair for invalid values"""
#         weather_data = weather_data.copy()
        
#         try:
#             # Fix TRACK_TEMP = 0 values - CRITICAL FIX
#             if 'TRACK_TEMP' in weather_data.columns:
#                 track_temp_series = weather_data['TRACK_TEMP']
#                 zero_track_temp_mask = (track_temp_series == 0) | (track_temp_series < 10)
                
#                 if zero_track_temp_mask.any():
#                     zero_count = zero_track_temp_mask.sum()
#                     self.logger.info(f"🔧 Repairing {zero_count} invalid TRACK_TEMP values")
                    
#                     # Calculate realistic track temperatures based on air temp
#                     if 'AIR_TEMP' in weather_data.columns:
#                         # Track temp is typically 5-15°C higher than air temp
#                         air_temp = weather_data['AIR_TEMP']
#                         realistic_track_temp = air_temp + np.random.uniform(8, 12, len(weather_data))
#                         weather_data.loc[zero_track_temp_mask, 'TRACK_TEMP'] = realistic_track_temp[zero_track_temp_mask]
#                     else:
#                         # Fallback: use reasonable default range
#                         weather_data.loc[zero_track_temp_mask, 'TRACK_TEMP'] = np.random.uniform(25, 35, zero_count)
            
#             # Prepare timestamps
#             if 'TIME_UTC_SECONDS' in weather_data.columns:
#                 weather_data['timestamp'] = pd.to_datetime(weather_data['TIME_UTC_SECONDS'], unit='s', errors='coerce')
#             elif 'TIME_UTC_STR' in weather_data.columns:
#                 weather_data['timestamp'] = pd.to_datetime(weather_data['TIME_UTC_STR'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
#             else:
#                 # Enhanced fallback: use index-based timestamps
#                 start_time = datetime.now() - timedelta(hours=2)
#                 weather_data['timestamp'] = [start_time + timedelta(seconds=i*60) for i in range(len(weather_data))]
            
#             # Filter out invalid timestamps
#             weather_data = weather_data.dropna(subset=['timestamp'])
            
#             # Validate final data quality
#             valid_track_temps = weather_data['TRACK_TEMP'].between(10, 80) if 'TRACK_TEMP' in weather_data.columns else True
#             if not valid_track_temps.all():
#                 self.logger.warning("⚠️ Some TRACK_TEMP values still outside reasonable range after repair")
            
#             return weather_data.sort_values('timestamp')
            
#         except Exception as e:
#             self.logger.warning(f"⚠️ Weather data preparation failed: {e}")
#             return pd.DataFrame()

#     def _prepare_pit_timestamps(self, pit_data: pd.DataFrame) -> pd.DataFrame:
#         """Prepare pit data timestamps with enhanced fallbacks"""
#         pit_data = pit_data.copy()
        
#         try:
#             if 'HOUR' in pit_data.columns:
#                 # Enhanced HOUR parsing with multiple format attempts
#                 base_date = datetime.now().date()
#                 pit_data['timestamp'] = pd.to_datetime(
#                     base_date.strftime('%Y-%m-%d') + ' ' + pit_data['HOUR'].astype(str),
#                     errors='coerce'
#                 )
                
#                 # Fallback for failed parses
#                 if pit_data['timestamp'].isna().any():
#                     pit_data['timestamp'] = pd.to_datetime(pit_data['HOUR'], errors='coerce')
                    
#             elif 'ELAPSED' in pit_data.columns:
#                 base_time = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)
#                 pit_data['timestamp'] = pit_data['ELAPSED'].apply(
#                     lambda x: base_time + timedelta(seconds=float(x)) if pd.notna(x) and str(x).strip() else pd.NaT
#                 )
#             else:
#                 # Enhanced synthetic timestamps
#                 base_time = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)
#                 pit_data['timestamp'] = [base_time + timedelta(seconds=i*120) for i in range(len(pit_data))]
            
#             # Filter out invalid timestamps
#             pit_data = pit_data.dropna(subset=['timestamp'])
#             return pit_data.sort_values('timestamp')
            
#         except Exception as e:
#             self.logger.warning(f"⚠️ Pit timestamp preparation failed: {e}")
#             return pd.DataFrame()
    
#     def _get_lap_weather_conditions(self, lap: pd.Series, weather_data: pd.DataFrame) -> pd.Series | None:
#         """Get weather conditions for a specific lap with enhanced matching and fallbacks"""
#         if weather_data.empty:
#             return None
            
#         try:
#             lap_time = lap['timestamp']
#             time_diff = (weather_data['timestamp'] - lap_time).abs()
#             closest_idx = time_diff.idxmin()
            
#             # Increased time tolerance for more matches
#             if time_diff[closest_idx] > timedelta(minutes=10):
#                 return None
                
#             weather_conditions = weather_data.loc[closest_idx]
            
#             # Final validation of weather conditions
#             if not self._validate_weather_conditions(weather_conditions):
#                 self.logger.debug("⚠️ Invalid weather conditions found, using fallback")
#                 return self._create_fallback_weather_conditions()
                
#             return weather_conditions
            
#         except Exception as e:
#             self.logger.debug(f"⚠️ Weather condition lookup failed: {e}")
#             return self._create_fallback_weather_conditions()
    
#     def _validate_weather_conditions(self, weather_conditions: pd.Series) -> bool:
#         """Validate individual weather condition record"""
#         try:
#             # Check track temperature
#             if 'TRACK_TEMP' in weather_conditions:
#                 track_temp = weather_conditions['TRACK_TEMP']
#                 if track_temp < 10 or track_temp > 80:
#                     return False
            
#             # Check air temperature
#             if 'AIR_TEMP' in weather_conditions:
#                 air_temp = weather_conditions['AIR_TEMP']
#                 if air_temp < -10 or air_temp > 50:
#                     return False
                    
#             return True
#         except:
#             return False

#     def _create_fallback_weather_conditions(self) -> pd.Series:
#         """Create fallback weather conditions when real data is invalid"""
#         return pd.Series({
#             'AIR_TEMP': 25.0,
#             'TRACK_TEMP': 32.0,
#             'HUMIDITY': 50.0,
#             'PRESSURE': 1013.0,
#             'WIND_SPEED': 2.0,
#             'WIND_DIRECTION': 0.0,
#             'RAIN': 0.0,
#             'timestamp': datetime.now()
#         })

#     def _get_lap_telemetry(self, lap: pd.Series, telemetry_data: pd.DataFrame, car_number: int) -> Dict[str, float]:
#         """Get telemetry data for a specific lap with enhanced vehicle matching"""
#         if telemetry_data.empty:
#             return self._get_fallback_telemetry()
            
#         try:
#             # Enhanced vehicle matching with multiple strategies
#             vehicle_id_pattern = f"GR86-{str(car_number).zfill(3)}-000"
            
#             lap_telemetry = telemetry_data[
#                 (telemetry_data['vehicle_id'] == vehicle_id_pattern) &
#                 (telemetry_data['lap'] == lap['LAP_NUMBER'])
#             ]
            
#             if lap_telemetry.empty:
#                 # Fallback: try partial matching
#                 lap_telemetry = telemetry_data[
#                     (telemetry_data['vehicle_id'].str.contains(str(car_number))) &
#                     (telemetry_data['lap'] == lap['LAP_NUMBER'])
#                 ]
            
#             if lap_telemetry.empty:
#                 return self._get_fallback_telemetry()
                
#             return {
#                 'avg_speed': lap_telemetry['speed'].mean() if 'speed' in lap_telemetry.columns else 120.0,
#                 'avg_long_accel': lap_telemetry['accx_can'].abs().mean() if 'accx_can' in lap_telemetry.columns else 0.3,
#                 'avg_lat_accel': lap_telemetry['accy_can'].abs().mean() if 'accy_can' in lap_telemetry.columns else 0.4,
#                 'avg_gear': lap_telemetry['gear'].mean() if 'gear' in lap_telemetry.columns else 3.0
#             }
            
#         except Exception as e:
#             self.logger.debug(f"⚠️ Telemetry lookup failed: {e}")
#             return self._get_fallback_telemetry()

#     def _get_fallback_telemetry(self) -> Dict[str, float]:
#         """Fallback telemetry data"""
#         return {
#             'avg_speed': 120.0,
#             'avg_long_accel': 0.3,
#             'avg_lat_accel': 0.4,
#             'avg_gear': 3.0
#         }
    
#     def _calculate_baseline_performance(self, car_laps: pd.DataFrame) -> float:
#         """Calculate baseline performance with enhanced robustness"""
#         try:
#             # Convert lap times to seconds
#             lap_times_seconds = car_laps['LAP_TIME'].apply(self._lap_time_to_seconds).dropna()
            
#             if len(lap_times_seconds) < 3:
#                 return lap_times_seconds.mean() if not lap_times_seconds.empty else 90.0
                
#             # Use best 40% of laps as baseline
#             n_best = max(2, int(len(lap_times_seconds) * 0.4))
#             best_laps = lap_times_seconds.nsmallest(n_best)
#             return best_laps.median()
            
#         except Exception as e:
#             self.logger.debug(f"⚠️ Baseline calculation failed: {e}")
#             return 90.0
    
#     def _calculate_weather_impact(self, lap: pd.Series, baseline_time: float, weather: pd.Series) -> float:
#         """Calculate weather impact with enhanced normalization"""
#         try:
#             actual_time = self._lap_time_to_seconds(lap['LAP_TIME'])
#             impact = actual_time - baseline_time
            
#             # Enhanced impact normalization
#             normal_variation = max(0.5, baseline_time * 0.05)
#             adjusted_impact = impact if abs(impact) > normal_variation else 0
            
#             # Cap impact to reasonable range
#             return max(-8.0, min(8.0, adjusted_impact))
            
#         except Exception:
#             return 0.0
    
#     def _create_weather_feature_vector(self, lap: pd.Series, weather: pd.Series, telemetry: Dict[str, float], track_name: str) -> Dict[str, float]:
#         """Create weather feature vector with enhanced derived features and validation"""
#         try:
#             # Basic weather conditions with validation
#             air_temp = weather.get('AIR_TEMP', 25.0)
#             track_temp = weather.get('TRACK_TEMP', 30.0)
#             humidity = weather.get('HUMIDITY', 50.0)
#             pressure = weather.get('PRESSURE', 1013.0)
#             wind_speed = weather.get('WIND_SPEED', 0.0)
#             wind_direction = weather.get('WIND_DIRECTION', 0.0)
#             rain = weather.get('RAIN', 0.0)
            
#             # Enhanced derived features
#             temp_difference = track_temp - air_temp
#             air_density = self._calculate_air_density(air_temp, pressure, humidity)
#             wind_effect = self._calculate_wind_effect(wind_speed, wind_direction)
            
#             # Track and context features
#             track_sensitivity = self._get_track_weather_sensitivity(track_name)
#             lap_number = lap.get('LAP_NUMBER', 1)
#             time_of_day = lap['timestamp'].hour + lap['timestamp'].minute / 60.0 if 'timestamp' in lap else 14.0
            
#             # Enhanced telemetry features
#             avg_speed = telemetry.get('avg_speed', 120.0)
#             driving_aggressiveness = (telemetry.get('avg_long_accel', 0.3) + telemetry.get('avg_lat_accel', 0.4)) / 2
#             gear_usage = telemetry.get('avg_gear', 3.0)
            
#             features = {
#                 # Basic weather
#                 'air_temp': air_temp,
#                 'track_temp': track_temp,
#                 'humidity': humidity,
#                 'pressure': pressure,
#                 'wind_speed': wind_speed,
#                 'wind_direction': wind_direction,
#                 'rain': rain,
                
#                 # Derived weather features
#                 'temp_difference': temp_difference,
#                 'air_density': air_density,
#                 'wind_effect': wind_effect,
#                 'heat_index': self._calculate_heat_index(air_temp, humidity),
#                 'dew_point': self._calculate_dew_point(air_temp, humidity),
                
#                 # Track and context
#                 'track_weather_sensitivity': track_sensitivity,
#                 'lap_number': lap_number,
#                 'time_of_day': time_of_day,
#                 'is_afternoon': 1.0 if 12 <= time_of_day <= 18 else 0.0,
                
#                 # Telemetry-based
#                 'avg_speed': avg_speed,
#                 'driving_aggressiveness': driving_aggressiveness,
#                 'gear_usage': gear_usage,
#                 'speed_normalized': avg_speed / 200.0
#             }
            
#             return features
            
#         except Exception as e:
#             self.logger.warning(f"⚠️ Weather feature vector creation failed: {e}")
#             return self._get_fallback_weather_features(track_name)

#     def _get_fallback_weather_features(self, track_name: str) -> Dict[str, float]:
#         """Fallback weather features"""
#         return {
#             'air_temp': 25.0, 'track_temp': 32.0, 'humidity': 50.0, 'pressure': 1013.0,
#             'wind_speed': 2.0, 'wind_direction': 0.0, 'rain': 0.0, 'temp_difference': 7.0,
#             'air_density': 1.2, 'wind_effect': 0.0, 'heat_index': 26.0, 'dew_point': 15.0,
#             'track_weather_sensitivity': self._get_track_weather_sensitivity(track_name),
#             'lap_number': 5.0, 'time_of_day': 14.0, 'is_afternoon': 1.0,
#             'avg_speed': 120.0, 'driving_aggressiveness': 0.35, 'gear_usage': 3.0, 'speed_normalized': 0.6
#         }
    
#     def _lap_time_to_seconds(self, lap_time: str) -> float:
#         """Convert lap time string to seconds with robust parsing"""
#         try:
#             if pd.isna(lap_time) or lap_time == 0:
#                 return 90.0
                
#             time_str = str(lap_time).strip()
#             parts = time_str.split(':')
            
#             if len(parts) == 3:
#                 hours, minutes, seconds = parts
#                 return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
#             elif len(parts) == 2:
#                 minutes, seconds = parts
#                 return float(minutes) * 60 + float(seconds)
#             else:
#                 return float(time_str)
#         except:
#             return 90.0
    
#     def _calculate_air_density(self, air_temp: float, pressure: float, humidity: float) -> float:
#         """Calculate air density with enhanced formula"""
#         try:
#             R = 287.05
#             temp_kelvin = air_temp + 273.15
            
#             vapor_pressure = 0.611 * np.exp(17.27 * air_temp / (air_temp + 237.3)) * (humidity / 100)
#             dry_air_pressure = pressure - vapor_pressure
            
#             density = (dry_air_pressure * 100) / (R * temp_kelvin)
#             return max(1.0, min(1.3, density))
            
#         except:
#             return 1.2

#     def _calculate_heat_index(self, temp: float, humidity: float) -> float:
#         """Calculate heat index (feels-like temperature)"""
#         try:
#             if temp < 27.0:
#                 return temp
#             return temp + 0.1 * (humidity - 50) * 0.5
#         except:
#             return temp

#     def _calculate_dew_point(self, temp: float, humidity: float) -> float:
#         """Calculate dew point temperature"""
#         try:
#             alpha = 17.27
#             beta = 237.7
#             gamma = (alpha * temp) / (beta + temp) + np.log(humidity / 100.0)
#             return (beta * gamma) / (alpha - gamma)
#         except:
#             return temp - 5.0
    
#     def _calculate_wind_effect(self, wind_speed: float, wind_direction: float) -> float:
#         """Calculate wind effect with directional component"""
#         base_effect = wind_speed * 0.08
#         directional_factor = abs(np.sin(np.radians(wind_direction)))
#         return base_effect * (1 + directional_factor * 0.5)
    
#     def _get_track_weather_sensitivity(self, track_name: str) -> float:
#         """Get track-specific weather sensitivity with enhanced mapping"""
#         sensitivity_map = {
#             'barber': 0.8, 'cota': 0.65, 'indianapolis': 0.5,
#             'road_america': 0.9, 'sebring': 0.85, 'sonoma': 0.75, 'virginia': 0.7
#         }
#         return sensitivity_map.get(track_name.lower(), 0.7)

#     def _train_with_fallback(self, reason: str, processed_tracks: List[str]) -> dict:
#         """Create fallback model with enhanced synthetic weather data"""
#         self.logger.warning(f"⚠️ Using fallback weather model: {reason}")
        
#         synthetic_features, synthetic_targets = self._generate_enhanced_synthetic_weather_data()
        
#         if len(synthetic_features) > 0:
#             X_synth = pd.DataFrame(synthetic_features)
#             y_synth = np.array(synthetic_targets)
            
#             X_scaled = self.scaler.fit_transform(X_synth)
#             self.feature_columns = X_synth.columns.tolist()
#             self.model.fit(X_scaled, y_synth)
            
#             self.logger.info("✅ Fallback weather model trained with synthetic data")
            
#             return {
#                 'model': self, 
#                 'features': self.feature_columns, 
#                 'train_score': 0.5, 
#                 'test_score': 0.4,
#                 'feature_importance': {col: 1.0/len(self.feature_columns) for col in self.feature_columns},
#                 'training_samples': len(X_synth), 
#                 'tracks_used': len(processed_tracks),
#                 'processed_tracks': processed_tracks,
#                 'status': 'fallback', 
#                 'fallback_reason': reason
#             }
        
#         return {
#             'error': f'Weather model training failed: {reason}', 
#             'status': 'error',
#             'tracks_used': len(processed_tracks),
#             'processed_tracks': processed_tracks
#         }

#     def _generate_enhanced_synthetic_weather_data(self, n_samples: int = 200) -> Tuple[List[Dict], List[float]]:
#         """Generate enhanced synthetic weather training data with realistic scenarios"""
#         features = []
#         targets = []
        
#         weather_scenarios = ['hot_dry', 'cool_humid', 'rainy', 'ideal', 'variable']
        
#         for i in range(n_samples):
#             scenario = np.random.choice(weather_scenarios)
            
#             if scenario == 'hot_dry':
#                 air_temp = 35 + np.random.normal(0, 3)
#                 track_temp = air_temp + 10 + np.random.normal(0, 2)  # Realistic track temp
#                 humidity = 30 + np.random.normal(0, 10)
#                 impact = 1.5 + np.random.normal(0, 0.5)
#             elif scenario == 'cool_humid':
#                 air_temp = 18 + np.random.normal(0, 3)
#                 track_temp = air_temp + 8 + np.random.normal(0, 2)   # Realistic track temp
#                 humidity = 80 + np.random.normal(0, 10)
#                 impact = 0.8 + np.random.normal(0, 0.3)
#             elif scenario == 'rainy':
#                 air_temp = 15 + np.random.normal(0, 5)
#                 track_temp = air_temp + 5 + np.random.normal(0, 2)   # Cooler track in rain
#                 humidity = 95 + np.random.normal(0, 3)
#                 impact = 3.0 + np.random.normal(0, 1.0)
#             elif scenario == 'variable':
#                 air_temp = 22 + np.random.normal(0, 8)
#                 track_temp = air_temp + np.random.uniform(5, 15)     # Variable track temp
#                 humidity = 60 + np.random.normal(0, 20)
#                 impact = 1.2 + np.random.normal(0, 0.8)
#             else:  # ideal
#                 air_temp = 22 + np.random.normal(0, 2)
#                 track_temp = air_temp + 10 + np.random.normal(0, 1)  # Ideal track temp
#                 humidity = 50 + np.random.normal(0, 10)
#                 impact = 0.2 + np.random.normal(0, 0.2)
            
#             feat = self._get_fallback_weather_features('generic')
#             # Update with scenario-specific values
#             feat.update({
#                 'air_temp': max(10, min(40, air_temp)),
#                 'track_temp': max(15, min(50, track_temp)),  # Ensured realistic track temps
#                 'humidity': max(20, min(100, humidity)),
#                 'rain': 1.0 if scenario == 'rainy' else 0.0,
#                 'wind_speed': np.random.uniform(0, 8),
#                 'wind_direction': np.random.uniform(0, 360)
#             })
            
#             # Recalculate derived features
#             feat['temp_difference'] = feat['track_temp'] - feat['air_temp']
#             feat['air_density'] = self._calculate_air_density(feat['air_temp'], feat['pressure'], feat['humidity'])
#             feat['heat_index'] = self._calculate_heat_index(feat['air_temp'], feat['humidity'])
#             feat['dew_point'] = self._calculate_dew_point(feat['air_temp'], feat['humidity'])
            
#             features.append(feat)
#             targets.append(impact)
            
#         return features, targets

#     def predict_weather_impact(self, weather_conditions: Dict[str, float], track_name: str, lap_context: Dict[str, float]) -> float:
#         """Predict weather impact on lap performance with comprehensive validation"""
#         try:
#             if not self.feature_columns:
#                 self.logger.warning("⚠️ No trained model available, using fallback")
#                 return self._fallback_weather_impact(weather_conditions, track_name)
                
#             # Validate input weather conditions
#             validated_weather = self._validate_and_repair_weather_input(weather_conditions)
                
#             # Create feature vector
#             feature_vector = self._create_weather_feature_vector(
#                 lap_context.get('lap_info', {}),
#                 validated_weather,
#                 lap_context.get('telemetry', {}),
#                 track_name
#             )
            
#             # Ensure all features are present
#             feature_array = np.array([feature_vector.get(col, 0.0) for col in self.feature_columns]).reshape(1, -1)
#             scaled_features = self.scaler.transform(feature_array)
            
#             impact = self.model.predict(scaled_features)[0]
#             return max(-8.0, min(8.0, impact))
            
#         except Exception as e:
#             self.logger.error(f"❌ Weather impact prediction failed: {e}")
#             return self._fallback_weather_impact(weather_conditions, track_name)
    
#     def _validate_and_repair_weather_input(self, weather_conditions: Dict[str, float]) -> Dict[str, float]:
#         """Validate and repair input weather conditions"""
#         repaired = weather_conditions.copy()
        
#         # Fix TRACK_TEMP if invalid
#         if 'TRACK_TEMP' in repaired:
#             track_temp = repaired['TRACK_TEMP']
#             if track_temp < 10 or track_temp > 80:
#                 air_temp = repaired.get('AIR_TEMP', 25.0)
#                 repaired['TRACK_TEMP'] = air_temp + 10  # Realistic default
        
#         # Ensure AIR_TEMP is reasonable
#         if 'AIR_TEMP' in repaired:
#             air_temp = repaired['AIR_TEMP']
#             if air_temp < -10 or air_temp > 50:
#                 repaired['AIR_TEMP'] = 25.0  # Safe default
        
#         return repaired
    
#     def _fallback_weather_impact(self, weather_conditions: Dict[str, float], track_name: str) -> float:
#         """Enhanced fallback weather impact calculation"""
#         base_impact = 0.0
        
#         # Temperature effect
#         ideal_temp = 22.0
#         temp_diff = abs(weather_conditions.get('AIR_TEMP', 25) - ideal_temp)
#         base_impact += temp_diff * 0.08
        
#         # Rain effect
#         rain = weather_conditions.get('RAIN', 0)
#         if rain > 0:
#             base_impact += 2.5 + (rain * 0.5)
            
#         # Humidity effect
#         humidity = weather_conditions.get('HUMIDITY', 50)
#         humidity_effect = abs(humidity - 50) * 0.02
#         base_impact += humidity_effect
        
#         # Wind effect
#         wind_speed = weather_conditions.get('WIND_SPEED', 0)
#         base_impact += wind_speed * 0.1
        
#         # Track sensitivity
#         sensitivity = self._get_track_weather_sensitivity(track_name)
        
#         return base_impact * sensitivity
    
#     def get_optimal_conditions(self, track_name: str) -> Dict[str, float]:
#         """Get optimal weather conditions for a track"""
#         return {
#             'AIR_TEMP': 22.0,
#             'TRACK_TEMP': 32.0,  # Realistic optimal track temp
#             'HUMIDITY': 50.0,
#             'PRESSURE': 1013.0,
#             'WIND_SPEED': 2.0,
#             'RAIN': 0.0
#         }
    
#     def estimate_tire_temperature(self, weather_conditions: Dict[str, float], track_name: str, lap_count: int) -> float:
#         """Estimate tire temperature based on weather and usage"""
#         try:
#             base_temp = weather_conditions.get('TRACK_TEMP', 30.0)
#             air_temp = weather_conditions.get('AIR_TEMP', 25.0)
            
#             # Heat from usage
#             usage_heat = min(20.0, lap_count * 0.6)
            
#             # Track-specific heating
#             track_heat = self._get_track_weather_sensitivity(track_name) * 6.0
            
#             estimated_temp = base_temp + usage_heat + track_heat
            
#             # Cooling effect
#             if air_temp < base_temp:
#                 cooling_effect = (base_temp - air_temp) * 0.15
#                 estimated_temp -= cooling_effect
                
#             return max(air_temp, min(110.0, estimated_temp))
            
#         except Exception as e:
#             self.logger.error(f"❌ Tire temperature estimation failed: {e}")
#             return weather_conditions.get('TRACK_TEMP', 30.0)
    
#     def save_model(self, filepath: str):
#         """Save trained model to file"""
#         model_data = {
#             'model': self.model,
#             'scaler': self.scaler,
#             'feature_columns': self.feature_columns
#         }
#         joblib.dump(model_data, filepath)
    
#     def load_model(self, filepath: str):
#         """Load trained model from file"""
#         model_data = joblib.load(filepath)
#         self.model = model_data['model']
#         self.scaler = model_data['scaler']
#         self.feature_columns = model_data['feature_columns']
























import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

class WeatherModelTrainer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.logger = logging.getLogger(__name__)
        
        # Minimum required columns for flexibility
        self.minimum_weather_cols = ['AIR_TEMP', 'TRACK_TEMP']
        self.minimum_pit_cols = ['NUMBER', 'LAP_NUMBER', 'LAP_TIME']

    def train(self, track_data: Dict[str, Dict[str, pd.DataFrame]]) -> dict:
        """Train weather impact model with enhanced error handling and fallback data"""
        try:
            self.logger.info("🚀 Starting weather impact model training...")
            
            features_list = []
            impact_list = []
            processed_tracks = []
            
            # Process each track's data with enhanced validation
            for track_name, data_dict in track_data.items():
                self.logger.info(f"📊 Processing track: {track_name}")
                
                if not self._validate_track_data(data_dict):
                    self.logger.warning(f"⚠️ Skipping {track_name}: validation failed")
                    continue
                    
                session_features, session_impacts = self._extract_weather_features(data_dict, track_name)
                
                if not session_features.empty and len(session_impacts) > 0:
                    features_list.append(session_features)
                    impact_list.extend(session_impacts)
                    processed_tracks.append(track_name)
                    self.logger.info(f"✅ {track_name}: extracted {len(session_features)} weather samples")
                else:
                    self.logger.warning(f"⚠️ No weather features extracted from {track_name}")
            
            if not features_list:
                return self._train_with_fallback("No valid weather features extracted from any track", processed_tracks)
            
            # Combine all track data
            X = pd.concat(features_list, ignore_index=True)
            y = np.array(impact_list)
            
            if X.empty or len(y) == 0:
                return self._train_with_fallback("Empty feature or target matrices after processing", processed_tracks)
            
            self.logger.info(f"📈 Final dataset: {len(X)} samples, {len(X.columns)} features")

            # Remove NaN values
            valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            if len(X_clean) < 15:  # Reduced threshold
                return self._train_with_fallback(f"Insufficient training samples: {len(X_clean)}", processed_tracks)
            
            # Scale features and train
            X_scaled = self.scaler.fit_transform(X_clean)
            self.feature_columns = X_clean.columns.tolist()
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_clean, test_size=0.2, random_state=42
            )
            
            self.logger.info("🏃 Training model...")
            self.model.fit(X_train, y_train)
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))

            self.logger.info(f"✅ Weather impact model trained successfully - Test Score: {test_score:.3f}")

            return {
                'model': self,
                'features': self.feature_columns,
                'train_score': train_score,
                'test_score': test_score,
                'feature_importance': feature_importance,
                'training_samples': len(X_clean),
                'tracks_used': len(processed_tracks),
                'processed_tracks': processed_tracks,
                'status': 'success'
            }
            
        except Exception as e:
            error_msg = f'Training failed: {str(e)}'
            self.logger.error(f"❌ {error_msg}")
            return self._train_with_fallback(error_msg, [])

    def _validate_track_data(self, data_dict: Dict[str, pd.DataFrame]) -> bool:
        """Enhanced validation with better logging and flexibility"""
        weather_data = data_dict.get('weather_data', pd.DataFrame())
        pit_data = data_dict.get('pit_data', pd.DataFrame())
        
        if weather_data.empty or pit_data.empty:
            self.logger.debug("⚠️ Missing weather_data or pit_data")
            return False
            
        # Check for minimum required columns
        missing_weather = [col for col in self.minimum_weather_cols if col not in weather_data.columns]
        missing_pit = [col for col in self.minimum_pit_cols if col not in pit_data.columns]
        
        if missing_weather or missing_pit:
            self.logger.debug(f"⚠️ Missing required columns - weather: {missing_weather}, pit: {missing_pit}")
            return False
            
        # Check for minimum data volume
        if len(weather_data) < 3 or len(pit_data) < 5:
            self.logger.debug(f"⚠️ Insufficient data - weather: {len(weather_data)}, pit: {len(pit_data)}")
            return False
            
        self.logger.debug(f"✅ Track data validated: {len(weather_data)} weather rows, {len(pit_data)} pit rows")
        return True

    def _extract_weather_features(self, data_dict: Dict[str, pd.DataFrame], track_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Extract weather impact features with enhanced timestamp handling and fallbacks"""
        weather_data = data_dict.get('weather_data', pd.DataFrame())
        pit_data = data_dict.get('pit_data', pd.DataFrame())
        telemetry_data = data_dict.get('telemetry_data', pd.DataFrame())
        
        features = []
        impacts = []
        
        if pit_data.empty or weather_data.empty:
            self.logger.debug("⚠️ Missing pit_data or weather_data")
            return pd.DataFrame(), np.array([])
        
        # Prepare timestamps with enhanced error handling
        weather_data_clean = self._prepare_weather_timestamps(weather_data)
        pit_data_clean = self._prepare_pit_timestamps(pit_data)
        
        if weather_data_clean.empty or pit_data_clean.empty:
            self.logger.debug("⚠️ Failed to prepare timestamps")
            return pd.DataFrame(), np.array([])
        
        self.logger.debug(f"🔧 Processing {len(pit_data_clean['NUMBER'].unique())} cars in {track_name}")

        # Process each car's laps with enhanced baseline calculation
        for car_number in pit_data_clean['NUMBER'].unique():
            car_laps = pit_data_clean[pit_data_clean['NUMBER'] == car_number].sort_values('LAP_NUMBER')
            if len(car_laps) < 3:  # Reduced minimum laps
                self.logger.debug(f"⚠️ Car {car_number}: insufficient laps ({len(car_laps)})")
                continue
                
            baseline_time = self._calculate_baseline_performance(car_laps)
            
            for _, lap in car_laps.iterrows():
                lap_weather = self._get_lap_weather_conditions(lap, weather_data_clean)
                if lap_weather is None:
                    continue
                    
                lap_telemetry = self._get_lap_telemetry(lap, telemetry_data, car_number)
                weather_impact = self._calculate_weather_impact(lap, baseline_time, lap_weather)
                feature_vector = self._create_weather_feature_vector(lap, lap_weather, lap_telemetry, track_name)
                
                features.append(pd.DataFrame([feature_vector]))
                impacts.append(weather_impact)
        
        if features:
            result_features = pd.concat(features, ignore_index=True)
            self.logger.debug(f"✅ {track_name}: extracted {len(result_features)} weather impact samples")
            return result_features, np.array(impacts)
        
        self.logger.debug(f"❌ {track_name}: no weather features extracted")
        return pd.DataFrame(), np.array([])
    
    def _prepare_weather_timestamps(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare weather timestamps with enhanced fallbacks"""
        weather_data = weather_data.copy()
        
        try:
            if 'TIME_UTC_SECONDS' in weather_data.columns:
                weather_data['timestamp'] = pd.to_datetime(weather_data['TIME_UTC_SECONDS'], unit='s', errors='coerce')
            elif 'TIME_UTC_STR' in weather_data.columns:
                weather_data['timestamp'] = pd.to_datetime(weather_data['TIME_UTC_STR'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
            else:
                # Enhanced fallback: use index-based timestamps
                start_time = datetime.now() - timedelta(hours=2)
                weather_data['timestamp'] = [start_time + timedelta(seconds=i*60) for i in range(len(weather_data))]
            
            # Filter out invalid timestamps
            weather_data = weather_data.dropna(subset=['timestamp'])
            return weather_data.sort_values('timestamp')
            
        except Exception as e:
            self.logger.warning(f"⚠️ Weather timestamp preparation failed: {e}")
            return pd.DataFrame()

    def _prepare_pit_timestamps(self, pit_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare pit data timestamps with enhanced fallbacks"""
        pit_data = pit_data.copy()
        
        try:
            if 'HOUR' in pit_data.columns:
                # Enhanced HOUR parsing with multiple format attempts
                base_date = datetime.now().date()
                pit_data['timestamp'] = pd.to_datetime(
                    base_date.strftime('%Y-%m-%d') + ' ' + pit_data['HOUR'].astype(str),
                    errors='coerce'
                )
                
                # Fallback for failed parses
                if pit_data['timestamp'].isna().any():
                    pit_data['timestamp'] = pd.to_datetime(pit_data['HOUR'], errors='coerce')
                    
            elif 'ELAPSED' in pit_data.columns:
                base_time = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)
                pit_data['timestamp'] = pit_data['ELAPSED'].apply(
                    lambda x: base_time + timedelta(seconds=float(x)) if pd.notna(x) and str(x).strip() else pd.NaT
                )
            else:
                # Enhanced synthetic timestamps
                base_time = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)
                pit_data['timestamp'] = [base_time + timedelta(seconds=i*120) for i in range(len(pit_data))]
            
            # Filter out invalid timestamps
            pit_data = pit_data.dropna(subset=['timestamp'])
            return pit_data.sort_values('timestamp')
            
        except Exception as e:
            self.logger.warning(f"⚠️ Pit timestamp preparation failed: {e}")
            return pd.DataFrame()
    
    def _get_lap_weather_conditions(self, lap: pd.Series, weather_data: pd.DataFrame) -> pd.Series | None:
        """Get weather conditions for a specific lap with enhanced matching"""
        if weather_data.empty:
            return None
            
        try:
            lap_time = lap['timestamp']
            time_diff = (weather_data['timestamp'] - lap_time).abs()
            closest_idx = time_diff.idxmin()
            
            # Increased time tolerance for more matches
            if time_diff[closest_idx] > timedelta(minutes=10):  # Increased from 5 to 10 minutes
                return None
                
            return weather_data.loc[closest_idx]
            
        except Exception as e:
            self.logger.debug(f"⚠️ Weather condition lookup failed: {e}")
            return None
    
    def _get_lap_telemetry(self, lap: pd.Series, telemetry_data: pd.DataFrame, car_number: int) -> Dict[str, float]:
        """Get telemetry data for a specific lap with enhanced vehicle matching"""
        if telemetry_data.empty:
            return self._get_fallback_telemetry()
            
        try:
            # Enhanced vehicle matching with multiple strategies
            vehicle_id_pattern = f"GR86-{str(car_number).zfill(3)}-000"
            
            lap_telemetry = telemetry_data[
                (telemetry_data['vehicle_id'] == vehicle_id_pattern) &
                (telemetry_data['lap'] == lap['LAP_NUMBER'])
            ]
            
            if lap_telemetry.empty:
                # Fallback: try partial matching
                lap_telemetry = telemetry_data[
                    (telemetry_data['vehicle_id'].str.contains(str(car_number))) &
                    (telemetry_data['lap'] == lap['LAP_NUMBER'])
                ]
            
            if lap_telemetry.empty:
                return self._get_fallback_telemetry()
                
            return {
                'avg_speed': lap_telemetry['speed'].mean() if 'speed' in lap_telemetry.columns else 120.0,
                'avg_long_accel': lap_telemetry['accx_can'].abs().mean() if 'accx_can' in lap_telemetry.columns else 0.3,
                'avg_lat_accel': lap_telemetry['accy_can'].abs().mean() if 'accy_can' in lap_telemetry.columns else 0.4,
                'avg_gear': lap_telemetry['gear'].mean() if 'gear' in lap_telemetry.columns else 3.0
            }
            
        except Exception as e:
            self.logger.debug(f"⚠️ Telemetry lookup failed: {e}")
            return self._get_fallback_telemetry()

    def _get_fallback_telemetry(self) -> Dict[str, float]:
        """Fallback telemetry data"""
        return {
            'avg_speed': 120.0,
            'avg_long_accel': 0.3,
            'avg_lat_accel': 0.4,
            'avg_gear': 3.0
        }
    
    def _calculate_baseline_performance(self, car_laps: pd.DataFrame) -> float:
        """Calculate baseline performance with enhanced robustness"""
        try:
            # Convert lap times to seconds
            lap_times_seconds = car_laps['LAP_TIME'].apply(self._lap_time_to_seconds).dropna()
            
            if len(lap_times_seconds) < 3:
                return lap_times_seconds.mean() if not lap_times_seconds.empty else 90.0
                
            # Use best 40% of laps as baseline (increased from 30%)
            n_best = max(2, int(len(lap_times_seconds) * 0.4))
            best_laps = lap_times_seconds.nsmallest(n_best)
            return best_laps.median()
            
        except Exception as e:
            self.logger.debug(f"⚠️ Baseline calculation failed: {e}")
            return 90.0
    
    def _calculate_weather_impact(self, lap: pd.Series, baseline_time: float, weather: pd.Series) -> float:
        """Calculate weather impact with enhanced normalization"""
        try:
            actual_time = self._lap_time_to_seconds(lap['LAP_TIME'])
            impact = actual_time - baseline_time
            
            # Enhanced impact normalization
            normal_variation = max(0.5, baseline_time * 0.05)  # Dynamic variation based on baseline
            adjusted_impact = impact if abs(impact) > normal_variation else 0
            
            # Cap impact to reasonable range
            return max(-8.0, min(8.0, adjusted_impact))
            
        except Exception:
            return 0.0
    
    def _create_weather_feature_vector(self, lap: pd.Series, weather: pd.Series, telemetry: Dict[str, float], track_name: str) -> Dict[str, float]:
        """Create weather feature vector with enhanced derived features"""
        try:
            # Basic weather conditions
            air_temp = weather.get('AIR_TEMP', 25.0)
            track_temp = weather.get('TRACK_TEMP', 30.0)
            humidity = weather.get('HUMIDITY', 50.0)
            pressure = weather.get('PRESSURE', 1013.0)
            wind_speed = weather.get('WIND_SPEED', 0.0)
            wind_direction = weather.get('WIND_DIRECTION', 0.0)
            rain = weather.get('RAIN', 0.0)
            
            # Enhanced derived features
            temp_difference = track_temp - air_temp
            air_density = self._calculate_air_density(air_temp, pressure, humidity)
            wind_effect = self._calculate_wind_effect(wind_speed, wind_direction)
            
            # Track and context features
            track_sensitivity = self._get_track_weather_sensitivity(track_name)
            lap_number = lap.get('LAP_NUMBER', 1)
            time_of_day = lap['timestamp'].hour + lap['timestamp'].minute / 60.0 if 'timestamp' in lap else 14.0
            
            # Enhanced telemetry features
            avg_speed = telemetry.get('avg_speed', 120.0)
            driving_aggressiveness = (telemetry.get('avg_long_accel', 0.3) + telemetry.get('avg_lat_accel', 0.4)) / 2
            gear_usage = telemetry.get('avg_gear', 3.0)
            
            features = {
                # Basic weather
                'air_temp': air_temp,
                'track_temp': track_temp,
                'humidity': humidity,
                'pressure': pressure,
                'wind_speed': wind_speed,
                'wind_direction': wind_direction,
                'rain': rain,
                
                # Derived weather features
                'temp_difference': temp_difference,
                'air_density': air_density,
                'wind_effect': wind_effect,
                'heat_index': self._calculate_heat_index(air_temp, humidity),
                'dew_point': self._calculate_dew_point(air_temp, humidity),
                
                # Track and context
                'track_weather_sensitivity': track_sensitivity,
                'lap_number': lap_number,
                'time_of_day': time_of_day,
                'is_afternoon': 1.0 if 12 <= time_of_day <= 18 else 0.0,
                
                # Telemetry-based
                'avg_speed': avg_speed,
                'driving_aggressiveness': driving_aggressiveness,
                'gear_usage': gear_usage,
                'speed_normalized': avg_speed / 200.0  # Normalized speed feature
            }
            
            return features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Weather feature vector creation failed: {e}")
            return self._get_fallback_weather_features(track_name)

    def _get_fallback_weather_features(self, track_name: str) -> Dict[str, float]:
        """Fallback weather features"""
        return {
            'air_temp': 25.0, 'track_temp': 30.0, 'humidity': 50.0, 'pressure': 1013.0,
            'wind_speed': 0.0, 'wind_direction': 0.0, 'rain': 0.0, 'temp_difference': 5.0,
            'air_density': 1.2, 'wind_effect': 0.0, 'heat_index': 26.0, 'dew_point': 15.0,
            'track_weather_sensitivity': self._get_track_weather_sensitivity(track_name),
            'lap_number': 5.0, 'time_of_day': 14.0, 'is_afternoon': 1.0,
            'avg_speed': 120.0, 'driving_aggressiveness': 0.35, 'gear_usage': 3.0, 'speed_normalized': 0.6
        }
    
    def _lap_time_to_seconds(self, lap_time: str) -> float:
        """Convert lap time string to seconds with robust parsing"""
        try:
            if pd.isna(lap_time) or lap_time == 0:
                return 90.0
                
            time_str = str(lap_time).strip()
            parts = time_str.split(':')
            
            if len(parts) == 3:
                hours, minutes, seconds = parts
                return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
            elif len(parts) == 2:
                minutes, seconds = parts
                return float(minutes) * 60 + float(seconds)
            else:
                return float(time_str)
        except:
            return 90.0
    
    def _calculate_air_density(self, air_temp: float, pressure: float, humidity: float) -> float:
        """Calculate air density with enhanced formula"""
        try:
            R = 287.05  # Specific gas constant for dry air (J/kg·K)
            temp_kelvin = air_temp + 273.15
            
            # Enhanced vapor pressure calculation
            vapor_pressure = 0.611 * np.exp(17.27 * air_temp / (air_temp + 237.3)) * (humidity / 100)
            dry_air_pressure = pressure - vapor_pressure
            
            density = (dry_air_pressure * 100) / (R * temp_kelvin)
            return max(1.0, min(1.3, density))  # Reasonable bounds
            
        except:
            return 1.2

    def _calculate_heat_index(self, temp: float, humidity: float) -> float:
        """Calculate heat index (feels-like temperature)"""
        try:
            # Simplified heat index formula
            if temp < 27.0:
                return temp
            return temp + 0.1 * (humidity - 50) * 0.5
        except:
            return temp

    def _calculate_dew_point(self, temp: float, humidity: float) -> float:
        """Calculate dew point temperature"""
        try:
            # Magnus formula approximation
            alpha = 17.27
            beta = 237.7
            gamma = (alpha * temp) / (beta + temp) + np.log(humidity / 100.0)
            return (beta * gamma) / (alpha - gamma)
        except:
            return temp - 5.0  # Conservative estimate
    
    def _calculate_wind_effect(self, wind_speed: float, wind_direction: float) -> float:
        """Calculate wind effect with directional component"""
        # Enhanced wind effect model considering direction
        base_effect = wind_speed * 0.08  # Reduced from 0.1
        directional_factor = abs(np.sin(np.radians(wind_direction)))  # Cross-wind effect
        return base_effect * (1 + directional_factor * 0.5)
    
    def _get_track_weather_sensitivity(self, track_name: str) -> float:
        """Get track-specific weather sensitivity with enhanced mapping"""
        sensitivity_map = {
            'barber': 0.8, 'cota': 0.65, 'indianapolis': 0.5,
            'road_america': 0.9, 'sebring': 0.85, 'sonoma': 0.75, 'virginia': 0.7
        }
        return sensitivity_map.get(track_name.lower(), 0.7)

    def _train_with_fallback(self, reason: str, processed_tracks: List[str]) -> dict:
        """Create fallback model with enhanced synthetic weather data"""
        self.logger.warning(f"⚠️ Using fallback weather model: {reason}")
        
        synthetic_features, synthetic_targets = self._generate_enhanced_synthetic_weather_data()
        
        if len(synthetic_features) > 0:
            X_synth = pd.DataFrame(synthetic_features)
            y_synth = np.array(synthetic_targets)
            
            X_scaled = self.scaler.fit_transform(X_synth)
            self.feature_columns = X_synth.columns.tolist()
            self.model.fit(X_scaled, y_synth)
            
            self.logger.info("✅ Fallback weather model trained with synthetic data")
            
            return {
                'model': self, 
                'features': self.feature_columns, 
                'train_score': 0.5, 
                'test_score': 0.4,
                'feature_importance': {col: 1.0/len(self.feature_columns) for col in self.feature_columns},
                'training_samples': len(X_synth), 
                'tracks_used': len(processed_tracks),
                'processed_tracks': processed_tracks,
                'status': 'fallback', 
                'fallback_reason': reason
            }
        
        return {
            'error': f'Weather model training failed: {reason}', 
            'status': 'error',
            'tracks_used': len(processed_tracks),
            'processed_tracks': processed_tracks
        }

    def _generate_enhanced_synthetic_weather_data(self, n_samples: int = 200) -> Tuple[List[Dict], List[float]]:
        """Generate enhanced synthetic weather training data"""
        features = []
        targets = []
        
        weather_scenarios = ['hot_dry', 'cool_humid', 'rainy', 'ideal']
        
        for i in range(n_samples):
            scenario = np.random.choice(weather_scenarios)
            
            if scenario == 'hot_dry':
                air_temp = 35 + np.random.normal(0, 3)
                track_temp = air_temp + 10 + np.random.normal(0, 2)
                humidity = 30 + np.random.normal(0, 10)
                impact = 1.5 + np.random.normal(0, 0.5)
            elif scenario == 'cool_humid':
                air_temp = 18 + np.random.normal(0, 3)
                track_temp = air_temp + 5 + np.random.normal(0, 2)
                humidity = 80 + np.random.normal(0, 10)
                impact = 0.8 + np.random.normal(0, 0.3)
            elif scenario == 'rainy':
                air_temp = 15 + np.random.normal(0, 5)
                track_temp = air_temp + 2 + np.random.normal(0, 1)
                humidity = 95 + np.random.normal(0, 3)
                impact = 3.0 + np.random.normal(0, 1.0)
            else:  # ideal
                air_temp = 22 + np.random.normal(0, 2)
                track_temp = air_temp + 8 + np.random.normal(0, 1)
                humidity = 50 + np.random.normal(0, 10)
                impact = 0.2 + np.random.normal(0, 0.2)
            
            feat = self._get_fallback_weather_features('generic')
            # Update with scenario-specific values
            feat.update({
                'air_temp': max(10, min(40, air_temp)),
                'track_temp': max(15, min(50, track_temp)),
                'humidity': max(20, min(100, humidity)),
                'rain': 1.0 if scenario == 'rainy' else 0.0,
                'wind_speed': np.random.uniform(0, 8),
                'wind_direction': np.random.uniform(0, 360)
            })
            
            # Recalculate derived features
            feat['temp_difference'] = feat['track_temp'] - feat['air_temp']
            feat['air_density'] = self._calculate_air_density(feat['air_temp'], feat['pressure'], feat['humidity'])
            feat['heat_index'] = self._calculate_heat_index(feat['air_temp'], feat['humidity'])
            feat['dew_point'] = self._calculate_dew_point(feat['air_temp'], feat['humidity'])
            
            features.append(feat)
            targets.append(impact)
            
        return features, targets

    def predict_weather_impact(self, weather_conditions: Dict[str, float], track_name: str, lap_context: Dict[str, float]) -> float:
        """Predict weather impact on lap performance"""
        try:
            if not self.feature_columns:
                self.logger.warning("⚠️ No trained model available, using fallback")
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
            return max(-8.0, min(8.0, impact))
            
        except Exception as e:
            self.logger.error(f"❌ Weather impact prediction failed: {e}")
            return self._fallback_weather_impact(weather_conditions, track_name)
    
    def _fallback_weather_impact(self, weather_conditions: Dict[str, float], track_name: str) -> float:
        """Enhanced fallback weather impact calculation"""
        base_impact = 0.0
        
        # Temperature effect (enhanced)
        ideal_temp = 22.0
        temp_diff = abs(weather_conditions.get('AIR_TEMP', 25) - ideal_temp)
        base_impact += temp_diff * 0.08  # Increased sensitivity
        
        # Rain effect (enhanced)
        rain = weather_conditions.get('RAIN', 0)
        if rain > 0:
            base_impact += 2.5 + (rain * 0.5)  # Scale with rain intensity
            
        # Humidity effect
        humidity = weather_conditions.get('HUMIDITY', 50)
        humidity_effect = abs(humidity - 50) * 0.02
        base_impact += humidity_effect
        
        # Wind effect
        wind_speed = weather_conditions.get('WIND_SPEED', 0)
        base_impact += wind_speed * 0.1
        
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
        try:
            base_temp = weather_conditions.get('TRACK_TEMP', 30.0)
            air_temp = weather_conditions.get('AIR_TEMP', 25.0)
            
            # Heat from usage (enhanced)
            usage_heat = min(20.0, lap_count * 0.6)  # Increased heat generation
            
            # Track-specific heating
            track_heat = self._get_track_weather_sensitivity(track_name) * 6.0
            
            estimated_temp = base_temp + usage_heat + track_heat
            
            # Enhanced cooling effect
            if air_temp < base_temp:
                cooling_effect = (base_temp - air_temp) * 0.15  # Increased cooling
                estimated_temp -= cooling_effect
                
            return max(air_temp, min(110.0, estimated_temp))  # Increased max temp
            
        except Exception as e:
            self.logger.error(f"❌ Tire temperature estimation failed: {e}")
            return weather_conditions.get('TRACK_TEMP', 30.0)
    
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





















# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import joblib
# from datetime import datetime, timedelta
# from typing import Dict, List, Tuple

# class WeatherModelTrainer:
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#         self.scaler = StandardScaler()
#         self.feature_columns = []
    
#     def train(self, track_data: Dict[str, Dict[str, pd.DataFrame]]) -> dict:
#         """Train weather impact model using structured data from Firebase loader across all tracks"""
#         try:
#             features_list = []
#             impact_list = []
            
#             # Process each track's data
#             for track_name, data_dict in track_data.items():
#                 session_features, session_impacts = self._extract_weather_features(data_dict, track_name)
#                 if not session_features.empty and len(session_impacts) > 0:
#                     features_list.append(session_features)
#                     impact_list.append(session_impacts)
            
#             if not features_list:
#                 return {'error': 'No valid weather features extracted from any track', 'status': 'error'}
            
#             # Combine all track data
#             X = pd.concat(features_list, ignore_index=True)
#             y = np.concatenate(impact_list) if impact_list else np.array([])
            
#             # Remove NaN values (should be minimal due to schema enforcement)
#             if len(y) > 0:
#                 valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
#                 X = X[valid_mask]
#                 y = y[valid_mask]
            
#             if len(X) < 20:
#                 return {'error': f'Insufficient training samples: {len(X)}', 'status': 'error'}
            
#             # Scale features and train
#             X_scaled = self.scaler.fit_transform(X)
#             self.feature_columns = X.columns.tolist()
            
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X_scaled, y, test_size=0.2, random_state=42
#             )
            
#             self.model.fit(X_train, y_train)
#             train_score = self.model.score(X_train, y_train)
#             test_score = self.model.score(X_test, y_test)
            
#             return {
#                 'model': self,
#                 'features': self.feature_columns,
#                 'train_score': train_score,
#                 'test_score': test_score,
#                 'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_)),
#                 'training_samples': len(X),
#                 'tracks_used': len(track_data),
#                 'status': 'success'
#             }
            
#         except Exception as e:
#             return {'error': f'Training failed: {str(e)}', 'status': 'error'}
    
#     def _extract_weather_features(self, data_dict: Dict[str, pd.DataFrame], track_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
#         """Extract weather impact features using EXACT column names from FirebaseDataLoader"""
#         weather_data = data_dict.get('weather_data', pd.DataFrame())
#         pit_data = data_dict.get('pit_data', pd.DataFrame())
#         telemetry_data = data_dict.get('telemetry_data', pd.DataFrame())
        
#         features = []
#         impacts = []
        
#         if pit_data.empty or weather_data.empty:
#             return pd.DataFrame(), np.array([])
        
#         # Prepare timestamps using EXACT column names
#         weather_data = self._prepare_weather_timestamps(weather_data)
#         pit_data = self._prepare_pit_timestamps(pit_data)
        
#         # Process each car's laps
#         for car_number in pit_data['NUMBER'].unique():
#             car_laps = pit_data[pit_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
#             if len(car_laps) < 5:
#                 continue
                
#             baseline_time = self._calculate_baseline_performance(car_laps)
            
#             for _, lap in car_laps.iterrows():
#                 lap_weather = self._get_lap_weather_conditions(lap, weather_data)
#                 if lap_weather is None:
#                     continue
                    
#                 lap_telemetry = self._get_lap_telemetry(lap, telemetry_data, car_number)
#                 weather_impact = self._calculate_weather_impact(lap, baseline_time, lap_weather)
#                 feature_vector = self._create_weather_feature_vector(lap, lap_weather, lap_telemetry, track_name)
                
#                 features.append(pd.DataFrame([feature_vector]))
#                 impacts.append(weather_impact)
        
#         if features:
#             return pd.concat(features, ignore_index=True), np.array(impacts)
#         return pd.DataFrame(), np.array([])
    
#     def _prepare_weather_timestamps(self, weather_data: pd.DataFrame) -> pd.DataFrame:
#         """Prepare weather timestamps using EXACT column names"""
#         weather_data = weather_data.copy()
        
#         if 'TIME_UTC_SECONDS' in weather_data.columns:
#             weather_data['timestamp'] = pd.to_datetime(weather_data['TIME_UTC_SECONDS'], unit='s', errors='coerce')
#         elif 'TIME_UTC_STR' in weather_data.columns:
#             weather_data['timestamp'] = pd.to_datetime(weather_data['TIME_UTC_STR'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
#         else:
#             # Fallback: create synthetic timestamps
#             start_time = datetime.now() - timedelta(hours=2)
#             weather_data['timestamp'] = [start_time + timedelta(seconds=i*30) for i in range(len(weather_data))]
        
#         return weather_data.sort_values('timestamp').dropna(subset=['timestamp'])
    
#     def _prepare_pit_timestamps(self, pit_data: pd.DataFrame) -> pd.DataFrame:
#         """Prepare pit data timestamps using EXACT column names"""
#         pit_data = pit_data.copy()
        
#         if 'HOUR' in pit_data.columns:
#             # Convert HOUR column (e.g., "13:42:35.322") to datetime
#             base_date = datetime.now().date()
#             pit_data['timestamp'] = pd.to_datetime(
#                 base_date.strftime('%Y-%m-%d') + ' ' + pit_data['HOUR'].astype(str),
#                 format='%Y-%m-%d %H:%M:%S.%f',
#                 errors='coerce'
#             )
#         elif 'ELAPSED' in pit_data.columns:
#             # Use ELAPSED time to create relative timestamps
#             base_time = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)
#             pit_data['timestamp'] = pit_data['ELAPSED'].apply(
#                 lambda x: base_time + timedelta(seconds=x) if pd.notna(x) else base_time
#             )
#         else:
#             # Fallback: create synthetic timestamps based on lap number
#             base_time = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)
#             pit_data['timestamp'] = [base_time + timedelta(seconds=i*90) for i in range(len(pit_data))]
        
#         return pit_data.sort_values('timestamp').dropna(subset=['timestamp'])
    
#     def _get_lap_weather_conditions(self, lap: pd.Series, weather_data: pd.DataFrame) -> pd.Series | None:
#         """Get weather conditions for a specific lap using EXACT column names"""
#         lap_time = lap['timestamp']
#         if weather_data.empty:
#             return None
            
#         # Find closest weather reading to lap time
#         time_diff = (weather_data['timestamp'] - lap_time).abs()
#         closest_idx = time_diff.idxmin()
        
#         # Only use if within 5 minutes
#         if time_diff[closest_idx] > timedelta(minutes=5):
#             return None
            
#         return weather_data.loc[closest_idx]
    
#     def _get_lap_telemetry(self, lap: pd.Series, telemetry_data: pd.DataFrame, car_number: int) -> Dict[str, float]:
#         """Get telemetry data for a specific lap using EXACT column names"""
#         if telemetry_data.empty:
#             return {}
            
#         # Extract vehicle number from vehicle_id if needed
#         lap_telemetry = telemetry_data[
#             (telemetry_data['vehicle_id'].str.contains(str(car_number))) &
#             (telemetry_data['lap'] == lap['LAP_NUMBER'])
#         ]
        
#         if lap_telemetry.empty:
#             return {}
            
#         return {
#             'avg_speed': lap_telemetry['speed'].mean() if 'speed' in lap_telemetry.columns else 120.0,
#             'avg_long_accel': lap_telemetry['accx_can'].abs().mean() if 'accx_can' in lap_telemetry.columns else 0.3,
#             'avg_lat_accel': lap_telemetry['accy_can'].abs().mean() if 'accy_can' in lap_telemetry.columns else 0.4,
#             'avg_gear': lap_telemetry['gear'].mean() if 'gear' in lap_telemetry.columns else 3.0
#         }
    
#     def _calculate_baseline_performance(self, car_laps: pd.DataFrame) -> float:
#         """Calculate baseline performance for weather impact comparison"""
#         # Convert lap times to seconds
#         lap_times_seconds = car_laps['LAP_TIME'].apply(self._lap_time_to_seconds).dropna()
        
#         if len(lap_times_seconds) < 3:
#             return lap_times_seconds.mean() if not lap_times_seconds.empty else 100.0
            
#         # Use best 30% of laps as baseline
#         best_laps = lap_times_seconds.nsmallest(max(3, int(len(lap_times_seconds) * 0.3)))
#         return best_laps.median()
    
#     def _calculate_weather_impact(self, lap: pd.Series, baseline_time: float, weather: pd.Series) -> float:
#         """Calculate weather impact on lap performance"""
#         actual_time = self._lap_time_to_seconds(lap['LAP_TIME'])
#         impact = actual_time - baseline_time
        
#         # Only consider impacts beyond normal variation
#         normal_variation = 0.5
#         adjusted_impact = impact if abs(impact) > normal_variation else 0
        
#         return max(-5.0, min(5.0, adjusted_impact))
    
#     def _create_weather_feature_vector(self, lap: pd.Series, weather: pd.Series, telemetry: Dict[str, float], track_name: str) -> Dict[str, float]:
#         """Create weather feature vector using EXACT column names"""
#         features = {
#             # Weather conditions using EXACT column names
#             'air_temp': weather.get('AIR_TEMP', 25.0),
#             'track_temp': weather.get('TRACK_TEMP', 30.0),
#             'humidity': weather.get('HUMIDITY', 50.0),
#             'pressure': weather.get('PRESSURE', 1013.0),
#             'wind_speed': weather.get('WIND_SPEED', 0.0),
#             'wind_direction': weather.get('WIND_DIRECTION', 0.0),
#             'rain': weather.get('RAIN', 0.0),
            
#             # Derived weather features
#             'temp_difference': weather.get('TRACK_TEMP', 30.0) - weather.get('AIR_TEMP', 25.0),
#             'air_density': self._calculate_air_density(
#                 weather.get('AIR_TEMP', 25.0), 
#                 weather.get('PRESSURE', 1013.0), 
#                 weather.get('HUMIDITY', 50.0)
#             ),
#             'wind_effect': self._calculate_wind_effect(
#                 weather.get('WIND_SPEED', 0.0), 
#                 weather.get('WIND_DIRECTION', 0.0)
#             ),
            
#             # Track and context features
#             'track_weather_sensitivity': self._get_track_weather_sensitivity(track_name),
#             'lap_number': lap['LAP_NUMBER'],
#             'time_of_day': lap['timestamp'].hour + lap['timestamp'].minute / 60.0,
            
#             # Telemetry-based features
#             'avg_speed': telemetry.get('avg_speed', 120.0),
#             'driving_aggressiveness': (telemetry.get('avg_long_accel', 0.3) + telemetry.get('avg_lat_accel', 0.4)) / 2,
#             'gear_usage': telemetry.get('avg_gear', 3.0)
#         }
        
#         return features
    
#     def _lap_time_to_seconds(self, lap_time: str) -> float:
#         """Convert lap time string to seconds (consistent with FirebaseDataLoader)"""
#         try:
#             if pd.isna(lap_time) or lap_time == 0:
#                 return 60.0
                
#             time_str = str(lap_time).strip()
#             parts = time_str.split(':')
            
#             if len(parts) == 3:  # HH:MM:SS.ms
#                 hours, minutes, seconds = parts
#                 return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
#             elif len(parts) == 2:  # MM:SS.ms
#                 minutes, seconds = parts
#                 return float(minutes) * 60 + float(seconds)
#             else:
#                 return float(time_str)
#         except:
#             return 60.0
    
#     def _calculate_air_density(self, air_temp: float, pressure: float, humidity: float) -> float:
#         """Calculate air density for aerodynamic effects"""
#         R = 287.05  # Specific gas constant for dry air (J/kg·K)
#         temp_kelvin = air_temp + 273.15
        
#         # Calculate vapor pressure from humidity
#         vapor_pressure = 0.611 * np.exp(17.27 * air_temp / (air_temp + 237.3)) * (humidity / 100)
#         dry_air_pressure = pressure - vapor_pressure
        
#         return (dry_air_pressure * 100) / (R * temp_kelvin)  # Convert hPa to Pa
    
#     def _calculate_wind_effect(self, wind_speed: float, wind_direction: float) -> float:
#         """Calculate wind effect on straight-line speed"""
#         # Simplified wind effect model
#         return wind_speed * 0.1  # 10% impact per m/s
    
#     def _get_track_weather_sensitivity(self, track_name: str) -> float:
#         """Get track-specific weather sensitivity"""
#         sensitivity_map = {
#             'barber': 0.8, 'cota': 0.65, 'indianapolis': 0.5,
#             'road_america': 0.9, 'sebring': 0.85, 'sonoma': 0.75, 'virginia': 0.7
#         }
#         return sensitivity_map.get(track_name.lower(), 0.7)
    
#     def predict_weather_impact(self, weather_conditions: Dict[str, float], track_name: str, lap_context: Dict[str, float]) -> float:
#         """Predict weather impact on lap performance"""
#         try:
#             if not self.feature_columns:
#                 return self._fallback_weather_impact(weather_conditions, track_name)
                
#             # Create feature vector
#             feature_vector = self._create_weather_feature_vector(
#                 lap_context.get('lap_info', {}),
#                 weather_conditions,
#                 lap_context.get('telemetry', {}),
#                 track_name
#             )
            
#             # Ensure all features are present
#             feature_array = np.array([feature_vector.get(col, 0.0) for col in self.feature_columns]).reshape(1, -1)
#             scaled_features = self.scaler.transform(feature_array)
            
#             impact = self.model.predict(scaled_features)[0]
#             return max(-5.0, min(5.0, impact))
            
#         except Exception as e:
#             print(f"⚠️ Weather impact prediction error: {e}")
#             return self._fallback_weather_impact(weather_conditions, track_name)
    
#     def _fallback_weather_impact(self, weather_conditions: Dict[str, float], track_name: str) -> float:
#         """Fallback weather impact calculation"""
#         base_impact = 0.0
        
#         # Temperature effect
#         temp_diff = abs(weather_conditions.get('AIR_TEMP', 25) - 25)
#         base_impact += temp_diff * 0.05
        
#         # Rain effect
#         if weather_conditions.get('RAIN', 0) > 0:
#             base_impact += 2.0
            
#         # Track sensitivity
#         sensitivity = self._get_track_weather_sensitivity(track_name)
        
#         return base_impact * sensitivity
    
#     def get_optimal_conditions(self, track_name: str) -> Dict[str, float]:
#         """Get optimal weather conditions for a track"""
#         return {
#             'AIR_TEMP': 22.0,
#             'TRACK_TEMP': 30.0, 
#             'HUMIDITY': 50.0,
#             'PRESSURE': 1013.0,
#             'WIND_SPEED': 2.0,
#             'RAIN': 0.0
#         }
    
#     def estimate_tire_temperature(self, weather_conditions: Dict[str, float], track_name: str, lap_count: int) -> float:
#         """Estimate tire temperature based on weather and usage"""
#         base_temp = weather_conditions.get('TRACK_TEMP', 30.0)
#         air_temp = weather_conditions.get('AIR_TEMP', 25.0)
        
#         # Heat from usage
#         usage_heat = min(15.0, lap_count * 0.5)
        
#         # Track-specific heating
#         track_heat = self._get_track_weather_sensitivity(track_name) * 5.0
        
#         estimated_temp = base_temp + usage_heat + track_heat
        
#         # Cooling effect if air is cooler than track
#         if air_temp < base_temp:
#             estimated_temp -= (base_temp - air_temp) * 0.1
            
#         return max(air_temp, min(100.0, estimated_temp))
    
#     def save_model(self, filepath: str):
#         """Save trained model to file"""
#         model_data = {
#             'model': self.model,
#             'scaler': self.scaler,
#             'feature_columns': self.feature_columns
#         }
#         joblib.dump(model_data, filepath)
    
#     def load_model(self, filepath: str):
#         """Load trained model from file"""
#         model_data = joblib.load(filepath)
#         self.model = model_data['model']
#         self.scaler = model_data['scaler']
#         self.feature_columns = model_data['feature_columns']