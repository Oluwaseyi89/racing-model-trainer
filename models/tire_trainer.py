import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import joblib
from typing import Dict, List, Tuple
import logging

class TireModelTrainer:
    def __init__(self):
        self.model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        )
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_columns = ['degradation_s1', 'degradation_s2', 'degradation_s3', 'grip_loss_rate']
        self.logger = logging.getLogger(__name__)
        
        # Minimum required columns - reduced for flexibility
        self.minimum_pit_cols = ['NUMBER', 'LAP_NUMBER', 'LAP_TIME']

    def train(self, track_data: Dict[str, Dict[str, pd.DataFrame]]) -> dict:
        """
        Train tire degradation model with enhanced error handling and vehicle_id mapping
        """
        try:
            self.logger.info("🚀 Starting tire model training...")
            
            # Validate input data structure
            if not isinstance(track_data, dict) or len(track_data) < 2:
                error_msg = 'Insufficient tracks for tire model'
                self.logger.error(f"❌ {error_msg}")
                return {'error': error_msg, 'status': 'error'}

            features_list = []
            targets_list = []
            processed_tracks = []

            # Process each track's data with enhanced logging
            for track_name, data_dict in track_data.items():
                self.logger.info(f"📊 Processing track: {track_name}")
                
                if not self._validate_track_data(data_dict):
                    self.logger.warning(f"⚠️ Skipping {track_name}: validation failed")
                    continue
                    
                track_features, track_targets = self._extract_track_tire_features(data_dict, track_name)
                
                if not track_features.empty and not track_targets.empty:
                    features_list.append(track_features)
                    targets_list.append(track_targets)
                    processed_tracks.append(track_name)
                    self.logger.info(f"✅ Extracted {len(track_features)} features from {track_name}")
                else:
                    self.logger.warning(f"⚠️ No features extracted from {track_name}")

            if not features_list:
                error_msg = 'No valid tire training data extracted from any track'
                self.logger.error(f"❌ {error_msg}")
                return {'error': error_msg, 'status': 'error'}

            # Combine all track data
            X = pd.concat(features_list, ignore_index=True)
            y = pd.concat(targets_list, ignore_index=True)
            
            # Ensure target columns exist
            missing_targets = [col for col in self.target_columns if col not in y.columns]
            if missing_targets:
                self.logger.warning(f"⚠️ Missing target columns: {missing_targets}, creating defaults")
                for col in missing_targets:
                    y[col] = 0.05  # Default degradation rate

            y = y[self.target_columns]

            if X.empty or y.empty:
                error_msg = 'Empty feature or target matrices after processing'
                self.logger.error(f"❌ {error_msg}")
                return {'error': error_msg, 'status': 'error'}

            self.logger.info(f"📈 Final dataset: {len(X)} samples, {len(X.columns)} features")

            # Clean data
            valid_mask = ~X.isna().any(axis=1) & ~y.isna().any(axis=1)
            X = X[valid_mask]
            y = y[valid_mask]

            if len(X) < 10:  # Reduced threshold for more flexibility
                error_msg = f'Insufficient training samples: {len(X)}'
                self.logger.error(f"❌ {error_msg}")
                return {'error': error_msg, 'status': 'error'}

            # Scale features and train
            X_scaled = self.scaler.fit_transform(X)
            self.feature_columns = X.columns.tolist()

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            self.logger.info("🏃 Training model...")
            self.model.fit(X_train, y_train)
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Calculate average feature importance across all outputs
            avg_feature_importance = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
            feature_importance = dict(zip(self.feature_columns, avg_feature_importance))

            self.logger.info(f"✅ Tire model trained successfully - Test Score: {test_score:.3f}")

            return {
                'model': self,
                'features': self.feature_columns,
                'targets': self.target_columns,
                'train_score': train_score,
                'test_score': test_score,
                'feature_importance': feature_importance,
                'training_samples': len(X),
                'tracks_used': len(processed_tracks),
                'processed_tracks': processed_tracks,
                'status': 'success'
            }
            
        except Exception as e:
            error_msg = f'Training failed: {str(e)}'
            self.logger.error(f"❌ {error_msg}")
            return {'error': error_msg, 'status': 'error'}

    def _validate_track_data(self, data_dict: Dict[str, pd.DataFrame]) -> bool:
        """Enhanced validation with better logging and flexibility"""
        pit_data = data_dict.get('pit_data', pd.DataFrame())
        
        if pit_data.empty:
            self.logger.debug("⚠️ No pit data available")
            return False
            
        # Check for minimum required columns
        missing_pit = [col for col in self.minimum_pit_cols if col not in pit_data.columns]
        if missing_pit:
            self.logger.debug(f"⚠️ Missing required columns: {missing_pit}")
            return False
            
        # Check for minimum data volume
        if len(pit_data) < 3:  # Reduced threshold
            self.logger.debug(f"⚠️ Insufficient pit data rows: {len(pit_data)}")
            return False
            
        self.logger.debug(f"✅ Track data validated: {len(pit_data)} rows, {len(pit_data.columns)} columns")
        return True

    def _extract_track_tire_features(self, data_dict: Dict[str, pd.DataFrame], track_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract tire degradation features with enhanced vehicle_id mapping
        """
        pit_data = data_dict['pit_data']
        telemetry_data = data_dict.get('telemetry_data', pd.DataFrame())
        weather_data = data_dict.get('weather_data', pd.DataFrame())

        features_list = []
        targets_list = []

        # Add vehicle_id mapping to pit_data for telemetry integration
        pit_data = self._add_vehicle_id_mapping(pit_data)
        
        self.logger.debug(f"🔧 Processing {len(pit_data['NUMBER'].unique())} cars in {track_name}")

        # Process each car's stint patterns
        for car_number in pit_data['NUMBER'].unique():
            car_laps = pit_data[pit_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
            if len(car_laps) < 3:  # Reduced minimum laps requirement
                self.logger.debug(f"⚠️ Car {car_number}: insufficient laps ({len(car_laps)})")
                continue

            # Analyze stint patterns
            stint_features, stint_targets = self._analyze_car_stints(
                car_laps, telemetry_data, weather_data, track_name, car_number
            )
            
            if not stint_features.empty and not stint_targets.empty:
                features_list.append(stint_features)
                targets_list.append(stint_targets)
                self.logger.debug(f"✅ Car {car_number}: extracted {len(stint_features)} stints")
            else:
                self.logger.debug(f"⚠️ Car {car_number}: no valid stints extracted")

        if features_list and targets_list:
            result_features = pd.concat(features_list, ignore_index=True)
            result_targets = pd.concat(targets_list, ignore_index=True)
            self.logger.debug(f"✅ {track_name}: extracted {len(result_features)} total stints")
            return result_features, result_targets
        
        self.logger.debug(f"❌ {track_name}: no features extracted")
        return pd.DataFrame(), pd.DataFrame()

    def _add_vehicle_id_mapping(self, pit_data: pd.DataFrame) -> pd.DataFrame:
        """Create vehicle_id mapping for telemetry data integration"""
        pit_data = pit_data.copy()
        
        # Create consistent vehicle_id format matching telemetry data
        pit_data['vehicle_id'] = 'GR86-' + pit_data['NUMBER'].astype(str).str.zfill(3) + '-000'
        
        return pit_data

    def _analyze_car_stints(self, car_laps: pd.DataFrame, telemetry_data: pd.DataFrame, 
                           weather_data: pd.DataFrame, track_name: str, car_number: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze tire degradation across different stints with enhanced error handling
        """
        features = []
        targets = []
        
        # Convert lap times to seconds
        car_laps = car_laps.copy()
        car_laps['LAP_TIME_SECONDS'] = car_laps['LAP_TIME'].apply(self._lap_time_to_seconds)
        
        # Use flexible window sizing based on available data
        max_window_size = min(5, len(car_laps) - 1)
        if max_window_size < 2:
            return pd.DataFrame(), pd.DataFrame()
            
        window_size = max(2, max_window_size)  # Minimum 2 laps per window
        
        for i in range(len(car_laps) - window_size):
            current_stint = car_laps.iloc[i:i + window_size]
            next_stint = car_laps.iloc[i + window_size:min(i + window_size * 2, len(car_laps))]
            
            if len(next_stint) < 2:  # Need at least 2 laps for target calculation
                continue
                
            try:
                # Extract features from current stint
                stint_features = self._calculate_stint_features(
                    current_stint, telemetry_data, weather_data, track_name, car_number
                )
                
                # Calculate degradation targets from next stint
                degradation_targets = self._calculate_degradation_targets(current_stint, next_stint)
                
                features.append(pd.DataFrame([stint_features]))
                targets.append(pd.DataFrame([degradation_targets]))
                
            except Exception as e:
                self.logger.debug(f"⚠️ Stint analysis failed for car {car_number}, stint {i}: {e}")
                continue
        
        if features and targets:
            return pd.concat(features, ignore_index=True), pd.concat(targets, ignore_index=True)
        return pd.DataFrame(), pd.DataFrame()

    def _calculate_stint_features(self, stint_laps: pd.DataFrame, telemetry_data: pd.DataFrame,
                                weather_data: pd.DataFrame, track_name: str, car_number: int) -> Dict[str, float]:
        """Calculate tire degradation features with robust fallbacks"""
        features = {}
        
        try:
            # Lap time degradation analysis
            lap_times = stint_laps['LAP_TIME_SECONDS'].values
            lap_numbers = stint_laps['LAP_NUMBER'].values
            
            # Linear degradation trend
            if len(lap_times) > 1:
                time_slope, time_r2 = self._linear_trend_analysis(lap_numbers, lap_times)
                features['lap_time_degradation_slope'] = max(0.0, time_slope)
                features['lap_time_consistency'] = time_r2
            else:
                features['lap_time_degradation_slope'] = 0.1
                features['lap_time_consistency'] = 0.0
                
            # Sector-specific degradation with robust handling
            for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
                if sector in stint_laps.columns:
                    sector_times = pd.to_numeric(stint_laps[sector], errors='coerce').fillna(0).values
                    if len(sector_times) > 1 and not np.all(sector_times == 0):
                        sector_slope, _ = self._linear_trend_analysis(lap_numbers, sector_times)
                        features[f'sector_{i}_degradation_slope'] = max(0.0, sector_slope)
                    else:
                        features[f'sector_{i}_degradation_slope'] = 0.03  # Conservative default
                else:
                    features[f'sector_{i}_degradation_slope'] = 0.03
            
            # Additional performance metrics with fallbacks
            features['avg_top_speed'] = stint_laps['TOP_SPEED'].mean() if 'TOP_SPEED' in stint_laps.columns else 150.0
            features['avg_kph'] = stint_laps['KPH'].mean() if 'KPH' in stint_laps.columns else 120.0
            features['lap_improvement_ratio'] = (stint_laps['LAP_IMPROVEMENT'] > 0).mean() if 'LAP_IMPROVEMENT' in stint_laps.columns else 0.5
            
            # Caution flag analysis
            if 'FLAG_AT_FL' in stint_laps.columns:
                caution_flags = stint_laps[stint_laps['FLAG_AT_FL'].str.contains('FCY|SC', na=False)]
                features['caution_flag_ratio'] = len(caution_flags) / len(stint_laps)
            else:
                features['caution_flag_ratio'] = 0.1
            
            # Performance metrics
            features['lap_time_variance'] = np.var(lap_times) if len(lap_times) > 1 else 1.0
            features['performance_consistency'] = 1.0 / (1.0 + features['lap_time_variance'])
            
            # Track and condition factors
            features.update(self._calculate_track_conditions(stint_laps, weather_data, track_name))
            
            # Driving style factors (from telemetry if available)
            features.update(self._calculate_driving_factors(stint_laps, telemetry_data, car_number))
            
            # Stint characteristics
            features['stint_length'] = len(stint_laps)
            features['cumulative_laps'] = stint_laps['LAP_NUMBER'].max()
            features['car_number'] = car_number  # Add car identifier for debugging
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature calculation failed: {e}")
            # Provide basic fallback features
            features = self._get_fallback_features(track_name, len(stint_laps))
            
        return features

    def _calculate_driving_factors(self, stint_laps: pd.DataFrame, telemetry_data: pd.DataFrame, car_number: int) -> Dict[str, float]:
        """Calculate driving style factors with enhanced telemetry integration"""
        factors = {
            'estimated_lateral_force': 0.5,
            'estimated_braking_force': 0.3,
            'driving_aggressiveness': 0.6,
            'gear_usage_efficiency': 0.7
        }
        
        if not telemetry_data.empty:
            try:
                # Use vehicle_id for matching
                vehicle_id = 'GR86-' + str(car_number).zfill(3) + '-000'
                stint_lap_numbers = stint_laps['LAP_NUMBER'].values
                
                # Filter telemetry for this car and stint laps
                car_telemetry = telemetry_data[
                    (telemetry_data['vehicle_id'] == vehicle_id) &
                    (telemetry_data['lap'].isin(stint_lap_numbers))
                ]
                
                if not car_telemetry.empty:
                    # Estimate lateral forces from lateral acceleration
                    if 'accy_can' in car_telemetry.columns:
                        factors['estimated_lateral_force'] = car_telemetry['accy_can'].abs().mean()
                    
                    # Estimate braking from longitudinal acceleration
                    if 'accx_can' in car_telemetry.columns:
                        braking_events = car_telemetry[car_telemetry['accx_can'] < -0.3]  # Less strict threshold
                        factors['estimated_braking_force'] = len(braking_events) / len(car_telemetry) if len(car_telemetry) > 0 else 0.3
                    
                    # Gear usage efficiency
                    if 'gear' in car_telemetry.columns:
                        optimal_gear_ratio = (car_telemetry['gear'].between(2, 6)).mean()  # Wider gear range
                        factors['gear_usage_efficiency'] = optimal_gear_ratio
                    
                    # Driving aggressiveness (speed variance + acceleration patterns)
                    if 'speed' in car_telemetry.columns:
                        speed_variance = car_telemetry['speed'].var()
                        factors['driving_aggressiveness'] = min(1.0, (speed_variance / 500.0) if not np.isnan(speed_variance) else 0.6)
                        
            except Exception as e:
                self.logger.debug(f"⚠️ Telemetry analysis failed for car {car_number}: {e}")
                
        return factors

    def _calculate_track_conditions(self, stint_laps: pd.DataFrame, weather_data: pd.DataFrame, 
                                  track_name: str) -> Dict[str, float]:
        """Calculate track and weather conditions with fallbacks"""
        conditions = {}
        
        # Track characteristics
        conditions['track_abrasiveness'] = self._get_track_abrasiveness(track_name)
        conditions['track_length_factor'] = self._get_track_length_factor(track_name)
        
        # Weather conditions with robust handling
        if not weather_data.empty:
            try:
                conditions['avg_track_temp'] = weather_data['TRACK_TEMP'].mean() if 'TRACK_TEMP' in weather_data.columns else 35.0
                conditions['track_temp_variance'] = weather_data['TRACK_TEMP'].var() if 'TRACK_TEMP' in weather_data.columns else 5.0
                conditions['avg_air_temp'] = weather_data['AIR_TEMP'].mean() if 'AIR_TEMP' in weather_data.columns else 25.0
                conditions['humidity_level'] = weather_data['HUMIDITY'].mean() if 'HUMIDITY' in weather_data.columns else 50.0
                conditions['pressure_level'] = weather_data['PRESSURE'].mean() if 'PRESSURE' in weather_data.columns else 1013.0
            except Exception:
                # Fallback if weather data exists but calculation fails
                conditions.update(self._get_fallback_weather_conditions())
        else:
            conditions.update(self._get_fallback_weather_conditions())
            
        return conditions

    def _calculate_degradation_targets(self, current_stint: pd.DataFrame, next_stint: pd.DataFrame) -> Dict[str, float]:
        """Calculate degradation targets with robust error handling"""
        targets = {}
        
        try:
            # Sector degradation (time increase per lap)
            for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
                if sector in current_stint.columns and sector in next_stint.columns:
                    current_avg = pd.to_numeric(current_stint[sector], errors='coerce').replace(0, np.nan).mean()
                    next_avg = pd.to_numeric(next_stint[sector], errors='coerce').replace(0, np.nan).mean()
                    
                    if not np.isnan(current_avg) and not np.isnan(next_avg) and current_avg > 0:
                        degradation_per_lap = (next_avg - current_avg) / len(next_stint)
                        targets[f'degradation_s{i}'] = max(0.001, min(0.5, degradation_per_lap))
                    else:
                        targets[f'degradation_s{i}'] = 0.03  # Conservative default
                else:
                    targets[f'degradation_s{i}'] = 0.03
            
            # Overall grip loss rate
            current_avg_time = current_stint['LAP_TIME_SECONDS'].mean()
            next_avg_time = next_stint['LAP_TIME_SECONDS'].mean()
            
            if current_avg_time > 0 and next_avg_time > 0:
                grip_loss = (next_avg_time - current_avg_time) / len(next_stint)
                targets['grip_loss_rate'] = max(0.001, min(1.0, grip_loss))
            else:
                targets['grip_loss_rate'] = 0.05
                
        except Exception as e:
            self.logger.debug(f"⚠️ Target calculation failed: {e}")
            # Fallback targets
            for i in range(1, 4):
                targets[f'degradation_s{i}'] = 0.03
            targets['grip_loss_rate'] = 0.05
        
        return targets

    def _get_fallback_features(self, track_name: str, stint_length: int) -> Dict[str, float]:
        """Provide fallback features when calculation fails"""
        return {
            'lap_time_degradation_slope': 0.1,
            'lap_time_consistency': 0.0,
            'sector_1_degradation_slope': 0.03,
            'sector_2_degradation_slope': 0.03,
            'sector_3_degradation_slope': 0.03,
            'avg_top_speed': 150.0,
            'avg_kph': 120.0,
            'lap_improvement_ratio': 0.5,
            'caution_flag_ratio': 0.1,
            'lap_time_variance': 1.0,
            'performance_consistency': 0.5,
            'track_abrasiveness': self._get_track_abrasiveness(track_name),
            'track_length_factor': self._get_track_length_factor(track_name),
            'stint_length': stint_length,
            'cumulative_laps': stint_length,
            **self._get_fallback_weather_conditions(),
            'estimated_lateral_force': 0.5,
            'estimated_braking_force': 0.3,
            'driving_aggressiveness': 0.6,
            'gear_usage_efficiency': 0.7
        }

    def _get_fallback_weather_conditions(self) -> Dict[str, float]:
        """Provide fallback weather conditions"""
        return {
            'avg_track_temp': 35.0,
            'track_temp_variance': 5.0,
            'avg_air_temp': 25.0,
            'humidity_level': 50.0,
            'pressure_level': 1013.0
        }

    def _linear_trend_analysis(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Calculate linear trend slope and R² value with enhanced robustness"""
        if len(x) < 2:
            return 0.0, 0.0
        try:
            # Remove NaN values
            mask = (~np.isnan(x)) & (~np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 2:
                return 0.0, 0.0
                
            slope = np.polyfit(x_clean, y_clean, 1)[0]
            correlation = np.corrcoef(x_clean, y_clean)[0, 1]
            r_squared = correlation ** 2 if not np.isnan(correlation) else 0.0
            return slope, r_squared
        except:
            return 0.0, 0.0

    def _get_track_abrasiveness(self, track_name: str) -> float:
        """Estimate track abrasiveness based on actual track names"""
        high_abrasion = ['barber', 'sonoma', 'sebring']
        medium_abrasion = ['cota', 'road_america', 'virginia']
        
        track_lower = track_name.lower()
        if any(track in track_lower for track in high_abrasion):
            return 0.8
        elif any(track in track_lower for track in medium_abrasion):
            return 0.5
        else:
            return 0.6

    def _get_track_length_factor(self, track_name: str) -> float:
        """Normalize by track length (simplified)"""
        long_tracks = ['road_america', 'cota']
        short_tracks = ['barber', 'sonoma']
        
        track_lower = track_name.lower()
        if any(track in track_lower for track in long_tracks):
            return 1.2
        elif any(track in track_lower for track in short_tracks):
            return 0.8
        else:
            return 1.0

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

    def predict_tire_degradation(self, features: Dict[str, float]) -> Dict[str, float]:
        """Predict tire degradation rates for given features"""
        try:
            if not self.feature_columns:
                self.logger.warning("⚠️ No trained model available, using fallback")
                return self._get_fallback_degradation()
                
            # Ensure all features are present
            feature_vector = [features.get(col, 0.0) for col in self.feature_columns]
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            predictions = self.model.predict(X_scaled)[0]
            return dict(zip(self.target_columns, predictions))
            
        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {e}")
            return self._get_fallback_degradation()

    def _get_fallback_degradation(self) -> Dict[str, float]:
        """Fallback degradation rates when model is unavailable"""
        return {
            'degradation_s1': 0.05,
            'degradation_s2': 0.05, 
            'degradation_s3': 0.05,
            'grip_loss_rate': 0.1
        }

    def estimate_optimal_stint_length(self, features: Dict[str, float], performance_threshold: float = 0.2) -> int:
        """Estimate optimal stint length before tire performance drops below threshold"""
        try:
            degradation_rates = self.predict_tire_degradation(features)
            avg_degradation = np.mean([
                degradation_rates['degradation_s1'],
                degradation_rates['degradation_s2'], 
                degradation_rates['degradation_s3']
            ])
            
            if avg_degradation <= 0:
                return 20
                
            optimal_laps = int(performance_threshold / avg_degradation)
            return max(5, min(30, optimal_laps))
            
        except Exception as e:
            self.logger.error(f"❌ Stint length estimation failed: {e}")
            return 15

    def save_model(self, filepath: str):
        """Save trained model to file"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns
        }, filepath)

    def load_model(self, filepath: str):
        """Load trained model from file"""
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
# from typing import Dict, List, Tuple

# class TireModelTrainer:
#     def __init__(self):
#         self.model = MultiOutputRegressor(
#             RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#         )
#         self.scaler = StandardScaler()
#         self.feature_columns = []
#         self.target_columns = ['degradation_s1', 'degradation_s2', 'degradation_s3', 'grip_loss_rate']
        
#         # Updated to match EXACT column names from FirebaseDataLoader schemas
#         self.required_pit_cols = ['NUMBER', 'DRIVER_NUMBER', 'LAP_NUMBER', 'LAP_TIME', 'LAP_IMPROVEMENT', 'CROSSING_FINISH_LINE_IN_PIT', 'S1', 'S1_IMPROVEMENT', 'S2', 'S2_IMPROVEMENT', 'S3', 'S3_IMPROVEMENT', 'KPH', 'ELAPSED', 'HOUR', 'S1_LARGE', 'S2_LARGE', 'S3_LARGE', 'TOP_SPEED', 'PIT_TIME', 'CLASS', 'GROUP', 'MANUFACTURER', 'FLAG_AT_FL', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS', 'IM1a_time', 'IM1a_elapsed', 'IM1_time', 'IM1_elapsed', 'IM2a_time', 'IM2a_elapsed', 'IM2_time', 'IM2_elapsed', 'IM3a_time', 'IM3a_elapsed', 'FL_time', 'FL_elapsed']
#         self.required_telemetry_cols = ['timestamp', 'vehicle_id', 'lap', 'outing', 'meta_session', 'accx_can', 'accy_can', 'gear', 'speed']
#         self.required_weather_cols = ['TIME_UTC_SECONDS', 'TIME_UTC_STR', 'AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 'PRESSURE', 'WIND_SPEED', 'WIND_DIRECTION', 'RAIN']

#     def train(self, track_data: Dict[str, Dict[str, pd.DataFrame]]) -> dict:
#         """
#         Train tire degradation model using structured data from Firebase loader across all tracks
#         """
#         try:
#             # Validate input data structure
#             if not isinstance(track_data, dict) or len(track_data) < 2:
#                 return {'error': 'Insufficient tracks for tire model', 'status': 'error'}

#             features_list = []
#             targets_list = []

#             # Process each track's data
#             for track_name, data_dict in track_data.items():
#                 if not self._validate_track_data(data_dict):
#                     continue
                    
#                 track_features, track_targets = self._extract_track_tire_features(data_dict, track_name)
#                 if not track_features.empty and not track_targets.empty:
#                     features_list.append(track_features)
#                     targets_list.append(track_targets)

#             if not features_list:
#                 return {'error': 'No valid tire training data extracted', 'status': 'error'}

#             # Combine all track data
#             X = pd.concat(features_list, ignore_index=True)
#             y = pd.concat(targets_list, ignore_index=True)[self.target_columns]

#             if X.empty or y.empty:
#                 return {'error': 'Empty feature or target matrices', 'status': 'error'}

#             # Clean data (should be minimal due to schema enforcement)
#             valid_mask = ~X.isna().any(axis=1) & ~y.isna().any(axis=1)
#             X = X[valid_mask]
#             y = y[valid_mask]

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
            
#             # Calculate average feature importance across all outputs
#             avg_feature_importance = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
#             feature_importance = dict(zip(self.feature_columns, avg_feature_importance))

#             return {
#                 'model': self,
#                 'features': self.feature_columns,
#                 'targets': self.target_columns,
#                 'train_score': train_score,
#                 'test_score': test_score,
#                 'feature_importance': feature_importance,
#                 'training_samples': len(X),
#                 'tracks_used': len(track_data),
#                 'status': 'success'
#             }
            
#         except Exception as e:
#             return {'error': f'Training failed: {str(e)}', 'status': 'error'}

#     def _validate_track_data(self, data_dict: Dict[str, pd.DataFrame]) -> bool:
#         """Validate that track data has required components for tire analysis using EXACT column names"""
#         pit_data = data_dict.get('pit_data', pd.DataFrame())
        
#         if pit_data.empty:
#             return False
            
#         # Check for required columns using EXACT names
#         missing_pit = [col for col in ['NUMBER', 'LAP_NUMBER', 'LAP_TIME', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'] if col not in pit_data.columns]
        
#         return len(missing_pit) == 0

#     def _extract_track_tire_features(self, data_dict: Dict[str, pd.DataFrame], track_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         """
#         Extract tire degradation features for all cars in a track using EXACT column names
#         """
#         pit_data = data_dict['pit_data']
#         telemetry_data = data_dict.get('telemetry_data', pd.DataFrame())
#         weather_data = data_dict.get('weather_data', pd.DataFrame())

#         features_list = []
#         targets_list = []

#         # Process each car's stint patterns using EXACT column names
#         for car_number in pit_data['NUMBER'].unique():
#             car_laps = pit_data[pit_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
#             if len(car_laps) < 5:  # Need sufficient laps for degradation analysis
#                 continue

#             # Analyze stint patterns (groups of consecutive laps)
#             stint_features, stint_targets = self._analyze_car_stints(car_laps, telemetry_data, weather_data, track_name)
            
#             if not stint_features.empty and not stint_targets.empty:
#                 features_list.append(stint_features)
#                 targets_list.append(stint_targets)

#         if features_list and targets_list:
#             return pd.concat(features_list, ignore_index=True), pd.concat(targets_list, ignore_index=True)
#         return pd.DataFrame(), pd.DataFrame()

#     def _analyze_car_stints(self, car_laps: pd.DataFrame, telemetry_data: pd.DataFrame, 
#                            weather_data: pd.DataFrame, track_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         """
#         Analyze tire degradation across different stints for a single car using EXACT column names
#         """
#         features = []
#         targets = []
        
#         # Convert lap times to seconds for analysis using EXACT column names
#         car_laps = car_laps.copy()
#         car_laps['LAP_TIME_SECONDS'] = car_laps['LAP_TIME'].apply(self._lap_time_to_seconds)
        
#         # Use a sliding window to analyze degradation patterns
#         window_size = min(5, len(car_laps) - 1)
        
#         for i in range(len(car_laps) - window_size):
#             current_stint = car_laps.iloc[i:i + window_size]
#             next_stint = car_laps.iloc[i + window_size:min(i + window_size * 2, len(car_laps))]
            
#             if len(next_stint) < 2:  # Need at least 2 laps for target calculation
#                 continue
                
#             # Extract features from current stint using EXACT column names
#             stint_features = self._calculate_stint_features(current_stint, telemetry_data, weather_data, track_name)
            
#             # Calculate degradation targets from next stint using EXACT column names
#             degradation_targets = self._calculate_degradation_targets(current_stint, next_stint)
            
#             features.append(pd.DataFrame([stint_features]))
#             targets.append(pd.DataFrame([degradation_targets]))
        
#         if features and targets:
#             return pd.concat(features, ignore_index=True), pd.concat(targets, ignore_index=True)
#         return pd.DataFrame(), pd.DataFrame()

#     def _calculate_stint_features(self, stint_laps: pd.DataFrame, telemetry_data: pd.DataFrame,
#                                 weather_data: pd.DataFrame, track_name: str) -> Dict[str, float]:
#         """Calculate tire degradation features from a stint using EXACT column names"""
#         features = {}
        
#         # Lap time degradation analysis using EXACT column names
#         lap_times = stint_laps['LAP_TIME_SECONDS'].values
#         lap_numbers = stint_laps['LAP_NUMBER'].values
        
#         # Linear degradation trend
#         if len(lap_times) > 1:
#             time_slope, time_r2 = self._linear_trend_analysis(lap_numbers, lap_times)
#             features['lap_time_degradation_slope'] = max(0.0, time_slope)
#             features['lap_time_consistency'] = time_r2
#         else:
#             features['lap_time_degradation_slope'] = 0.1
#             features['lap_time_consistency'] = 0.0
            
#         # Sector-specific degradation using EXACT column names
#         for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
#             if sector in stint_laps.columns:
#                 sector_times = pd.to_numeric(stint_laps[sector], errors='coerce').fillna(0).values
#                 if len(sector_times) > 1:
#                     sector_slope, _ = self._linear_trend_analysis(lap_numbers, sector_times)
#                     features[f'sector_{i}_degradation_slope'] = max(0.0, sector_slope)
#                 else:
#                     features[f'sector_{i}_degradation_slope'] = 0.05
#             else:
#                 features[f'sector_{i}_degradation_slope'] = 0.05
        
#         # Additional performance metrics using EXACT column names
#         if 'TOP_SPEED' in stint_laps.columns:
#             features['avg_top_speed'] = stint_laps['TOP_SPEED'].mean()
#         else:
#             features['avg_top_speed'] = 150.0
            
#         if 'KPH' in stint_laps.columns:
#             features['avg_kph'] = stint_laps['KPH'].mean()
#         else:
#             features['avg_kph'] = 120.0
            
#         if 'LAP_IMPROVEMENT' in stint_laps.columns:
#             features['lap_improvement_ratio'] = (stint_laps['LAP_IMPROVEMENT'] > 0).mean()
#         else:
#             features['lap_improvement_ratio'] = 0.5
            
#         if 'FLAG_AT_FL' in stint_laps.columns:
#             caution_flags = stint_laps[stint_laps['FLAG_AT_FL'].str.contains('FCY|SC', na=False)]
#             features['caution_flag_ratio'] = len(caution_flags) / len(stint_laps)
#         else:
#             features['caution_flag_ratio'] = 0.1
        
#         # Performance metrics
#         features['lap_time_variance'] = np.var(lap_times) if len(lap_times) > 1 else 1.0
#         features['performance_consistency'] = 1.0 / (1.0 + features['lap_time_variance'])
        
#         # Track and condition factors using EXACT column names
#         features.update(self._calculate_track_conditions(stint_laps, weather_data, track_name))
        
#         # Driving style factors (from telemetry if available) using EXACT column names
#         features.update(self._calculate_driving_factors(stint_laps, telemetry_data))
        
#         # Stint characteristics
#         features['stint_length'] = len(stint_laps)
#         features['cumulative_laps'] = stint_laps['LAP_NUMBER'].max()
        
#         return features

#     def _calculate_track_conditions(self, stint_laps: pd.DataFrame, weather_data: pd.DataFrame, 
#                                   track_name: str) -> Dict[str, float]:
#         """Calculate track and weather conditions affecting tire wear using EXACT column names"""
#         conditions = {}
        
#         # Track characteristics
#         conditions['track_abrasiveness'] = self._get_track_abrasiveness(track_name)
#         conditions['track_length_factor'] = self._get_track_length_factor(track_name)
        
#         # Weather conditions using EXACT column names
#         if not weather_data.empty:
#             conditions['avg_track_temp'] = weather_data['TRACK_TEMP'].mean() if 'TRACK_TEMP' in weather_data.columns else 35.0
#             conditions['track_temp_variance'] = weather_data['TRACK_TEMP'].var() if 'TRACK_TEMP' in weather_data.columns else 5.0
#             conditions['avg_air_temp'] = weather_data['AIR_TEMP'].mean() if 'AIR_TEMP' in weather_data.columns else 25.0
#             conditions['humidity_level'] = weather_data['HUMIDITY'].mean() if 'HUMIDITY' in weather_data.columns else 50.0
#             conditions['pressure_level'] = weather_data['PRESSURE'].mean() if 'PRESSURE' in weather_data.columns else 1013.0
#         else:
#             conditions['avg_track_temp'] = 35.0
#             conditions['track_temp_variance'] = 5.0
#             conditions['avg_air_temp'] = 25.0
#             conditions['humidity_level'] = 50.0
#             conditions['pressure_level'] = 1013.0
            
#         return conditions

#     def _calculate_driving_factors(self, stint_laps: pd.DataFrame, telemetry_data: pd.DataFrame) -> Dict[str, float]:
#         """Calculate driving style factors from telemetry data using EXACT column names"""
#         factors = {
#             'estimated_lateral_force': 0.5,
#             'estimated_braking_force': 0.3,
#             'driving_aggressiveness': 0.6,
#             'gear_usage_efficiency': 0.7
#         }
        
#         if not telemetry_data.empty:
#             car_number = stint_laps['NUMBER'].iloc[0]
#             stint_lap_numbers = stint_laps['LAP_NUMBER'].values
            
#             # Filter telemetry for this car and stint laps using EXACT column names
#             car_telemetry = telemetry_data[
#                 (telemetry_data['vehicle_id'].str.contains(str(car_number))) &
#                 (telemetry_data['lap'].isin(stint_lap_numbers))
#             ]
            
#             if not car_telemetry.empty:
#                 # Estimate lateral forces from lateral acceleration using EXACT column names
#                 if 'accy_can' in car_telemetry.columns:
#                     factors['estimated_lateral_force'] = car_telemetry['accy_can'].abs().mean()
                
#                 # Estimate braking from longitudinal acceleration using EXACT column names
#                 if 'accx_can' in car_telemetry.columns:
#                     braking_events = car_telemetry[car_telemetry['accx_can'] < -0.5]
#                     factors['estimated_braking_force'] = len(braking_events) / len(car_telemetry) if len(car_telemetry) > 0 else 0.3
                
#                 # Gear usage efficiency using EXACT column names
#                 if 'gear' in car_telemetry.columns:
#                     optimal_gear_ratio = (car_telemetry['gear'].between(3, 5)).mean()
#                     factors['gear_usage_efficiency'] = optimal_gear_ratio
                
#                 # Driving aggressiveness (speed variance + acceleration patterns)
#                 if 'speed' in car_telemetry.columns:
#                     speed_variance = car_telemetry['speed'].var()
#                     factors['driving_aggressiveness'] = min(1.0, speed_variance / 1000.0)
                    
#         return factors

#     def _calculate_degradation_targets(self, current_stint: pd.DataFrame, next_stint: pd.DataFrame) -> Dict[str, float]:
#         """Calculate actual degradation observed between stints using EXACT column names"""
#         targets = {}
        
#         # Sector degradation (time increase per lap) using EXACT column names
#         for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
#             if sector in current_stint.columns and sector in next_stint.columns:
#                 current_avg = pd.to_numeric(current_stint[sector], errors='coerce').mean()
#                 next_avg = pd.to_numeric(next_stint[sector], errors='coerce').mean()
#                 degradation_per_lap = (next_avg - current_avg) / len(next_stint)
#                 targets[f'degradation_s{i}'] = max(0.001, min(0.5, degradation_per_lap))
#             else:
#                 targets[f'degradation_s{i}'] = 0.05
        
#         # Overall grip loss rate using EXACT column names
#         current_avg_time = current_stint['LAP_TIME_SECONDS'].mean()
#         next_avg_time = next_stint['LAP_TIME_SECONDS'].mean()
#         grip_loss = (next_avg_time - current_avg_time) / len(next_stint)
#         targets['grip_loss_rate'] = max(0.001, min(1.0, grip_loss))
        
#         return targets

#     def _linear_trend_analysis(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
#         """Calculate linear trend slope and R² value"""
#         if len(x) < 2:
#             return 0.0, 0.0
#         try:
#             slope = np.polyfit(x, y, 1)[0]
#             correlation = np.corrcoef(x, y)[0, 1]
#             r_squared = correlation ** 2 if not np.isnan(correlation) else 0.0
#             return slope, r_squared
#         except:
#             return 0.0, 0.0

#     def _get_track_abrasiveness(self, track_name: str) -> float:
#         """Estimate track abrasiveness based on actual track names"""
#         high_abrasion = ['barber', 'sonoma', 'sebring']
#         medium_abrasion = ['cota', 'road_america', 'virginia']
        
#         track_lower = track_name.lower()
#         if any(track in track_lower for track in high_abrasion):
#             return 0.8
#         elif any(track in track_lower for track in medium_abrasion):
#             return 0.5
#         else:
#             return 0.6

#     def _get_track_length_factor(self, track_name: str) -> float:
#         """Normalize by track length (simplified)"""
#         long_tracks = ['road_america', 'cota']
#         short_tracks = ['barber', 'sonoma']
        
#         track_lower = track_name.lower()
#         if any(track in track_lower for track in long_tracks):
#             return 1.2
#         elif any(track in track_lower for track in short_tracks):
#             return 0.8
#         else:
#             return 1.0

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

#     def predict_tire_degradation(self, features: Dict[str, float]) -> Dict[str, float]:
#         """Predict tire degradation rates for given features"""
#         try:
#             if not self.feature_columns:
#                 return self._get_fallback_degradation()
                
#             # Ensure all features are present
#             feature_vector = [features.get(col, 0.0) for col in self.feature_columns]
#             X = np.array(feature_vector).reshape(1, -1)
#             X_scaled = self.scaler.transform(X)
            
#             predictions = self.model.predict(X_scaled)[0]
#             return dict(zip(self.target_columns, predictions))
            
#         except Exception:
#             return self._get_fallback_degradation()

#     def _get_fallback_degradation(self) -> Dict[str, float]:
#         """Fallback degradation rates when model is unavailable"""
#         return {
#             'degradation_s1': 0.05,
#             'degradation_s2': 0.05, 
#             'degradation_s3': 0.05,
#             'grip_loss_rate': 0.1
#         }

#     def estimate_optimal_stint_length(self, features: Dict[str, float], performance_threshold: float = 0.2) -> int:
#         """Estimate optimal stint length before tire performance drops below threshold"""
#         try:
#             degradation_rates = self.predict_tire_degradation(features)
#             avg_degradation = np.mean([
#                 degradation_rates['degradation_s1'],
#                 degradation_rates['degradation_s2'], 
#                 degradation_rates['degradation_s3']
#             ])
            
#             if avg_degradation <= 0:
#                 return 20
                
#             optimal_laps = int(performance_threshold / avg_degradation)
#             return max(5, min(30, optimal_laps))
            
#         except Exception:
#             return 15

#     def save_model(self, filepath: str):
#         """Save trained model to file"""
#         joblib.dump({
#             'model': self.model,
#             'scaler': self.scaler,
#             'feature_columns': self.feature_columns,
#             'target_columns': self.target_columns
#         }, filepath)

#     def load_model(self, filepath: str):
#         """Load trained model from file"""
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
# from typing import Dict, List, Tuple

# class TireModelTrainer:
#     def __init__(self):
#         self.model = MultiOutputRegressor(
#             RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#         )
#         self.scaler = StandardScaler()
#         self.feature_columns = []
#         self.target_columns = ['degradation_s1', 'degradation_s2', 'degradation_s3', 'grip_loss_rate']
        
#         # Define expected data structures based on your schema
#         self.required_pit_cols = ['NUMBER', 'LAP_NUMBER', 'LAP_TIME', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']
#         self.required_telemetry_cols = ['vehicle_id', 'lap', 'accx_can', 'accy_can', 'speed']
#         self.required_weather_cols = ['TIME_UTC_SECONDS', 'TRACK_TEMP', 'AIR_TEMP']

#     def train(self, processed_data: Dict[str, Dict[str, pd.DataFrame]]) -> dict:
#         """
#         Train tire degradation model using structured data from Firebase loader
#         """
#         try:
#             # Validate input data structure
#             if not isinstance(processed_data, dict) or len(processed_data) < 2:
#                 return {'error': 'Insufficient tracks for tire model', 'status': 'error'}

#             features_list = []
#             targets_list = []

#             # Process each track's data
#             for track_name, track_data in processed_data.items():
#                 if not self._validate_track_data(track_data):
#                     continue
                    
#                 track_features, track_targets = self._extract_track_tire_features(track_data, track_name)
#                 if not track_features.empty and not track_targets.empty:
#                     features_list.append(track_features)
#                     targets_list.append(track_targets)

#             if not features_list:
#                 return {'error': 'No valid tire training data extracted', 'status': 'error'}

#             # Combine all track data
#             X = pd.concat(features_list, ignore_index=True)
#             y = pd.concat(targets_list, ignore_index=True)[self.target_columns]

#             if X.empty or y.empty:
#                 return {'error': 'Empty feature or target matrices', 'status': 'error'}

#             # Clean data
#             valid_mask = ~X.isna().any(axis=1) & ~y.isna().any(axis=1)
#             X = X[valid_mask]
#             y = y[valid_mask]

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
            
#             # Calculate average feature importance across all outputs
#             avg_feature_importance = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
#             feature_importance = dict(zip(self.feature_columns, avg_feature_importance))

#             return {
#                 'model': self,
#                 'features': self.feature_columns,
#                 'targets': self.target_columns,
#                 'train_score': train_score,
#                 'test_score': test_score,
#                 'feature_importance': feature_importance,
#                 'training_samples': len(X),
#                 'status': 'success'
#             }
            
#         except Exception as e:
#             return {'error': f'Training failed: {str(e)}', 'status': 'error'}

#     def _validate_track_data(self, track_data: Dict[str, pd.DataFrame]) -> bool:
#         """Validate that track data has required components for tire analysis"""
#         pit_data = track_data.get('pit_data', pd.DataFrame())
#         telemetry_data = track_data.get('telemetry_data', pd.DataFrame())
        
#         if pit_data.empty:
#             return False
            
#         # Check for required columns
#         missing_pit = [col for col in self.required_pit_cols if col not in pit_data.columns]
        
#         return len(missing_pit) == 0

#     def _extract_track_tire_features(self, track_data: Dict[str, pd.DataFrame], track_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         """
#         Extract tire degradation features for all cars in a track
#         """
#         pit_data = track_data['pit_data']
#         telemetry_data = track_data.get('telemetry_data', pd.DataFrame())
#         weather_data = track_data.get('weather_data', pd.DataFrame())

#         features_list = []
#         targets_list = []

#         # Process each car's stint patterns
#         for car_number in pit_data['NUMBER'].unique():
#             car_laps = pit_data[pit_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
#             if len(car_laps) < 5:  # Need sufficient laps for degradation analysis
#                 continue

#             # Analyze stint patterns (groups of consecutive laps)
#             stint_features, stint_targets = self._analyze_car_stints(car_laps, telemetry_data, weather_data, track_name)
            
#             if not stint_features.empty and not stint_targets.empty:
#                 features_list.append(stint_features)
#                 targets_list.append(stint_targets)

#         if features_list and targets_list:
#             return pd.concat(features_list, ignore_index=True), pd.concat(targets_list, ignore_index=True)
#         return pd.DataFrame(), pd.DataFrame()

#     def _analyze_car_stints(self, car_laps: pd.DataFrame, telemetry_data: pd.DataFrame, 
#                            weather_data: pd.DataFrame, track_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         """
#         Analyze tire degradation across different stints for a single car
#         """
#         features = []
#         targets = []
        
#         # Convert lap times to seconds for analysis
#         car_laps = car_laps.copy()
#         car_laps['LAP_TIME_SECONDS'] = car_laps['LAP_TIME'].apply(self._lap_time_to_seconds)
        
#         # Use a sliding window to analyze degradation patterns
#         window_size = min(5, len(car_laps) - 1)
        
#         for i in range(len(car_laps) - window_size):
#             current_stint = car_laps.iloc[i:i + window_size]
#             next_stint = car_laps.iloc[i + window_size:min(i + window_size * 2, len(car_laps))]
            
#             if len(next_stint) < 2:  # Need at least 2 laps for target calculation
#                 continue
                
#             # Extract features from current stint
#             stint_features = self._calculate_stint_features(current_stint, telemetry_data, weather_data, track_name)
            
#             # Calculate degradation targets from next stint
#             degradation_targets = self._calculate_degradation_targets(current_stint, next_stint)
            
#             features.append(pd.DataFrame([stint_features]))
#             targets.append(pd.DataFrame([degradation_targets]))
        
#         if features and targets:
#             return pd.concat(features, ignore_index=True), pd.concat(targets, ignore_index=True)
#         return pd.DataFrame(), pd.DataFrame()

#     def _calculate_stint_features(self, stint_laps: pd.DataFrame, telemetry_data: pd.DataFrame,
#                                 weather_data: pd.DataFrame, track_name: str) -> Dict[str, float]:
#         """Calculate tire degradation features from a stint"""
#         features = {}
        
#         # Lap time degradation analysis
#         lap_times = stint_laps['LAP_TIME_SECONDS'].values
#         lap_numbers = stint_laps['LAP_NUMBER'].values
        
#         # Linear degradation trend
#         if len(lap_times) > 1:
#             time_slope, time_r2 = self._linear_trend_analysis(lap_numbers, lap_times)
#             features['lap_time_degradation_slope'] = max(0.0, time_slope)  # Positive = degradation
#             features['lap_time_consistency'] = time_r2
#         else:
#             features['lap_time_degradation_slope'] = 0.1
#             features['lap_time_consistency'] = 0.0
            
#         # Sector-specific degradation
#         for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
#             if sector in stint_laps.columns:
#                 sector_times = pd.to_numeric(stint_laps[sector], errors='coerce').fillna(0).values
#                 if len(sector_times) > 1:
#                     sector_slope, _ = self._linear_trend_analysis(lap_numbers, sector_times)
#                     features[f'sector_{i}_degradation_slope'] = max(0.0, sector_slope)
#                 else:
#                     features[f'sector_{i}_degradation_slope'] = 0.05
#             else:
#                 features[f'sector_{i}_degradation_slope'] = 0.05
        
#         # Performance metrics
#         features['lap_time_variance'] = np.var(lap_times) if len(lap_times) > 1 else 1.0
#         features['performance_consistency'] = 1.0 / (1.0 + features['lap_time_variance'])
        
#         # Track and condition factors
#         features.update(self._calculate_track_conditions(stint_laps, weather_data, track_name))
        
#         # Driving style factors (from telemetry if available)
#         features.update(self._calculate_driving_factors(stint_laps, telemetry_data))
        
#         # Stint characteristics
#         features['stint_length'] = len(stint_laps)
#         features['cumulative_laps'] = stint_laps['LAP_NUMBER'].max()
        
#         return features

#     def _calculate_track_conditions(self, stint_laps: pd.DataFrame, weather_data: pd.DataFrame, 
#                                   track_name: str) -> Dict[str, float]:
#         """Calculate track and weather conditions affecting tire wear"""
#         conditions = {}
        
#         # Track characteristics
#         conditions['track_abrasiveness'] = self._get_track_abrasiveness(track_name)
#         conditions['track_length_factor'] = self._get_track_length_factor(track_name)
        
#         # Weather conditions
#         if not weather_data.empty:
#             # Use average weather during stint period (simplified)
#             conditions['avg_track_temp'] = weather_data['TRACK_TEMP'].mean() if 'TRACK_TEMP' in weather_data.columns else 35.0
#             conditions['track_temp_variance'] = weather_data['TRACK_TEMP'].var() if 'TRACK_TEMP' in weather_data.columns else 5.0
#             conditions['avg_air_temp'] = weather_data['AIR_TEMP'].mean() if 'AIR_TEMP' in weather_data.columns else 25.0
#         else:
#             conditions['avg_track_temp'] = 35.0
#             conditions['track_temp_variance'] = 5.0
#             conditions['avg_air_temp'] = 25.0
            
#         return conditions

#     def _calculate_driving_factors(self, stint_laps: pd.DataFrame, telemetry_data: pd.DataFrame) -> Dict[str, float]:
#         """Calculate driving style factors from telemetry data"""
#         factors = {
#             'estimated_lateral_force': 0.5,
#             'estimated_braking_force': 0.3,
#             'driving_aggressiveness': 0.6
#         }
        
#         if not telemetry_data.empty:
#             car_number = stint_laps['NUMBER'].iloc[0]
#             stint_lap_numbers = stint_laps['LAP_NUMBER'].values
            
#             # Filter telemetry for this car and stint laps
#             car_telemetry = telemetry_data[
#                 (telemetry_data['vehicle_id'].str.contains(str(car_number))) &
#                 (telemetry_data['lap'].isin(stint_lap_numbers))
#             ]
            
#             if not car_telemetry.empty:
#                 # Estimate lateral forces from lateral acceleration
#                 if 'accy_can' in car_telemetry.columns:
#                     factors['estimated_lateral_force'] = car_telemetry['accy_can'].abs().mean()
                
#                 # Estimate braking from longitudinal acceleration
#                 if 'accx_can' in car_telemetry.columns:
#                     braking_events = car_telemetry[car_telemetry['accx_can'] < -0.5]
#                     factors['estimated_braking_force'] = len(braking_events) / len(car_telemetry) if len(car_telemetry) > 0 else 0.3
                
#                 # Driving aggressiveness (speed variance + acceleration patterns)
#                 if 'speed' in car_telemetry.columns:
#                     speed_variance = car_telemetry['speed'].var()
#                     factors['driving_aggressiveness'] = min(1.0, speed_variance / 1000.0)  # Normalized
                    
#         return factors

#     def _calculate_degradation_targets(self, current_stint: pd.DataFrame, next_stint: pd.DataFrame) -> Dict[str, float]:
#         """Calculate actual degradation observed between stints"""
#         targets = {}
        
#         # Sector degradation (time increase per lap)
#         for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
#             if sector in current_stint.columns and sector in next_stint.columns:
#                 current_avg = pd.to_numeric(current_stint[sector], errors='coerce').mean()
#                 next_avg = pd.to_numeric(next_stint[sector], errors='coerce').mean()
#                 degradation_per_lap = (next_avg - current_avg) / len(next_stint)
#                 targets[f'degradation_s{i}'] = max(0.001, min(0.5, degradation_per_lap))
#             else:
#                 targets[f'degradation_s{i}'] = 0.05
        
#         # Overall grip loss rate
#         current_avg_time = current_stint['LAP_TIME_SECONDS'].mean()
#         next_avg_time = next_stint['LAP_TIME_SECONDS'].mean()
#         grip_loss = (next_avg_time - current_avg_time) / len(next_stint)
#         targets['grip_loss_rate'] = max(0.001, min(1.0, grip_loss))
        
#         return targets

#     def _linear_trend_analysis(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
#         """Calculate linear trend slope and R² value"""
#         if len(x) < 2:
#             return 0.0, 0.0
#         try:
#             slope = np.polyfit(x, y, 1)[0]
#             correlation = np.corrcoef(x, y)[0, 1]
#             r_squared = correlation ** 2 if not np.isnan(correlation) else 0.0
#             return slope, r_squared
#         except:
#             return 0.0, 0.0

#     def _get_track_abrasiveness(self, track_name: str) -> float:
#         """Estimate track abrasiveness based on track characteristics"""
#         high_abrasion = ['sebring', 'sonoma', 'interlagos']
#         medium_abrasion = ['cota', 'silverstone', 'suzuka']
#         low_abrasion = ['monaco', 'hungaroring', 'sochi']
        
#         track_lower = track_name.lower()
#         if any(track in track_lower for track in high_abrasion):
#             return 0.8
#         elif any(track in track_lower for track in medium_abrasion):
#             return 0.5
#         elif any(track in track_lower for track in low_abrasion):
#             return 0.3
#         else:
#             return 0.6  # Default medium

#     def _get_track_length_factor(self, track_name: str) -> float:
#         """Normalize by track length (simplified)"""
#         long_tracks = ['spa', 'suzuka', 'silverstone']
#         short_tracks = ['monaco', 'hungaroring', 'sochi']
        
#         track_lower = track_name.lower()
#         if any(track in track_lower for track in long_tracks):
#             return 1.2
#         elif any(track in track_lower for track in short_tracks):
#             return 0.8
#         else:
#             return 1.0

#     def _lap_time_to_seconds(self, lap_time: str) -> float:
#         """Convert lap time string to seconds"""
#         try:
#             if ':' in lap_time:
#                 parts = lap_time.split(':')
#                 if len(parts) == 2:
#                     return float(parts[0]) * 60 + float(parts[1])
#             return float(lap_time)
#         except:
#             return 60.0

#     def predict_tire_degradation(self, features: Dict[str, float]) -> Dict[str, float]:
#         """Predict tire degradation rates for given features"""
#         try:
#             if not self.feature_columns:
#                 return self._get_fallback_degradation()
                
#             # Ensure all features are present
#             feature_vector = [features.get(col, 0.0) for col in self.feature_columns]
#             X = np.array(feature_vector).reshape(1, -1)
#             X_scaled = self.scaler.transform(X)
            
#             predictions = self.model.predict(X_scaled)[0]
#             return dict(zip(self.target_columns, predictions))
            
#         except Exception:
#             return self._get_fallback_degradation()

#     def _get_fallback_degradation(self) -> Dict[str, float]:
#         """Fallback degradation rates when model is unavailable"""
#         return {
#             'degradation_s1': 0.05,
#             'degradation_s2': 0.05, 
#             'degradation_s3': 0.05,
#             'grip_loss_rate': 0.1
#         }

#     def estimate_optimal_stint_length(self, features: Dict[str, float], performance_threshold: float = 0.2) -> int:
#         """Estimate optimal stint length before tire performance drops below threshold"""
#         try:
#             degradation_rates = self.predict_tire_degradation(features)
#             avg_degradation = np.mean([
#                 degradation_rates['degradation_s1'],
#                 degradation_rates['degradation_s2'], 
#                 degradation_rates['degradation_s3']
#             ])
            
#             if avg_degradation <= 0:
#                 return 20  # Default stint length
                
#             optimal_laps = int(performance_threshold / avg_degradation)
#             return max(5, min(30, optimal_laps))
            
#         except Exception:
#             return 15  # Fallback stint length

#     def save_model(self, filepath: str):
#         """Save trained model to file"""
#         joblib.dump({
#             'model': self.model,
#             'scaler': self.scaler,
#             'feature_columns': self.feature_columns,
#             'target_columns': self.target_columns
#         }, filepath)

#     def load_model(self, filepath: str):
#         """Load trained model from file"""
#         data = joblib.load(filepath)
#         self.model = data['model']
#         self.scaler = data['scaler']
#         self.feature_columns = data['feature_columns']
#         self.target_columns = data['target_columns']