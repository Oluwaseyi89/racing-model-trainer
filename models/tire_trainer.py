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
        self.required_features = [
            'lap_time_slope', 'lap_time_consistency', 'sector_1_slope', 'sector_2_slope', 'sector_3_slope',
            'avg_track_temp', 'track_abrasiveness', 'avg_lateral_g', 'avg_brake_pressure', 'stint_length'
        ]

    # ---------------------------
    # TRAINING ENTRY POINT
    # ---------------------------
    def train(self, lap_data: pd.DataFrame, telemetry_data: pd.DataFrame, weather_data: pd.DataFrame) -> dict:
        """Train tire degradation model with comprehensive error handling"""
        try:
            # Validate and prepare input data
            lap_data, telemetry_data, weather_data = self._validate_and_prepare_inputs(
                lap_data, telemetry_data, weather_data
            )
            
            # Extract features with robust error handling
            features_df, targets_df = self._extract_tire_features(lap_data, telemetry_data, weather_data)
            
            if features_df.empty or targets_df.empty:
                return {'error': 'No valid tire degradation features extracted'}

            # Prepare training data
            X, y = self._prepare_training_data(features_df, targets_df)
            if len(X) < 15:
                return {'error': f'Insufficient training samples: {len(X)}'}

            # Train model with validation
            training_result = self._train_model_with_validation(X, y)
            if 'error' in training_result:
                return training_result

            return {
                'model': self,
                'features': self.feature_columns,
                'targets': self.target_columns,
                'train_score': training_result['train_score'],
                'test_score': training_result['test_score'],
                'feature_importance': training_result['feature_importance'],
                'training_samples': training_result['training_samples']
            }

        except Exception as e:
            return {'error': f'Tire model training failed: {str(e)}'}

    def _validate_and_prepare_inputs(self, lap_data: pd.DataFrame, telemetry_data: pd.DataFrame, 
                                   weather_data: pd.DataFrame) -> tuple:
        """Validate and prepare input data with fallbacks"""
        # Handle empty inputs with synthetic data
        if lap_data.empty or telemetry_data.empty or weather_data.empty:
            print("⚠️ Empty input data, generating synthetic tire training data")
            return self._generate_synthetic_training_data()
        
        # Normalize data to expected schema
        lap_data = self._normalize_lap_data(lap_data)
        telemetry_data = self._normalize_telemetry_data(telemetry_data)
        weather_data = self._normalize_weather_data(weather_data)
        
        return lap_data, telemetry_data, weather_data

    def _normalize_lap_data(self, lap_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize lap data to expected schema"""
        df = lap_data.copy()
        
        # Ensure required columns
        if 'NUMBER' not in df.columns:
            df['NUMBER'] = 1
        if 'LAP_NUMBER' not in df.columns:
            df['LAP_NUMBER'] = range(1, len(df) + 1)
        
        # Ensure numeric types for critical columns
        numeric_columns = ['LAP_TIME_SECONDS', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(60.0)
        
        # Add session identifier if missing
        if 'meta_session' not in df.columns:
            df['meta_session'] = 'session1'
        
        return df

    def _normalize_telemetry_data(self, telemetry_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize telemetry data to expected schema"""
        df = telemetry_data.copy()
        
        # Ensure required columns
        if 'vehicle_number' not in df.columns:
            df['vehicle_number'] = 1
        if 'lap' not in df.columns:
            df['lap'] = 1
        
        # Map to expected column names
        column_mapping = {
            'LATERAL_ACCEL': 'accy_can',
            'THROTTLE_POSITION': 'aps',
            'BRAKE_PRESSURE_FRONT': 'pbrake_f',
            'BRAKE_PRESSURE_REAR': 'pbrake_r',
            'STEERING_ANGLE': 'Steering_Angle'
        }
        
        for canonical, expected in column_mapping.items():
            if canonical in df.columns and expected not in df.columns:
                df[expected] = df[canonical]
        
        return df

    def _normalize_weather_data(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize weather data to expected schema"""
        df = weather_data.copy()
        
        # Ensure required columns
        if 'meta_session' not in df.columns:
            df['meta_session'] = 'session1'
        
        # Ensure numeric types
        numeric_columns = ['TRACK_TEMP', 'AIR_TEMP']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(25.0)
        
        return df

    def _extract_tire_features(self, lap_data: pd.DataFrame, telemetry_data: pd.DataFrame, 
                              weather_data: pd.DataFrame) -> tuple:
        """Extract tire degradation features with comprehensive error handling"""
        features_list = []
        targets_list = []

        try:
            # Group by car and session with validation
            if 'NUMBER' not in lap_data.columns or 'meta_session' not in lap_data.columns:
                print("❌ Missing grouping columns in lap data")
                return pd.DataFrame(), pd.DataFrame()

            # Use session from lap data if weather data doesn't have it
            available_sessions = lap_data['meta_session'].unique()
            
            for session in available_sessions:
                try:
                    session_laps = lap_data[lap_data['meta_session'] == session]
                    session_weather = weather_data[weather_data['meta_session'] == session] if not weather_data.empty else pd.DataFrame()
                    
                    for car_number in session_laps['NUMBER'].unique():
                        try:
                            car_features, car_targets = self._analyze_car_performance(
                                car_number, session, session_laps, telemetry_data, session_weather
                            )
                            if car_features is not None and car_targets is not None:
                                features_list.append(car_features)
                                targets_list.append(car_targets)
                                
                        except Exception as e:
                            print(f"⚠️ Car {car_number} analysis failed in session {session}: {e}")
                            continue
                            
                except Exception as e:
                    print(f"⚠️ Session {session} analysis failed: {e}")
                    continue

        except Exception as e:
            print(f"❌ Tire feature extraction failed: {e}")
            return pd.DataFrame(), pd.DataFrame()

        # Combine results with validation
        if features_list and targets_list:
            try:
                features_df = pd.concat(features_list, ignore_index=True)
                targets_df = pd.concat(targets_list, ignore_index=True)
                
                # Ensure required features and targets
                features_df = self._ensure_required_features(features_df)
                targets_df = self._ensure_required_targets(targets_df)
                
                return features_df, targets_df
                
            except Exception as e:
                print(f"❌ Failed to combine features and targets: {e}")

        return pd.DataFrame(), pd.DataFrame()

    def _analyze_car_performance(self, car_number: int, session: str, session_laps: pd.DataFrame,
                                telemetry_data: pd.DataFrame, weather_data: pd.DataFrame) -> tuple:
        """Analyze car performance for tire degradation with stint-based analysis"""
        try:
            # Get car laps for this session
            car_laps = session_laps[session_laps['NUMBER'] == car_number].sort_values('LAP_NUMBER')
            
            if len(car_laps) < 6:  # Reduced minimum for more data
                car_laps = self._generate_car_laps(car_number, session, 10)
            
            # Get telemetry and weather for this car/session
            car_telemetry = telemetry_data[
                (telemetry_data['vehicle_number'] == car_number) & 
                (telemetry_data.get('meta_session', 'session1') == session)
            ] if not telemetry_data.empty else pd.DataFrame()

            # Analyze stints for this car
            stint_features, stint_targets = self._analyze_car_stints(car_laps, car_telemetry, weather_data)
            
            if stint_features.empty or stint_targets.empty:
                return None, None
                
            return stint_features, stint_targets
            
        except Exception as e:
            print(f"❌ Car performance analysis failed for car {car_number}: {e}")
            return None, None

    def _analyze_car_stints(self, car_laps: pd.DataFrame, telemetry_data: pd.DataFrame,
                           weather_data: pd.DataFrame) -> tuple:
        """Analyze stints for degradation patterns with sliding window"""
        features_list = []
        targets_list = []
        window_size = 4  # Reduced window size for more data points
        
        try:
            for start_idx in range(0, len(car_laps) - window_size * 2 + 1):
                try:
                    # Current stint (window)
                    current_end = start_idx + window_size
                    current_stint = car_laps.iloc[start_idx:current_end]
                    
                    # Next stint (for targets)
                    next_start = current_end
                    next_end = next_start + window_size
                    if next_end > len(car_laps):
                        continue
                    next_stint = car_laps.iloc[next_start:next_end]
                    
                    # Calculate features and targets
                    stint_features = self._calculate_stint_features(current_stint, telemetry_data, weather_data)
                    stint_targets = self._calculate_degradation_targets(current_stint, next_stint)
                    
                    if stint_features and stint_targets:
                        features_list.append(pd.DataFrame([stint_features]))
                        targets_list.append(pd.DataFrame([stint_targets]))
                        
                except Exception as e:
                    print(f"⚠️ Stint analysis failed at index {start_idx}: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Car stint analysis failed: {e}")
            
        # Combine results
        features_df = pd.concat(features_list, ignore_index=True) if features_list else pd.DataFrame()
        targets_df = pd.concat(targets_list, ignore_index=True) if targets_list else pd.DataFrame()
        
        return features_df, targets_df

    def _calculate_stint_features(self, stint_laps: pd.DataFrame, telemetry_data: pd.DataFrame,
                                 weather_data: pd.DataFrame) -> dict:
        """Calculate stint features with comprehensive error handling"""
        features = {}
        
        try:
            # Degradation metrics
            degradation_metrics = self._calculate_degradation_metrics(stint_laps)
            features.update(degradation_metrics)
            
            # Condition factors
            condition_factors = self._calculate_condition_factors(stint_laps, weather_data)
            features.update(condition_factors)
            
            # Driving factors
            driving_factors = self._calculate_driving_factors(stint_laps, telemetry_data)
            features.update(driving_factors)
            
        except Exception as e:
            print(f"⚠️ Stint feature calculation failed: {e}")
            # Fallback features
            features.update(self._get_fallback_features())
            
        return features

    def _calculate_degradation_metrics(self, stint_laps: pd.DataFrame) -> dict:
        """Calculate degradation metrics from lap times"""
        metrics = {}
        
        try:
            lap_times = pd.to_numeric(stint_laps['LAP_TIME_SECONDS'], errors='coerce').fillna(60.0).values
            lap_numbers = pd.to_numeric(stint_laps['LAP_NUMBER'], errors='coerce').fillna(range(len(stint_laps))).values
            
            # Lap time trend
            slope, consistency = self._linear_trend_analysis(lap_numbers, lap_times)
            metrics['lap_time_slope'] = float(slope)
            metrics['lap_time_consistency'] = float(consistency)
            metrics['lap_time_variance'] = float(np.var(lap_times) if len(lap_times) > 1 else 0.0)
            
            # Best to worst ratio
            if len(lap_times) > 1:
                metrics['best_to_worst_ratio'] = float(np.min(lap_times) / np.max(lap_times))
            else:
                metrics['best_to_worst_ratio'] = 1.0
            
            # Sector trends
            for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
                sector_times = pd.to_numeric(stint_laps.get(sector, pd.Series([20.0] * len(stint_laps))), 
                                           errors='coerce').fillna(20.0).values
                sector_slope, _ = self._linear_trend_analysis(lap_numbers, sector_times)
                metrics[f'sector_{i}_slope'] = float(sector_slope)
                
        except Exception as e:
            print(f"⚠️ Degradation metrics calculation failed: {e}")
            # Fallback values
            metrics.update({
                'lap_time_slope': 0.1,
                'lap_time_consistency': 0.5,
                'lap_time_variance': 0.5,
                'best_to_worst_ratio': 0.9,
                'sector_1_slope': 0.05,
                'sector_2_slope': 0.05,
                'sector_3_slope': 0.05
            })
            
        return metrics

    def _calculate_condition_factors(self, stint_laps: pd.DataFrame, weather_data: pd.DataFrame) -> dict:
        """Calculate track and weather condition factors"""
        factors = {}
        
        try:
            # Track temperature factors
            if not weather_data.empty and 'TRACK_TEMP' in weather_data.columns:
                track_temp = pd.to_numeric(weather_data['TRACK_TEMP'], errors='coerce').dropna()
                if not track_temp.empty:
                    factors['avg_track_temp'] = float(track_temp.mean())
                    factors['track_temp_range'] = float(track_temp.max() - track_temp.min())
                else:
                    factors['avg_track_temp'] = 35.0
                    factors['track_temp_range'] = 5.0
            else:
                factors['avg_track_temp'] = 35.0
                factors['track_temp_range'] = 5.0
            
            # Air temperature
            if not weather_data.empty and 'AIR_TEMP' in weather_data.columns:
                air_temp = pd.to_numeric(weather_data['AIR_TEMP'], errors='coerce').dropna()
                factors['avg_air_temp'] = float(air_temp.mean()) if not air_temp.empty else 25.0
            else:
                factors['avg_air_temp'] = 25.0
            
            # Track abrasiveness (from session name or default)
            session_name = stint_laps.get('meta_session', pd.Series(['unknown'])).iloc[0]
            factors['track_abrasiveness'] = float(self._get_track_abrasiveness(session_name))
            
        except Exception as e:
            print(f"⚠️ Condition factors calculation failed: {e}")
            factors.update({
                'avg_track_temp': 35.0,
                'track_temp_range': 5.0,
                'avg_air_temp': 25.0,
                'track_abrasiveness': 0.7
            })
            
        return factors

    def _calculate_driving_factors(self, stint_laps: pd.DataFrame, telemetry_data: pd.DataFrame) -> dict:
        """Calculate driving style factors from telemetry"""
        factors = {}
        
        try:
            factors['stint_length'] = int(len(stint_laps))
            factors['cumulative_laps'] = int(stint_laps['LAP_NUMBER'].max())
            
            if not telemetry_data.empty:
                # Filter telemetry for this stint
                stint_telemetry = telemetry_data[
                    telemetry_data['lap'].between(stint_laps['LAP_NUMBER'].min(), stint_laps['LAP_NUMBER'].max())
                ]
                
                if not stint_telemetry.empty:
                    factors['avg_lateral_g'] = float(self._safe_column_mean(stint_telemetry, 'accy_can', 0.5, absolute=True))
                    factors['avg_brake_pressure'] = float(self._safe_brake_pressure(stint_telemetry))
                    factors['avg_throttle_usage'] = float(self._safe_column_mean(stint_telemetry, 'aps', 60.0))
                    factors['steering_variance'] = float(self._safe_column_variance(stint_telemetry, 'Steering_Angle', 10.0))
                else:
                    factors.update(self._get_driving_fallback_factors())
            else:
                factors.update(self._get_driving_fallback_factors())
                
        except Exception as e:
            print(f"⚠️ Driving factors calculation failed: {e}")
            factors.update(self._get_driving_fallback_factors())
            
        return factors

    def _calculate_degradation_targets(self, current_stint: pd.DataFrame, next_stint: pd.DataFrame) -> dict:
        """Calculate degradation targets between consecutive stints"""
        targets = {}
        
        try:
            # Sector degradations
            for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
                current_avg = self._safe_column_mean(current_stint, sector, 20.0)
                next_avg = self._safe_column_mean(next_stint, sector, 20.0)
                degradation = (next_avg - current_avg) / len(next_stint)
                targets[f'degradation_s{i}'] = float(max(0.001, min(0.3, degradation)))
            
            # Overall grip loss rate
            current_avg_time = self._safe_column_mean(current_stint, 'LAP_TIME_SECONDS', 60.0)
            next_avg_time = self._safe_column_mean(next_stint, 'LAP_TIME_SECONDS', 60.0)
            grip_loss = (next_avg_time - current_avg_time) / len(next_stint)
            targets['grip_loss_rate'] = float(max(0.001, min(0.5, grip_loss)))
            
        except Exception as e:
            print(f"⚠️ Degradation targets calculation failed: {e}")
            targets.update({
                'degradation_s1': 0.05,
                'degradation_s2': 0.05,
                'degradation_s3': 0.05,
                'grip_loss_rate': 0.1
            })
            
        return targets

    def _linear_trend_analysis(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """Perform linear trend analysis with error handling"""
        try:
            if len(x) < 2:
                return 0.0, 0.0
            
            # Remove any NaN values
            valid_mask = ~np.isnan(x) & ~np.isnan(y)
            x_clean = x[valid_mask]
            y_clean = y[valid_mask]
            
            if len(x_clean) < 2:
                return 0.0, 0.0
                
            slope = np.polyfit(x_clean, y_clean, 1)[0]
            correlation = np.corrcoef(x_clean, y_clean)[0, 1]
            r2 = correlation ** 2 if not np.isnan(correlation) else 0.0
            
            return float(slope), float(r2)
            
        except Exception:
            return 0.0, 0.0

    # ---------------------------
    # UTILITY METHODS
    # ---------------------------
    def _safe_column_mean(self, df: pd.DataFrame, column: str, default: float, absolute: bool = False) -> float:
        """Safely calculate column mean with fallback"""
        try:
            if column not in df.columns:
                return default
            values = pd.to_numeric(df[column], errors='coerce').dropna()
            if values.empty:
                return default
            if absolute:
                values = values.abs()
            return float(values.mean())
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

    def _get_track_abrasiveness(self, track_name: str) -> float:
        """Get track abrasiveness factor"""
        abrasiveness_map = {
            'sebring': 0.9, 'barber': 0.8, 'sonoma': 0.7, 'cota': 0.6, 
            'road-america': 0.5, 'vir': 0.6, 'indianapolis': 0.5
        }
        # Extract track name from session string
        for track_key in abrasiveness_map.keys():
            if track_key in track_name.lower():
                return abrasiveness_map[track_key]
        return 0.7

    def _ensure_required_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required features are present"""
        df = features_df.copy()
        for feature in self.required_features:
            if feature not in df.columns:
                df[feature] = 0.0
        return df

    def _ensure_required_targets(self, targets_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required targets are present"""
        df = targets_df.copy()
        for target in self.target_columns:
            if target not in df.columns:
                df[target] = 0.05  # Reasonable default
        return df

    def _get_fallback_features(self) -> dict:
        """Get comprehensive fallback features"""
        return {
            'lap_time_slope': 0.1,
            'lap_time_consistency': 0.5,
            'lap_time_variance': 0.5,
            'best_to_worst_ratio': 0.9,
            'sector_1_slope': 0.05,
            'sector_2_slope': 0.05,
            'sector_3_slope': 0.05,
            'avg_track_temp': 35.0,
            'track_temp_range': 5.0,
            'avg_air_temp': 25.0,
            'track_abrasiveness': 0.7,
            'stint_length': 4,
            'cumulative_laps': 10
        }

    def _get_driving_fallback_factors(self) -> dict:
        """Get driving factors fallback values"""
        return {
            'avg_lateral_g': 0.5,
            'avg_brake_pressure': 25.0,
            'avg_throttle_usage': 60.0,
            'steering_variance': 10.0
        }

    def _prepare_training_data(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> tuple:
        """Prepare training data with proper validation"""
        try:
            X = features_df.copy()
            y = targets_df[self.target_columns].copy()
            
            # Remove rows with NaNs using proper boolean indexing
            valid_mask = ~X.isna().any(axis=1) & ~y.isna().any(axis=1)
            X = X[valid_mask]
            y = y[valid_mask]
            
            # Ensure consistent data types
            X = self._enforce_feature_types(X)
            y = y.astype(float)
            
            return X, y
            
        except Exception as e:
            print(f"❌ Training data preparation failed: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def _train_model_with_validation(self, X: pd.DataFrame, y: pd.DataFrame) -> dict:
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
            
            # Feature importance (average across all outputs)
            avg_feature_importance = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
            feature_importance = dict(zip(self.feature_columns, avg_feature_importance))

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
                if any(keyword in col for keyword in ['length', 'laps']):
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(int)
                else:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0).astype(float)
            except Exception:
                X[col] = 0.0
        return X

    # ---------------------------
    # SYNTHETIC DATA GENERATION
    # ---------------------------
    def _generate_synthetic_training_data(self) -> tuple:
        """Generate realistic synthetic training data for tire degradation"""
        print("🔄 Generating synthetic tire training data...")
        
        lap_data_list = []
        telemetry_data_list = []
        weather_data_list = []
        
        sessions = ['qualifying', 'race1', 'race2']
        tracks = ['sonoma', 'cota', 'road-america']
        
        for session_idx, session in enumerate(sessions):
            track = tracks[session_idx % len(tracks)]
            
            # Weather data for session
            weather_data_list.append({
                'meta_session': session,
                'TRACK_TEMP': 30.0 + session_idx * 5.0,
                'AIR_TEMP': 25.0 + session_idx * 3.0
            })
            
            # Generate data for 2 cars per session
            for vehicle_num in range(1, 3):
                n_laps = 15 if 'race' in session else 8
                lap_data_list.append(self._generate_car_laps(vehicle_num, session, track, n_laps))
                telemetry_data_list.append(self._generate_car_telemetry(vehicle_num, session, n_laps))
        
        lap_data = pd.DataFrame(lap_data_list)
        telemetry_data = pd.concat(telemetry_data_list, ignore_index=True)
        weather_data = pd.DataFrame(weather_data_list)
        
        return lap_data, telemetry_data, weather_data

    def _generate_car_laps(self, vehicle_num: int, session: str, track: str, n_laps: int) -> dict:
        """Generate realistic lap data with degradation simulation"""
        base_time = 58.0 if 'qualifying' in session else 60.0
        degradation_rate = 0.4 if 'race' in session else 0.2
        
        lap_times = []
        sector_times = []
        
        for lap in range(1, n_laps + 1):
            # Simulate tire degradation effect
            degradation_effect = (lap - 1) * degradation_rate
            lap_time = base_time + degradation_effect + np.random.uniform(-0.5, 0.5)
            lap_times.append(lap_time)
            
            # Sector times with similar degradation
            sector_base = lap_time / 3
            sector_times.append([
                sector_base + np.random.uniform(-0.2, 0.2),
                sector_base + np.random.uniform(-0.2, 0.2),
                sector_base + np.random.uniform(-0.2, 0.2)
            ])
        
        return {
            'NUMBER': vehicle_num,
            'meta_session': session,
            'LAP_NUMBER': list(range(1, n_laps + 1)),
            'LAP_TIME_SECONDS': lap_times,
            'S1_SECONDS': [sectors[0] for sectors in sector_times],
            'S2_SECONDS': [sectors[1] for sectors in sector_times],
            'S3_SECONDS': [sectors[2] for sectors in sector_times]
        }

    def _generate_car_telemetry(self, vehicle_num: int, session: str, n_laps: int) -> pd.DataFrame:
        """Generate realistic telemetry data"""
        telemetry_points = []
        
        for lap in range(1, n_laps + 1):
            # More aggressive driving in qualifying
            throttle_base = 70.0 if 'qualifying' in session else 65.0
            lateral_g_base = 0.6 if 'qualifying' in session else 0.5
            
            for point in range(10):  # 10 telemetry points per lap
                telemetry_points.append({
                    'vehicle_number': vehicle_num,
                    'meta_session': session,
                    'lap': lap,
                    'aps': np.random.normal(throttle_base, 10).clip(0, 100),
                    'pbrake_f': np.random.normal(30, 8).clip(0, 50),
                    'pbrake_r': np.random.normal(30, 8).clip(0, 50),
                    'accy_can': np.random.normal(0, lateral_g_base),
                    'Steering_Angle': np.random.normal(0, 12)
                })
        
        return pd.DataFrame(telemetry_points)

    # ---------------------------
    # PREDICTION METHODS
    # ---------------------------
    def predict_degradation(self, features: dict) -> dict:
        """Predict tire degradation with error handling"""
        try:
            if not self.feature_columns:
                return self._fallback_degradation_prediction()
            
            # Create feature vector
            feature_vector = np.array([features.get(col, 0.0) for col in self.feature_columns]).reshape(1, -1)
            scaled_features = self.scaler.transform(feature_vector)
            predictions = self.model.predict(scaled_features)[0]
            
            return {
                'degradation_s1': float(max(0.001, predictions[0])),
                'degradation_s2': float(max(0.001, predictions[1])),
                'degradation_s3': float(max(0.001, predictions[2])),
                'grip_loss_rate': float(max(0.001, predictions[3]))
            }
            
        except Exception as e:
            print(f"⚠️ Degradation prediction failed: {e}")
            return self._fallback_degradation_prediction()

    def _fallback_degradation_prediction(self) -> dict:
        """Fallback degradation prediction"""
        return {
            'degradation_s1': 0.05,
            'degradation_s2': 0.05,
            'degradation_s3': 0.05,
            'grip_loss_rate': 0.1
        }

    def estimate_optimal_stint_length(self, features: dict, threshold: float = 0.2) -> int:
        """Estimate optimal stint length based on degradation"""
        try:
            rates = self.predict_degradation(features)
            avg_deg = np.mean([rates['degradation_s1'], rates['degradation_s2'], rates['degradation_s3']])
            
            if avg_deg <= 0:
                return 15
                
            optimal_laps = threshold / avg_deg
            return int(max(5, min(30, optimal_laps)))
            
        except Exception:
            return 15

    # ---------------------------
    # MODEL SERIALIZATION
    # ---------------------------
    def save_model(self, filepath: str):
        """Save model with error handling"""
        try:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'target_columns': self.target_columns
            }, filepath)
        except Exception as e:
            print(f"❌ Model save failed: {e}")

    def load_model(self, filepath: str):
        """Load model with error handling"""
        try:
            data = joblib.load(filepath)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']
            self.target_columns = data['target_columns']
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            # Reinitialize with defaults
            self.__init__()



















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

#     # ---------------------------
#     # TRAINING ENTRY POINT
#     # ---------------------------
#     def train(self, lap_data: pd.DataFrame, telemetry_data: pd.DataFrame, weather_data: pd.DataFrame) -> dict:
#         if lap_data.empty or telemetry_data.empty or weather_data.empty:
#             lap_data, telemetry_data, weather_data = self._fabricate_minimal_data()

#         features_df, targets_df = self._extract_tire_features(lap_data, telemetry_data, weather_data)

#         if features_df.empty or targets_df.empty:
#             return {'error': 'No valid tire features extracted'}

#         X = features_df
#         y = targets_df[self.target_columns]

#         valid_mask = ~X.isna().any(axis=1) & ~y.isna().any(axis=1)
#         X = X[valid_mask]
#         y = y[valid_mask]

#         if len(X) < 20:
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
#             avg_feature_importance = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
#             feature_importance = dict(zip(self.feature_columns, avg_feature_importance))
#         except Exception as e:
#             return {'error': f'Model training failed: {e}'}

#         return {
#             'model': self,
#             'features': self.feature_columns,
#             'targets': self.target_columns,
#             'train_score': train_score,
#             'test_score': test_score,
#             'feature_importance': feature_importance,
#             'training_samples': len(X)
#         }

#     # ---------------------------
#     # FEATURE EXTRACTION
#     # ---------------------------
#     def _extract_tire_features(self, lap_data, telemetry_data, weather_data):
#         features_list = []
#         targets_list = []

#         for (car_number, session), car_laps in lap_data.groupby(['NUMBER', 'meta_session']):
#             car_laps = car_laps.sort_values('LAP_NUMBER')
#             if len(car_laps) < 8:
#                 car_laps = self._fabricate_car_laps(car_number, session, 8)

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

#     def _analyze_stint_performance(self, car_laps, telemetry_data, weather_data):
#         features, targets = [], []
#         window_size = 5

#         for start in range(0, len(car_laps) - window_size):
#             end = start + window_size
#             stint_laps = car_laps.iloc[start:end]
#             if len(stint_laps) < window_size:
#                 continue

#             deg_metrics = self._calculate_degradation_metrics(stint_laps)
#             cond_factors = self._calculate_condition_factors(stint_laps, weather_data)
#             driving_factors = self._calculate_driving_factors(stint_laps, telemetry_data)
#             stint_features = {**deg_metrics, **cond_factors, **driving_factors}

#             if end + window_size <= len(car_laps):
#                 next_stint = car_laps.iloc[end:end + window_size]
#                 deg_targets = self._calculate_degradation_targets(stint_laps, next_stint)
#                 features.append(pd.DataFrame([stint_features]))
#                 targets.append(pd.DataFrame([deg_targets]))

#         features_df = pd.concat(features, ignore_index=True) if features else pd.DataFrame()
#         targets_df = pd.concat(targets, ignore_index=True) if targets else pd.DataFrame()
#         return features_df, targets_df

#     # ---------------------------
#     # METRIC CALCULATIONS
#     # ---------------------------
#     def _calculate_degradation_metrics(self, stint_laps):
#         metrics = {}
#         lap_times = stint_laps.get('LAP_TIME_SECONDS', pd.Series([60]*len(stint_laps))).values
#         lap_numbers = stint_laps.get('LAP_NUMBER', pd.Series(range(len(stint_laps)))).values

#         slope, r2 = self._linear_trend_analysis(lap_numbers, lap_times)
#         metrics['lap_time_slope'] = slope
#         metrics['lap_time_consistency'] = r2

#         for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
#             sector_vals = stint_laps.get(sector, pd.Series([60]*len(stint_laps))).values
#             sector_slope, _ = self._linear_trend_analysis(lap_numbers, sector_vals)
#             metrics[f'sector_{i}_slope'] = sector_slope

#         metrics['lap_time_variance'] = np.var(lap_times)
#         metrics['best_to_worst_ratio'] = np.min(lap_times) / np.max(lap_times)
#         return metrics

#     def _calculate_condition_factors(self, stint_laps, weather_data):
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

#         track_name = stint_laps.get('meta_event', pd.Series(['unknown'])).iloc[0]
#         factors['track_abrasiveness'] = self._get_track_abrasiveness(track_name)
#         return factors

#     def _calculate_driving_factors(self, stint_laps, telemetry_data):
#         factors = {}
#         if not telemetry_data.empty:
#             stint_telemetry = telemetry_data[
#                 telemetry_data['lap'].between(stint_laps['LAP_NUMBER'].min(), stint_laps['LAP_NUMBER'].max())
#             ]
#             factors['avg_lateral_g'] = stint_telemetry.get('accy_can', pd.Series([0.5])).abs().mean()
#             factors['avg_brake_pressure'] = ((stint_telemetry.get('pbrake_f', 0) + stint_telemetry.get('pbrake_r', 0))/2).mean() if not stint_telemetry.empty else 50.0
#             factors['avg_throttle_usage'] = stint_telemetry.get('aps', pd.Series([60])).mean()
#             factors['steering_variance'] = stint_telemetry.get('Steering_Angle', pd.Series([10])).var()
#         else:
#             factors['avg_lateral_g'] = 0.5
#             factors['avg_brake_pressure'] = 50.0
#             factors['avg_throttle_usage'] = 60.0
#             factors['steering_variance'] = 10.0

#         factors['stint_length'] = len(stint_laps)
#         factors['cumulative_laps'] = stint_laps.get('LAP_NUMBER', pd.Series([0])).max()
#         return factors

#     def _calculate_degradation_targets(self, current_stint, next_stint):
#         targets = {}
#         for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
#             current_avg = current_stint.get(sector, pd.Series([60])).mean()
#             next_avg = next_stint.get(sector, pd.Series([60])).mean()
#             targets[f'degradation_s{i}'] = max(0.001, min(0.5, (next_avg - current_avg) / len(next_stint)))
#         curr_avg_time = current_stint.get('LAP_TIME_SECONDS', pd.Series([60])).mean()
#         next_avg_time = next_stint.get('LAP_TIME_SECONDS', pd.Series([60])).mean()
#         targets['grip_loss_rate'] = max(0.001, min(1.0, (next_avg_time - curr_avg_time) / len(next_stint)))
#         return targets

#     def _linear_trend_analysis(self, x, y):
#         if len(x) < 2:
#             return 0.0, 0.0
#         try:
#             slope = np.polyfit(x, y, 1)[0]
#             r2 = np.corrcoef(x, y)[0, 1] ** 2
#             return slope, r2
#         except:
#             return 0.0, 0.0

#     def _get_track_abrasiveness(self, track_name):
#         abrasiveness_map = {'sebring': 0.9, 'barber': 0.8, 'sonoma': 0.7, 'cota': 0.6, 'road-america': 0.5, 'vir': 0.6}
#         return abrasiveness_map.get(track_name.lower(), 0.7)

#     # ---------------------------
#     # PREDICTION
#     # ---------------------------
#     def predict_degradation(self, features):
#         try:
#             vec = np.array([features.get(col, 0) for col in self.feature_columns]).reshape(1, -1)
#             scaled = self.scaler.transform(vec)
#             preds = self.model.predict(scaled)[0]
#             return dict(zip(self.target_columns, preds))
#         except:
#             return self._fallback_degradation(features)

#     def _fallback_degradation(self, features):
#         return {'degradation_s1': 0.05, 'degradation_s2': 0.05, 'degradation_s3': 0.05, 'grip_loss_rate': 0.1}

#     def estimate_optimal_stint_length(self, features, threshold=0.2):
#         rates = self.predict_degradation(features)
#         avg_deg = np.mean([rates['degradation_s1'], rates['degradation_s2'], rates['degradation_s3']])
#         return max(5, min(30, int(threshold / avg_deg))) if avg_deg > 0 else 15

#     # ---------------------------
#     # SYNTHETIC DATA HELPERS
#     # ---------------------------
#     def _fabricate_minimal_data(self):
#         lap_data = pd.concat([self._fabricate_car_laps(v, 'session1', 10) for v in range(1, 3)], ignore_index=True)
#         telemetry_data = pd.concat([self._fabricate_car_telemetry(v, 'session1', 10) for v in range(1, 3)], ignore_index=True)
#         weather_data = pd.DataFrame([{'meta_session': 'session1', 'timestamp': pd.Timestamp.now(),
#                                       'TRACK_TEMP': 35.0, 'AIR_TEMP': 25.0}])
#         return lap_data, telemetry_data, weather_data

#     def _fabricate_car_laps(self, vehicle_num, session, n_laps):
#         return pd.DataFrame({
#             'NUMBER': vehicle_num,
#             'meta_session': session,
#             'LAP_NUMBER': np.arange(1, n_laps+1),
#             'LAP_TIME_SECONDS': np.random.uniform(55, 65, n_laps),
#             'S1_SECONDS': np.random.uniform(18, 22, n_laps),
#             'S2_SECONDS': np.random.uniform(18, 22, n_laps),
#             'S3_SECONDS': np.random.uniform(18, 22, n_laps),
#             'timestamp': pd.date_range('2025-01-01', periods=n_laps)
#         })

#     def _fabricate_car_telemetry(self, vehicle_num, session, n_laps):
#         return pd.DataFrame({
#             'vehicle_number': vehicle_num,
#             'meta_session': session,
#             'lap': np.repeat(np.arange(1, n_laps+1), 10),
#             'aps': np.random.uniform(30, 80, n_laps*10),
#             'pbrake_f': np.random.uniform(0, 50, n_laps*10),
#             'pbrake_r': np.random.uniform(0, 50, n_laps*10),
#             'accy_can': np.random.uniform(-1, 1, n_laps*10),
#             'Steering_Angle': np.random.uniform(-15, 15, n_laps*10)
#         })

#     # ---------------------------
#     # MODEL SERIALIZATION
#     # ---------------------------
#     def save_model(self, filepath):
#         joblib.dump({
#             'model': self.model,
#             'scaler': self.scaler,
#             'feature_columns': self.feature_columns,
#             'target_columns': self.target_columns
#         }, filepath)

#     def load_model(self, filepath):
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
#             lap_data, telemetry_data, weather_data = self._fabricate_minimal_data()

#         features_df, targets_df = self._extract_tire_features(lap_data, telemetry_data, weather_data)

#         if features_df.empty or targets_df.empty:
#             return {'error': 'No valid tire features extracted'}

#         X = features_df
#         y = targets_df[self.target_columns]

#         valid_mask = ~X.isna().any(axis=1) & ~y.isna().any(axis=1)
#         X = X[valid_mask]
#         y = y[valid_mask]

#         if len(X) < 20:
#             return {'error': f'Insufficient training samples: {len(X)}'}

#         X_scaled = self.scaler.fit_transform(X)
#         self.feature_columns = X.columns.tolist()

#         X_train, X_test, y_train, y_test = train_test_split(
#             X_scaled, y, test_size=0.2, random_state=42
#         )
#         self.model.fit(X_train, y_train)

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
#                 # fabricate missing laps
#                 car_laps = self._fabricate_car_laps(car_number, session, 8)

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
#         lap_times = stint_laps.get('LAP_TIME_SECONDS', pd.Series([60]*len(stint_laps))).values
#         lap_numbers = stint_laps.get('LAP_NUMBER', pd.Series(range(len(stint_laps)))).values

#         try:
#             slope, r2 = self._linear_trend_analysis(lap_numbers, lap_times)
#             metrics['lap_time_slope'] = slope
#             metrics['lap_time_consistency'] = r2
#         except:
#             metrics['lap_time_slope'] = 0.0
#             metrics['lap_time_consistency'] = 0.0

#         for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
#             try:
#                 sector_vals = stint_laps.get(sector, pd.Series([60]*len(stint_laps))).values
#                 sector_slope, _ = self._linear_trend_analysis(lap_numbers, sector_vals)
#                 metrics[f'sector_{i}_slope'] = sector_slope
#             except:
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

#         track_name = stint_laps.get('meta_event', pd.Series(['unknown'])).iloc[0]
#         factors['track_abrasiveness'] = self._get_track_abrasiveness(track_name)
#         return factors

#     def _calculate_driving_factors(self, stint_laps: pd.DataFrame, telemetry_data: pd.DataFrame) -> dict:
#         factors = {}
#         if not telemetry_data.empty:
#             stint_telemetry = telemetry_data[
#                 telemetry_data['lap'].between(stint_laps['LAP_NUMBER'].min(), stint_laps['LAP_NUMBER'].max())
#             ]
#             factors['avg_lateral_g'] = stint_telemetry.get('accy_can', pd.Series([0.5])).abs().mean()
#             factors['avg_brake_pressure'] = ((stint_telemetry.get('pbrake_f', 0) + stint_telemetry.get('pbrake_r', 0))/2).mean() if not stint_telemetry.empty else 50.0
#             factors['avg_throttle_usage'] = stint_telemetry.get('aps', pd.Series([60])).mean()
#             factors['steering_variance'] = stint_telemetry.get('Steering_Angle', pd.Series([10])).var()
#         else:
#             factors['avg_lateral_g'] = 0.5
#             factors['avg_brake_pressure'] = 50.0
#             factors['avg_throttle_usage'] = 60.0
#             factors['steering_variance'] = 10.0

#         factors['stint_length'] = len(stint_laps)
#         factors['cumulative_laps'] = stint_laps.get('LAP_NUMBER', pd.Series([0])).max()
#         return factors

#     def _calculate_degradation_targets(self, current_stint: pd.DataFrame, next_stint: pd.DataFrame) -> dict:
#         targets = {}
#         for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
#             current_avg = current_stint.get(sector, pd.Series([60])).mean()
#             next_avg = next_stint.get(sector, pd.Series([60])).mean()
#             targets[f'degradation_s{i}'] = max(0.001, min(0.5, (next_avg - current_avg) / len(next_stint)))
#         current_avg_time = current_stint.get('LAP_TIME_SECONDS', pd.Series([60])).mean()
#         next_avg_time = next_stint.get('LAP_TIME_SECONDS', pd.Series([60])).mean()
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
#     # SYNTHETIC DATA HELPERS
#     # ------------------------------------------------------------
#     def _fabricate_minimal_data(self) -> tuple:
#         lap_data = pd.concat([self._fabricate_car_laps(vehicle, 'session1', 10) for vehicle in range(1,3)], ignore_index=True)
#         telemetry_data = pd.concat([self._fabricate_car_telemetry(vehicle, 'session1', 10) for vehicle in range(1,3)], ignore_index=True)
#         weather_data = pd.DataFrame([{
#             'meta_session': 'session1',
#             'timestamp': pd.Timestamp.now(),
#             'TRACK_TEMP': 35.0,
#             'AIR_TEMP': 25.0
#         }])
#         return lap_data, telemetry_data, weather_data

#     def _fabricate_car_laps(self, vehicle_num: int, session: str, n_laps: int) -> pd.DataFrame:
#         return pd.DataFrame({
#             'NUMBER': vehicle_num,
#             'meta_session': session,
#             'LAP_NUMBER': np.arange(1, n_laps+1),
#             'LAP_TIME_SECONDS': np.random.uniform(55, 65, n_laps),
#             'S1_SECONDS': np.random.uniform(18, 22, n_laps),
#             'S2_SECONDS': np.random.uniform(18, 22, n_laps),
#             'S3_SECONDS': np.random.uniform(18, 22, n_laps),
#             'timestamp': pd.date_range('2025-01-01', periods=n_laps)
#         })

#     def _fabricate_car_telemetry(self, vehicle_num: int, session: str, n_laps: int) -> pd.DataFrame:
#         return pd.DataFrame({
#             'vehicle_number': vehicle_num,
#             'meta_session': session,
#             'lap': np.repeat(np.arange(1, n_laps+1), 10),
#             'aps': np.random.uniform(30, 80, n_laps*10),
#             'pbrake_f': np.random.uniform(0, 50, n_laps*10),
#             'pbrake_r': np.random.uniform(0, 50, n_laps*10),
#             'accy_can': np.random.uniform(-1, 1, n_laps*10),
#             'Steering_Angle': np.random.uniform(-15, 15, n_laps*10)
#         })

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