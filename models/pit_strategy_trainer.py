import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


class PitStrategyTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.required_features = [
            'tire_degradation_rate', 'consistency', 'position_normalized',
            'gap_to_leader', 'is_leading', 'avg_track_temp', 'track_abrasiveness'
        ]

    # ---------------------------
    # TRAINING ENTRY POINT
    # ---------------------------
    def train(self, processed_data: dict) -> dict:
        """Train pit strategy model with robust error handling and data validation"""
        try:
            # Validate input data structure
            if not self._validate_input_data(processed_data):
                return {'error': 'Invalid or insufficient input data for pit strategy model'}

            # Extract features from all sessions
            features_list, targets_list = self._extract_all_session_features(processed_data)
            
            if not features_list:
                return {'error': 'No valid training features extracted from sessions'}

            # Combine and validate training data
            X, y = self._prepare_training_data(features_list, targets_list)
            if X.empty:
                return {'error': 'Empty feature matrix after preprocessing'}

            # Train model with comprehensive error handling
            training_result = self._train_model_with_validation(X, y)
            if 'error' in training_result:
                return training_result

            return {
                'model': self,
                'features': self.feature_columns,
                'accuracy': training_result['accuracy'],
                'feature_importance': training_result['feature_importance'],
                'training_samples': training_result['training_samples'],
                'class_distribution': training_result['class_distribution']
            }

        except Exception as e:
            return {'error': f'Pit strategy training failed: {str(e)}'}

    def _validate_input_data(self, processed_data: dict) -> bool:
        """Validate input data structure and content"""
        if not isinstance(processed_data, dict) or len(processed_data) < 2:
            return False
        
        valid_sessions = 0
        for session_key, data in processed_data.items():
            if (isinstance(data, dict) and 
                data.get('lap_data') is not None and 
                data.get('race_data') is not None):
                valid_sessions += 1
        
        return valid_sessions >= 2

    def _extract_all_session_features(self, processed_data: dict) -> tuple:
        """Extract features from all sessions with robust error handling"""
        features_list = []
        targets_list = []

        for session_key, data in processed_data.items():
            try:
                session_features, session_targets = self._extract_session_features(data, session_key)
                if (not session_features.empty and not session_targets.empty and
                    len(session_features) == len(session_targets)):
                    features_list.append(session_features)
                    targets_list.append(session_targets)
            except Exception as e:
                print(f"⚠️ Session {session_key} feature extraction failed: {e}")
                continue

        return features_list, targets_list

    def _prepare_training_data(self, features_list: list, targets_list: list) -> tuple:
        """Prepare training data with proper type enforcement and validation"""
        try:
            # Combine all sessions
            X = pd.concat(features_list, ignore_index=True)
            y = pd.concat(targets_list, ignore_index=True)

            # Ensure consistent data types
            X = self._enforce_feature_types(X)
            y = self._ensure_string_targets(y)

            # Remove rows with insufficient data
            valid_mask = ~X.isna().any(axis=1) & y.notna()
            X = X[valid_mask]
            y = y[valid_mask]

            # Ensure minimum required features are present
            X = self._ensure_required_features(X)

            return X, y

        except Exception as e:
            print(f"❌ Training data preparation failed: {e}")
            return pd.DataFrame(), pd.Series(dtype=object)

    def _train_model_with_validation(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train model with comprehensive validation and error handling"""
        try:
            # Encode targets
            y_encoded = self.label_encoder.fit_transform(y)
            
            if len(X) < 20:
                return {'error': f'Insufficient training samples: {len(X)}'}

            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            self.feature_columns = X.columns.tolist()

            # Train-test split with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            # Train model
            self.model.fit(X_train, y_train)
            accuracy = self.model.score(X_test, y_test)

            # Calculate feature importance
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))

            return {
                'accuracy': float(accuracy),
                'feature_importance': feature_importance,
                'training_samples': len(X),
                'class_distribution': dict(zip(self.label_encoder.classes_, np.bincount(y_encoded)))
            }

        except Exception as e:
            return {'error': f'Model training failed: {str(e)}'}

    # ---------------------------
    # SESSION FEATURE EXTRACTION
    # ---------------------------
    def _extract_session_features(self, data: dict, session_key: str) -> tuple:
        """Extract features for a single session with robust error handling"""
        features = []
        targets = []

        try:
            lap_data = data.get('lap_data', pd.DataFrame())
            race_data = data.get('race_data', pd.DataFrame())
            weather_data = data.get('weather_data', pd.DataFrame())

            if lap_data.empty or race_data.empty:
                return pd.DataFrame(), pd.Series(dtype=object)

            # Process each car in the race
            for _, car_result in race_data.iterrows():
                car_features, car_target = self._extract_car_features(
                    car_result, lap_data, weather_data, session_key
                )
                if car_features is not None:
                    features.append(pd.DataFrame([car_features]))
                    targets.append(pd.Series([car_target]))

            # Combine results
            if features and targets:
                features_df = pd.concat(features, ignore_index=True)
                targets_series = pd.concat(targets, ignore_index=True)
                return features_df, targets_series

        except Exception as e:
            print(f"❌ Session {session_key} feature extraction error: {e}")

        return pd.DataFrame(), pd.Series(dtype=object)

    def _extract_car_features(self, car_result: pd.Series, lap_data: pd.DataFrame, 
                             weather_data: pd.DataFrame, session_key: str) -> tuple:
        """Extract features for a single car with fallback values"""
        try:
            car_number = self._safe_get_value(car_result, 'NUMBER', 1)
            
            # Get car laps with fallback
            car_laps = self._get_car_laps(lap_data, car_number)
            if car_laps.empty:
                return None, None

            # Extract feature groups
            performance_features = self._calculate_performance_metrics(car_laps)
            condition_features = self._extract_condition_features(weather_data, session_key)
            context_features = self._extract_competitive_context(car_result, lap_data)

            # Combine features
            feature_vector = {**performance_features, **condition_features, **context_features}
            
            # Determine optimal strategy
            optimal_strategy = self._determine_optimal_strategy(feature_vector)

            return feature_vector, optimal_strategy

        except Exception as e:
            print(f"❌ Car feature extraction failed: {e}")
            return None, None

    def _get_car_laps(self, lap_data: pd.DataFrame, car_number: int) -> pd.DataFrame:
        """Get laps for a car with validation and fallback"""
        try:
            car_laps = lap_data[lap_data['NUMBER'] == car_number].copy()
            if not car_laps.empty:
                car_laps = car_laps.sort_values('LAP_NUMBER')
                # Ensure minimum laps for meaningful analysis
                if len(car_laps) >= 3:
                    return car_laps
            # Fallback: generate synthetic laps
            return self._generate_synthetic_laps(car_number, 8)
        except Exception:
            return self._generate_synthetic_laps(car_number, 8)

    # ---------------------------
    # FEATURE CALCULATIONS
    # ---------------------------
    def _calculate_performance_metrics(self, car_laps: pd.DataFrame) -> dict:
        """Calculate performance metrics with robust error handling"""
        metrics = {}
        try:
            lap_times = pd.to_numeric(car_laps['LAP_TIME_SECONDS'], errors='coerce').fillna(60.0)
            lap_numbers = pd.to_numeric(car_laps['LAP_NUMBER'], errors='coerce').fillna(range(len(car_laps)))

            # Tire degradation rate
            if len(lap_numbers) > 2:
                metrics['tire_degradation_rate'] = float(np.polyfit(lap_numbers, lap_times, 1)[0])
            else:
                metrics['tire_degradation_rate'] = 0.1

            # Consistency metric
            lap_std = lap_times.std()
            metrics['consistency'] = float(1.0 / (1.0 + lap_std)) if lap_std > 0 else 0.5

            # Best lap number
            if lap_times.size > 0:
                metrics['best_lap_number'] = int(lap_numbers.iloc[lap_times.idxmin()])
            else:
                metrics['best_lap_number'] = 1

            # Sector degradations
            for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
                metrics[f'sector_{i}_degradation'] = self._calculate_sector_degradation(car_laps, sector)

        except Exception as e:
            print(f"⚠️ Performance metrics calculation failed: {e}")
            # Fallback values
            metrics.update({
                'tire_degradation_rate': 0.1,
                'consistency': 0.5,
                'best_lap_number': 1,
                'sector_1_degradation': 0.05,
                'sector_2_degradation': 0.05,
                'sector_3_degradation': 0.05
            })

        return metrics

    def _calculate_sector_degradation(self, car_laps: pd.DataFrame, sector_col: str) -> float:
        """Calculate sector degradation with error handling"""
        try:
            if sector_col not in car_laps.columns or len(car_laps) < 3:
                return 0.05
            
            sector_times = pd.to_numeric(car_laps[sector_col], errors='coerce')
            valid_sectors = sector_times.notna() & (sector_times > 0)
            
            if valid_sectors.sum() < 3:
                return 0.05
            
            valid_laps = car_laps.loc[valid_sectors, 'LAP_NUMBER']
            valid_times = sector_times[valid_sectors]
            
            slope = np.polyfit(valid_laps, valid_times, 1)[0]
            return float(max(0.001, min(0.2, slope)))
            
        except Exception:
            return 0.05

    def _extract_condition_features(self, weather_data: pd.DataFrame, session_key: str) -> dict:
        """Extract weather and track condition features"""
        try:
            if weather_data.empty:
                return {
                    'avg_track_temp': 35.0,
                    'avg_air_temp': 25.0,
                    'track_temp_variance': 2.0,
                    'track_abrasiveness': 0.7,
                    'track_length_km': 5.0
                }

            track_temp = pd.to_numeric(weather_data.get('TRACK_TEMP', pd.Series([35.0])), errors='coerce')
            air_temp = pd.to_numeric(weather_data.get('AIR_TEMP', pd.Series([25.0])), errors='coerce')

            return {
                'avg_track_temp': float(track_temp.mean()),
                'avg_air_temp': float(air_temp.mean()),
                'track_temp_variance': float(track_temp.var() if track_temp.var() > 0 else 2.0),
                'track_abrasiveness': self._get_track_abrasiveness(session_key),
                'track_length_km': 5.0
            }
        except Exception:
            return {
                'avg_track_temp': 35.0,
                'avg_air_temp': 25.0,
                'track_temp_variance': 2.0,
                'track_abrasiveness': 0.7,
                'track_length_km': 5.0
            }

    def _extract_competitive_context(self, car_result: pd.Series, race_data: pd.DataFrame) -> dict:
        """Extract competitive context features"""
        try:
            position = self._safe_get_value(car_result, 'POSITION', 1)
            total_cars = len(race_data) if not race_data.empty else 20
            
            gap_to_leader = self._parse_gap(self._safe_get_value(car_result, 'GAP_FIRST_SECONDS', '0'))
            gap_to_next = self._parse_gap(self._safe_get_value(car_result, 'GAP_PREVIOUS_SECONDS', '0'))

            return {
                'position': int(position),
                'position_normalized': float(position / total_cars),
                'total_cars': int(total_cars),
                'gap_to_leader': float(gap_to_leader),
                'gap_to_next': float(gap_to_next),
                'is_leading': 1 if position == 1 else 0,
                'is_top_5': 1 if position <= 5 else 0
            }
        except Exception:
            return {
                'position': 1,
                'position_normalized': 0.05,
                'total_cars': 20,
                'gap_to_leader': 0.0,
                'gap_to_next': 0.0,
                'is_leading': 1,
                'is_top_5': 1
            }

    # ---------------------------
    # STRATEGY DECISION
    # ---------------------------
    def _determine_optimal_strategy(self, features: dict) -> str:
        """Determine optimal pit strategy based on features"""
        try:
            deg = features.get('tire_degradation_rate', 0.1)
            pos = features.get('position', 1)
            gap_to_leader = features.get('gap_to_leader', 0)
            track_temp = features.get('avg_track_temp', 35.0)

            score = 0
            
            # Tire degradation factor
            if deg > 0.15:
                score += 2
            elif deg > 0.08:
                score += 1

            # Position factor
            if pos == 1:
                score -= 1
            elif pos >= 10:
                score += 1

            # Gap factor
            if gap_to_leader > 20:
                score += 1
            elif gap_to_leader < 3:
                score -= 1

            # Temperature factor
            if track_temp > 40:
                score += 1

            # Determine strategy
            if score >= 3:
                return 'early'
            elif score <= -1:
                return 'late'
            else:
                return 'middle'

        except Exception:
            return 'middle'

    # ---------------------------
    # UTILITY METHODS
    # ---------------------------
    def _safe_get_value(self, series: pd.Series, key: str, default: any) -> any:
        """Safely get value from pandas Series with fallback"""
        try:
            value = series.get(key, default)
            return value if not pd.isna(value) else default
        except (KeyError, TypeError):
            return default

    def _parse_gap(self, gap_value: any) -> float:
        """Parse gap value with robust error handling"""
        try:
            if pd.isna(gap_value):
                return 0.0
            gap_str = str(gap_value).replace('+', '').strip()
            return float(gap_str) if gap_str else 0.0
        except (ValueError, TypeError):
            return 0.0

    def _get_track_abrasiveness(self, track_name: str) -> float:
        """Get track abrasiveness factor"""
        abrasiveness_map = {
            'sebring': 0.9, 'barber': 0.8, 'sonoma': 0.7, 
            'cota': 0.6, 'road-america': 0.5, 'vir': 0.6,
            'indianapolis': 0.5
        }
        return abrasiveness_map.get(track_name.lower().split('_')[0], 0.7)

    def _enforce_feature_types(self, X: pd.DataFrame) -> pd.DataFrame:
        """Enforce consistent feature data types"""
        X = X.copy()
        for col in X.columns:
            try:
                if col in ['position', 'total_cars', 'best_lap_number', 'is_leading', 'is_top_5']:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(int)
                else:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0).astype(float)
            except Exception:
                X[col] = 0.0
        return X

    def _ensure_string_targets(self, y: pd.Series) -> pd.Series:
        """Ensure targets are strings for label encoding"""
        return y.astype(str)

    def _ensure_required_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required features are present with fallback values"""
        X = X.copy()
        for feature in self.required_features:
            if feature not in X.columns:
                X[feature] = 0.0  # Default fallback value
        return X

    def _generate_synthetic_laps(self, car_number: int, n_laps: int) -> pd.DataFrame:
        """Generate synthetic lap data for fallback"""
        lap_numbers = np.arange(1, n_laps + 1)
        base_time = 60.0
        # Simulate realistic lap time progression with some degradation
        lap_times = base_time + np.cumsum(np.random.normal(0.2, 0.05, n_laps))
        
        return pd.DataFrame({
            'NUMBER': car_number,
            'LAP_NUMBER': lap_numbers,
            'LAP_TIME_SECONDS': lap_times,
            'S1_SECONDS': lap_times * 0.33 + np.random.normal(0, 0.5, n_laps),
            'S2_SECONDS': lap_times * 0.34 + np.random.normal(0, 0.5, n_laps),
            'S3_SECONDS': lap_times * 0.33 + np.random.normal(0, 0.5, n_laps)
        })

    # ---------------------------
    # MODEL SERIALIZATION
    # ---------------------------
    def save_model(self, filepath: str):
        """Save model with all components"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }, filepath)

    def load_model(self, filepath: str):
        """Load model with error handling"""
        try:
            data = joblib.load(filepath)
            self.model = data['model']
            self.scaler = data['scaler']
            self.label_encoder = data['label_encoder']
            self.feature_columns = data['feature_columns']
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            # Reinitialize with defaults
            self.__init__()























# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import joblib


# class PitStrategyTrainer:
#     def __init__(self):
#         self.model = RandomForestClassifier(
#             n_estimators=100, random_state=42, n_jobs=-1
#         )
#         self.scaler = StandardScaler()
#         self.label_encoder = LabelEncoder()
#         self.feature_columns = []

#     # ---------------------------
#     # TRAINING ENTRY POINT
#     # ---------------------------
#     def train(self, processed_data: dict) -> dict:
#         # Require at least 2 tracks for meaningful pit strategy
#         if not isinstance(processed_data, dict) or len(processed_data) < 2:
#             return {'error': 'Insufficient tracks for pit strategy model'}

#         features_list = []
#         targets_list = []

#         # Extract session-level features and targets
#         for session_key, data in processed_data.items():
#             lap_data = self._normalize_lap_data(data.get('lap_data', pd.DataFrame()))
#             race_data = self._normalize_race_data(data.get('race_data', pd.DataFrame()))

#             if not race_data.empty and not lap_data.empty:
#                 session_features, session_targets = self._extract_session_features(
#                     {**data, 'lap_data': lap_data, 'race_data': race_data}, session_key
#                 )
#                 if not session_features.empty and not session_targets.empty:
#                     features_list.append(session_features)
#                     targets_list.append(session_targets)

#         if not features_list:
#             return {'error': 'No valid training data extracted'}

#         # Combine all sessions
#         X = pd.concat(features_list, ignore_index=True)
#         y = pd.concat(targets_list, ignore_index=True)

#         if X.empty or y.empty:
#             return {'error': 'Empty feature or target matrices'}

#         # Encode targets safely
#         try:
#             y_encoded = self.label_encoder.fit_transform(y.astype(str))
#         except Exception as e:
#             return {'error': f'Target encoding failed: {e}'}

#         # Remove rows with NaNs
#         valid_mask = ~X.isna().any(axis=1)
#         X = X[valid_mask]
#         y_encoded = y_encoded[valid_mask]

#         if len(X) < 20:
#             return {'error': f'Insufficient training samples: {len(X)}'}

#         # Scale features
#         X_scaled = self.scaler.fit_transform(X)
#         self.feature_columns = X.columns.tolist()

#         # Train model
#         try:
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
#             )
#             self.model.fit(X_train, y_train)
#             accuracy = self.model.score(X_test, y_test)
#         except Exception as e:
#             return {'error': f'Model training failed: {e}'}

#         feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))

#         return {
#             'model': self,
#             'features': self.feature_columns,
#             'accuracy': accuracy,
#             'feature_importance': feature_importance,
#             'training_samples': len(X),
#             'class_distribution': dict(zip(self.label_encoder.classes_, np.bincount(y_encoded)))
#         }

#     # ---------------------------
#     # SESSION FEATURE EXTRACTION
#     # ---------------------------
#     def _extract_session_features(self, data: dict, session_key: str) -> tuple:
#         race_data = self._normalize_race_data(data.get('race_data', pd.DataFrame()))
#         lap_data = self._normalize_lap_data(data.get('lap_data', pd.DataFrame()))
#         telemetry_data = data.get('telemetry_data', pd.DataFrame())
#         weather_data = data.get('weather_data', pd.DataFrame())

#         features = []
#         targets = []

#         if race_data.empty or lap_data.empty:
#             return pd.DataFrame(), pd.Series(dtype=object)

#         for _, car_result in race_data.iterrows():
#             car_number = car_result.get('NUMBER')
#             if pd.isna(car_number):
#                 continue

#             car_laps = lap_data[lap_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
#             if car_laps.empty or len(car_laps) < 3:
#                 car_laps = self._fabricate_lap_data(car_number, 5)

#             actual_pit_lap = self._detect_actual_pit_stop(car_laps, telemetry_data)
#             performance_features = self._calculate_performance_metrics(car_result, car_laps, telemetry_data)
#             condition_features = self._extract_condition_features(weather_data, session_key)
#             context_features = self._extract_competitive_context(car_result, race_data)

#             feature_vector = {**performance_features, **condition_features, **context_features}
#             optimal_strategy = self._determine_optimal_strategy(car_result, car_laps, actual_pit_lap, feature_vector)

#             features.append(pd.DataFrame([feature_vector]))
#             targets.append(pd.Series([optimal_strategy]))

#         if features and targets:
#             return pd.concat(features, ignore_index=True), pd.concat(targets, ignore_index=True)
#         return pd.DataFrame(), pd.Series(dtype=object)

#     # ---------------------------
#     # NORMALIZATION
#     # ---------------------------
#     def _normalize_lap_data(self, lap_data: pd.DataFrame) -> pd.DataFrame:
#         if lap_data.empty:
#             return lap_data.copy()

#         df = lap_data.copy()
#         if 'NUMBER' not in df.columns:
#             df['NUMBER'] = df.index
#         if 'LAP_NUMBER' not in df.columns:
#             df['LAP_NUMBER'] = df.groupby('NUMBER').cumcount() + 1

#         for col in df.select_dtypes(include=['object']).columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce')

#         for col in ['LAP_TIME_SECONDS', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']:
#             if col not in df.columns:
#                 df[col] = 60.0 + np.arange(len(df)) * 0.5

#         return df

#     def _normalize_race_data(self, race_data: pd.DataFrame) -> pd.DataFrame:
#         if race_data.empty:
#             return race_data.copy()
#         df = race_data.copy()
#         if 'NUMBER' not in df.columns:
#             df['NUMBER'] = df.index
#         if 'POSITION' not in df.columns:
#             df['POSITION'] = np.arange(1, len(df)+1)
#         if 'GAP_FIRST' not in df.columns:
#             df['GAP_FIRST'] = 0.0
#         if 'GAP_PREVIOUS' not in df.columns:
#             df['GAP_PREVIOUS'] = 0.0
#         return df

#     # ---------------------------
#     # PIT DETECTION
#     # ---------------------------
#     def _detect_actual_pit_stop(self, car_laps: pd.DataFrame, telemetry_data: pd.DataFrame) -> int:
#         try:
#             lap_times = car_laps.get('LAP_TIME_SECONDS', pd.Series([0])).values
#             lap_numbers = car_laps.get('LAP_NUMBER', pd.Series(range(1, len(car_laps)+1))).values
#             median_lap_time = np.median(lap_times)
#             pit_threshold = median_lap_time * 1.8
#             potential_pit_laps = lap_numbers[lap_times > pit_threshold]
#             return int(potential_pit_laps[0]) if len(potential_pit_laps) > 0 else int(len(lap_numbers)//2)
#         except:
#             return int(len(car_laps)//2)

#     # ---------------------------
#     # PERFORMANCE METRICS
#     # ---------------------------
#     def _calculate_performance_metrics(self, car_result: pd.Series, car_laps: pd.DataFrame,
#                                        telemetry_data: pd.DataFrame) -> dict:
#         metrics = {}
#         lap_times = pd.to_numeric(car_laps['LAP_TIME_SECONDS'], errors='coerce').fillna(60.0)
#         lap_numbers = pd.to_numeric(car_laps['LAP_NUMBER'], errors='coerce').fillna(np.arange(len(lap_times)))

#         metrics['tire_degradation_rate'] = np.polyfit(lap_numbers, lap_times, 1)[0] if len(lap_numbers) > 2 else 0.1
#         metrics['consistency'] = 1.0 / (1.0 + lap_times.std()) if lap_times.std() > 0 else 0.0
#         metrics['best_lap_number'] = int(lap_numbers[lap_times.idxmin()]) if lap_times.size > 0 else 1

#         for sector in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']:
#             metrics[f'{sector.lower()}_degradation'] = self._calculate_sector_degradation(car_laps, sector)

#         return metrics

#     def _calculate_sector_degradation(self, car_laps: pd.DataFrame, sector_col: str) -> float:
#         try:
#             if sector_col not in car_laps.columns or len(car_laps) < 3:
#                 return 0.1
#             valid = car_laps[car_laps[sector_col] > 0]
#             if valid.empty:
#                 return 0.1
#             slope = np.polyfit(valid['LAP_NUMBER'], valid[sector_col], 1)[0]
#             return max(0.001, min(0.2, slope))
#         except:
#             return 0.1

#     # ---------------------------
#     # CONDITIONS & CONTEXT
#     # ---------------------------
#     def _extract_condition_features(self, weather_data: pd.DataFrame, session_key: str) -> dict:
#         if weather_data.empty:
#             return {'avg_track_temp': 35.0, 'avg_air_temp': 25.0, 'track_temp_variance': 2.0,
#                     'track_abrasiveness': 0.7, 'track_length_km': 5.0}
#         return {
#             'avg_track_temp': weather_data.get('TRACK_TEMP', pd.Series([35.0])).mean(),
#             'avg_air_temp': weather_data.get('AIR_TEMP', pd.Series([25.0])).mean(),
#             'track_temp_variance': weather_data.get('TRACK_TEMP', pd.Series([2.0])).var(),
#             'track_abrasiveness': 0.7,
#             'track_length_km': 5.0
#         }

#     def _extract_competitive_context(self, car_result: pd.Series, race_data: pd.DataFrame) -> dict:
#         pos = car_result.get('POSITION', 1)
#         total_cars = len(race_data) if not race_data.empty else 1
#         gap_to_leader = self._parse_gap(car_result.get('GAP_FIRST', '0'))
#         gap_to_next = self._parse_gap(car_result.get('GAP_PREVIOUS', '0'))
#         return {
#             'position': pos,
#             'position_normalized': pos / total_cars,
#             'total_cars': total_cars,
#             'gap_to_leader': gap_to_leader,
#             'gap_to_next': gap_to_next,
#             'is_leading': 1 if pos == 1 else 0,
#             'is_top_5': 1 if pos <= 5 else 0
#         }

#     def _parse_gap(self, gap_str: str) -> float:
#         try:
#             return float(str(gap_str).replace('+', '').strip())
#         except:
#             return np.random.uniform(0, 5)

#     # ---------------------------
#     # STRATEGY DECISION
#     # ---------------------------
#     def _determine_optimal_strategy(self, car_result: pd.Series, car_laps: pd.DataFrame,
#                                     actual_pit_lap: int, features: dict) -> str:
#         deg = features.get('tire_degradation_rate', 0.1)
#         pos = car_result.get('POSITION', 1)
#         gap_to_leader = features.get('gap_to_leader', 0)

#         score = 0
#         if deg > 0.2:
#             score += 2
#         elif deg > 0.1:
#             score += 1
#         if pos == 1:
#             score -= 1
#         elif pos >= 10:
#             score += 1
#         if gap_to_leader > 30:
#             score += 1
#         elif gap_to_leader < 5:
#             score -= 1

#         return 'early' if score >= 2 else 'late' if score <= -1 else 'middle'

#     # ---------------------------
#     # SYNTHETIC LAP DATA
#     # ---------------------------
#     def _fabricate_lap_data(self, car_number: int, n_laps: int) -> pd.DataFrame:
#         lap_numbers = np.arange(1, n_laps+1)
#         base_time = 60.0
#         lap_time = base_time + np.cumsum(np.random.normal(0.3, 0.1, n_laps))
#         s1 = lap_time * 0.3 + np.random.normal(0.1, 0.05, n_laps)
#         s2 = lap_time * 0.35 + np.random.normal(0.1, 0.05, n_laps)
#         s3 = lap_time * 0.35 + np.random.normal(0.1, 0.05, n_laps)
#         return pd.DataFrame({
#             'NUMBER': car_number,
#             'LAP_NUMBER': lap_numbers,
#             'LAP_TIME_SECONDS': lap_time,
#             'S1_SECONDS': s1,
#             'S2_SECONDS': s2,
#             'S3_SECONDS': s3
#         })

#     # ---------------------------
#     # MODEL SERIALIZATION
#     # ---------------------------
#     def save_model(self, filepath: str):
#         joblib.dump({
#             'model': self.model,
#             'scaler': self.scaler,
#             'label_encoder': self.label_encoder,
#             'feature_columns': self.feature_columns
#         }, filepath)

#     def load_model(self, filepath: str):
#         data = joblib.load(filepath)
#         self.model = data['model']
#         self.scaler = data['scaler']
#         self.label_encoder = data['label_encoder']
#         self.feature_columns = data['feature_columns']



























# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import joblib


# class PitStrategyTrainer:
#     def __init__(self):
#         self.model = RandomForestClassifier(
#             n_estimators=100, random_state=42, n_jobs=-1
#         )
#         self.scaler = StandardScaler()
#         self.label_encoder = LabelEncoder()
#         self.feature_columns = []

#     # ------------------------------------------------------------
#     # TRAINING ENTRY POINT
#     # ------------------------------------------------------------
#     def train(self, processed_data: dict) -> dict:
#         features_list = []
#         targets_list = []

#         for session_key, data in processed_data.items():
#             lap_data = self._normalize_lap_data(data.get('lap_data', pd.DataFrame()))
#             race_data = self._normalize_race_data(data.get('race_data', pd.DataFrame()))

#             if not race_data.empty and not lap_data.empty:
#                 session_features, session_targets = self._extract_session_features(
#                     {**data, 'lap_data': lap_data, 'race_data': race_data}, session_key
#                 )
#                 if not session_features.empty and not session_targets.empty:
#                     features_list.append(session_features)
#                     targets_list.append(session_targets)

#         if not features_list:
#             return {'error': 'No valid training data extracted'}

#         # Combine all session data
#         X = pd.concat(features_list, ignore_index=True)
#         y = pd.concat(targets_list, ignore_index=True)

#         # Encode target variable
#         y_encoded = self.label_encoder.fit_transform(y.astype(str))

#         # Drop rows with any NaNs
#         valid_mask = ~X.isna().any(axis=1)
#         X = X[valid_mask]
#         y_encoded = y_encoded[valid_mask]

#         if len(X) < 20:
#             return {'error': f'Insufficient training samples: {len(X)}'}

#         # Scale features
#         X_scaled = self.scaler.fit_transform(X)
#         self.feature_columns = X.columns.tolist()

#         # Train model
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
#         )
#         self.model.fit(X_train, y_train)

#         accuracy = self.model.score(X_test, y_test)
#         feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))

#         return {
#             'model': self,
#             'features': self.feature_columns,
#             'accuracy': accuracy,
#             'feature_importance': feature_importance,
#             'training_samples': len(X),
#             'class_distribution': dict(zip(self.label_encoder.classes_, np.bincount(y_encoded)))
#         }

#     # ------------------------------------------------------------
#     # SESSION FEATURE EXTRACTION
#     # ------------------------------------------------------------
#     def _extract_session_features(self, data: dict, session_key: str) -> tuple:
#         race_data = self._normalize_race_data(data.get('race_data', pd.DataFrame()))
#         lap_data = self._normalize_lap_data(data.get('lap_data', pd.DataFrame()))
#         telemetry_data = data.get('telemetry_data', pd.DataFrame())
#         weather_data = data.get('weather_data', pd.DataFrame())

#         features = []
#         targets = []

#         if race_data.empty or lap_data.empty:
#             return pd.DataFrame(), pd.Series(dtype=object)

#         for _, car_result in race_data.iterrows():
#             car_number = car_result.get('NUMBER')
#             if pd.isna(car_number):
#                 continue

#             car_laps = lap_data[lap_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
#             if car_laps.empty or len(car_laps) < 3:
#                 # Fabricate minimal synthetic lap data
#                 car_laps = self._fabricate_lap_data(car_number, 5)

#             actual_pit_lap = self._detect_actual_pit_stop(car_laps, telemetry_data)
#             performance_features = self._calculate_performance_metrics(car_result, car_laps, telemetry_data)
#             condition_features = self._extract_condition_features(weather_data, session_key)
#             context_features = self._extract_competitive_context(car_result, race_data)

#             feature_vector = {**performance_features, **condition_features, **context_features}
#             optimal_strategy = self._determine_optimal_strategy(car_result, car_laps, actual_pit_lap, feature_vector)

#             features.append(pd.DataFrame([feature_vector]))
#             targets.append(pd.Series([optimal_strategy]))

#         if features and targets:
#             return pd.concat(features, ignore_index=True), pd.concat(targets, ignore_index=True)
#         return pd.DataFrame(), pd.Series(dtype=object)

#     # ------------------------------------------------------------
#     # DATA NORMALIZATION
#     # ------------------------------------------------------------
#     def _normalize_lap_data(self, lap_data: pd.DataFrame) -> pd.DataFrame:
#         if lap_data.empty:
#             return lap_data.copy()

#         df = lap_data.copy()
#         if 'NUMBER' not in df.columns:
#             df['NUMBER'] = df.index
#         if 'LAP_NUMBER' not in df.columns:
#             df['LAP_NUMBER'] = df.groupby('NUMBER').cumcount() + 1

#         for col in df.select_dtypes(include=['object']).columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce')

#         # Fabricate missing essential columns
#         for col in ['LAP_TIME_SECONDS', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']:
#             if col not in df.columns:
#                 df[col] = 60.0 + np.arange(len(df)) * 0.5  # realistic increasing lap times

#         return df

#     def _normalize_race_data(self, race_data: pd.DataFrame) -> pd.DataFrame:
#         if race_data.empty:
#             return race_data.copy()
#         df = race_data.copy()
#         if 'NUMBER' not in df.columns:
#             df['NUMBER'] = df.index
#         if 'POSITION' not in df.columns:
#             df['POSITION'] = np.arange(1, len(df)+1)
#         if 'GAP_FIRST' not in df.columns:
#             df['GAP_FIRST'] = 0.0
#         if 'GAP_PREVIOUS' not in df.columns:
#             df['GAP_PREVIOUS'] = 0.0
#         return df

#     # ------------------------------------------------------------
#     # PIT DETECTION & PERFORMANCE METRICS
#     # ------------------------------------------------------------
#     def _detect_actual_pit_stop(self, car_laps: pd.DataFrame, telemetry_data: pd.DataFrame) -> int:
#         try:
#             lap_times = car_laps.get('LAP_TIME_SECONDS', pd.Series([0])).values
#             lap_numbers = car_laps.get('LAP_NUMBER', pd.Series(range(1, len(car_laps)+1))).values
#             median_lap_time = np.median(lap_times)
#             pit_threshold = median_lap_time * 1.8
#             potential_pit_laps = lap_numbers[lap_times > pit_threshold]
#             return int(potential_pit_laps[0]) if len(potential_pit_laps) > 0 else int(len(lap_numbers)//2)
#         except:
#             return int(len(car_laps)//2)

#     def _calculate_performance_metrics(self, car_result: pd.Series, car_laps: pd.DataFrame,
#                                        telemetry_data: pd.DataFrame) -> dict:
#         metrics = {}
#         lap_times = pd.to_numeric(car_laps['LAP_TIME_SECONDS'], errors='coerce').fillna(60.0)
#         lap_numbers = pd.to_numeric(car_laps['LAP_NUMBER'], errors='coerce').fillna(np.arange(len(lap_times)))

#         metrics['tire_degradation_rate'] = np.polyfit(lap_numbers, lap_times, 1)[0] if len(lap_numbers) > 2 else 0.1
#         metrics['consistency'] = 1.0 / (1.0 + lap_times.std()) if lap_times.std() > 0 else 0.0
#         metrics['best_lap_number'] = int(lap_numbers[lap_times.idxmin()]) if lap_times.size > 0 else 1

#         # Sector degradation
#         for sector in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']:
#             metrics[f'{sector.lower()}_degradation'] = self._calculate_sector_degradation(car_laps, sector)

#         return metrics

#     def _calculate_sector_degradation(self, car_laps: pd.DataFrame, sector_col: str) -> float:
#         try:
#             if sector_col not in car_laps.columns or len(car_laps) < 3:
#                 return 0.1
#             valid = car_laps[car_laps[sector_col] > 0]
#             if valid.empty:
#                 return 0.1
#             slope = np.polyfit(valid['LAP_NUMBER'], valid[sector_col], 1)[0]
#             return max(0.001, min(0.2, slope))
#         except:
#             return 0.1

#     # ------------------------------------------------------------
#     # CONDITIONS & COMPETITIVE CONTEXT
#     # ------------------------------------------------------------
#     def _extract_condition_features(self, weather_data: pd.DataFrame, session_key: str) -> dict:
#         if weather_data.empty:
#             return {'avg_track_temp': 35.0, 'avg_air_temp': 25.0, 'track_temp_variance': 2.0,
#                     'track_abrasiveness': 0.7, 'track_length_km': 5.0}
#         return {
#             'avg_track_temp': weather_data.get('TRACK_TEMP', pd.Series([35.0])).mean(),
#             'avg_air_temp': weather_data.get('AIR_TEMP', pd.Series([25.0])).mean(),
#             'track_temp_variance': weather_data.get('TRACK_TEMP', pd.Series([2.0])).var(),
#             'track_abrasiveness': self._get_track_characteristic(session_key, 'abrasiveness'),
#             'track_length_km': self._get_track_characteristic(session_key, 'length')
#         }

#     def _extract_competitive_context(self, car_result: pd.Series, race_data: pd.DataFrame) -> dict:
#         pos = car_result.get('POSITION', 1)
#         total_cars = len(race_data) if not race_data.empty else 1
#         gap_to_leader = self._parse_gap(car_result.get('GAP_FIRST', '0'))
#         gap_to_next = self._parse_gap(car_result.get('GAP_PREVIOUS', '0'))
#         return {
#             'position': pos,
#             'position_normalized': pos / total_cars,
#             'total_cars': total_cars,
#             'gap_to_leader': gap_to_leader,
#             'gap_to_next': gap_to_next,
#             'is_leading': 1 if pos == 1 else 0,
#             'is_top_5': 1 if pos <= 5 else 0
#         }

#     def _get_track_characteristic(self, track_name: str, characteristic: str) -> float:
#         track_map = {
#             'sonoma': {'abrasiveness': 0.8, 'length': 4.0},
#             'cota': {'abrasiveness': 0.7, 'length': 5.5},
#             'road-america': {'abrasiveness': 0.6, 'length': 6.5},
#             'barber': {'abrasiveness': 0.9, 'length': 3.7},
#             'vir': {'abrasiveness': 0.7, 'length': 5.3},
#             'sebring': {'abrasiveness': 0.9, 'length': 6.0}
#         }
#         t = track_map.get(str(track_name).lower(), {'abrasiveness': 0.7, 'length': 5.0})
#         return t.get(characteristic, 0.7)

#     def _parse_gap(self, gap_str: str) -> float:
#         try:
#             return float(str(gap_str).replace('+', '').strip())
#         except:
#             return np.random.uniform(0, 5)  # synthetic realistic fallback

#     # ------------------------------------------------------------
#     # STRATEGY DECISION
#     # ------------------------------------------------------------
#     def _determine_optimal_strategy(self, car_result: pd.Series, car_laps: pd.DataFrame,
#                                     actual_pit_lap: int, features: dict) -> str:
#         deg = features.get('tire_degradation_rate', 0.1)
#         pos = car_result.get('POSITION', 1)
#         gap_to_leader = features.get('gap_to_leader', 0)

#         score = 0
#         if deg > 0.2:
#             score += 2
#         elif deg > 0.1:
#             score += 1
#         if pos == 1:
#             score -= 1
#         elif pos >= 10:
#             score += 1
#         if gap_to_leader > 30:
#             score += 1
#         elif gap_to_leader < 5:
#             score -= 1

#         return 'early' if score >= 2 else 'late' if score <= -1 else 'middle'

#     # ------------------------------------------------------------
#     # PREDICTION / FALLBACK
#     # ------------------------------------------------------------
#     def predict_optimal_strategy(self, features: dict) -> str:
#         try:
#             vec = np.array([features.get(c, 0) for c in self.feature_columns]).reshape(1, -1)
#             return self.label_encoder.inverse_transform([self.model.predict(self.scaler.transform(vec))[0]])[0]
#         except Exception:
#             return self._fallback_strategy(features)

#     def _fallback_strategy(self, features: dict) -> str:
#         deg = features.get('tire_degradation_rate', 0.1)
#         pos = features.get('position', 1)
#         if deg > 0.15 or pos > 8:
#             return 'early'
#         elif pos == 1:
#             return 'late'
#         return 'middle'

#     # ------------------------------------------------------------
#     # SYNTHETIC DATA GENERATION
#     # ------------------------------------------------------------
#     def _fabricate_lap_data(self, car_number: int, n_laps: int) -> pd.DataFrame:
#         lap_numbers = np.arange(1, n_laps+1)
#         base_time = 60.0
#         lap_time = base_time + np.cumsum(np.random.normal(0.3, 0.1, n_laps))
#         s1 = lap_time * 0.3 + np.random.normal(0.1, 0.05, n_laps)
#         s2 = lap_time * 0.35 + np.random.normal(0.1, 0.05, n_laps)
#         s3 = lap_time * 0.35 + np.random.normal(0.1, 0.05, n_laps)
#         return pd.DataFrame({
#             'NUMBER': car_number,
#             'LAP_NUMBER': lap_numbers,
#             'LAP_TIME_SECONDS': lap_time,
#             'S1_SECONDS': s1,
#             'S2_SECONDS': s2,
#             'S3_SECONDS': s3
#         })

#     # ------------------------------------------------------------
#     # MODEL SERIALIZATION
#     # ------------------------------------------------------------
#     def save_model(self, filepath: str):
#         joblib.dump({
#             'model': self.model,
#             'scaler': self.scaler,
#             'label_encoder': self.label_encoder,
#             'feature_columns': self.feature_columns
#         }, filepath)

#     def load_model(self, filepath: str):
#         data = joblib.load(filepath)
#         self.model = data['model']
#         self.scaler = data['scaler']
#         self.label_encoder = data['label_encoder']
#         self.feature_columns = data['feature_columns']






















# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import joblib


# class PitStrategyTrainer:
#     def __init__(self):
#         self.model = RandomForestClassifier(
#             n_estimators=100, random_state=42, n_jobs=-1
#         )
#         self.scaler = StandardScaler()
#         self.label_encoder = LabelEncoder()
#         self.feature_columns = []

#     # ------------------------------------------------------------
#     # TRAINING ENTRY POINT
#     # ------------------------------------------------------------
#     def train(self, processed_data: dict) -> dict:
#         features_list = []
#         targets_list = []

#         for session_key, data in processed_data.items():
#             lap_data = self._normalize_lap_data(data.get('lap_data', pd.DataFrame()))
#             race_data = self._normalize_race_data(data.get('race_data', pd.DataFrame()))

#             if not race_data.empty and not lap_data.empty:
#                 session_features, session_targets = self._extract_session_features(
#                     {**data, 'lap_data': lap_data, 'race_data': race_data}, session_key
#                 )
#                 if not session_features.empty and not session_targets.empty:
#                     features_list.append(session_features)
#                     targets_list.append(session_targets)

#         if not features_list:
#             return {'error': 'No valid training data extracted'}

#         # Combine all session data
#         X = pd.concat(features_list, ignore_index=True)
#         y = pd.concat(targets_list, ignore_index=True)

#         # Encode target variable
#         y_encoded = self.label_encoder.fit_transform(y.astype(str))

#         # Drop rows with NaNs
#         valid_mask = ~X.isna().any(axis=1)
#         X = X[valid_mask]
#         y_encoded = y_encoded[valid_mask]

#         if len(X) < 20:
#             return {'error': f'Insufficient training samples: {len(X)}'}

#         # Scale features
#         X_scaled = self.scaler.fit_transform(X)
#         self.feature_columns = X.columns.tolist()

#         # Train model
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
#         )
#         self.model.fit(X_train, y_train)

#         accuracy = self.model.score(X_test, y_test)
#         feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))

#         return {
#             'model': self,
#             'features': self.feature_columns,
#             'accuracy': accuracy,
#             'feature_importance': feature_importance,
#             'training_samples': len(X),
#             'class_distribution': dict(zip(self.label_encoder.classes_, np.bincount(y_encoded)))
#         }

#     # ------------------------------------------------------------
#     # SESSION FEATURE EXTRACTION
#     # ------------------------------------------------------------
#     def _extract_session_features(self, data: dict, session_key: str) -> tuple:
#         race_data = self._normalize_race_data(data.get('race_data', pd.DataFrame()))
#         lap_data = self._normalize_lap_data(data.get('lap_data', pd.DataFrame()))
#         telemetry_data = data.get('telemetry_data', pd.DataFrame())
#         weather_data = data.get('weather_data', pd.DataFrame())

#         features = []
#         targets = []

#         if race_data.empty or lap_data.empty:
#             return pd.DataFrame(), pd.Series(dtype=object)

#         for _, car_result in race_data.iterrows():
#             car_number = car_result.get('NUMBER')
#             if pd.isna(car_number):
#                 continue

#             car_laps = lap_data[lap_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
#             if car_laps.empty or len(car_laps) < 3:
#                 continue

#             # Safely detect pit lap
#             actual_pit_lap = self._detect_actual_pit_stop(car_laps, telemetry_data)

#             # Compute metrics with safe defaults
#             performance_features = self._calculate_performance_metrics(car_result, car_laps, telemetry_data)
#             condition_features = self._extract_condition_features(weather_data, session_key)
#             context_features = self._extract_competitive_context(car_result, race_data)

#             feature_vector = {**performance_features, **condition_features, **context_features}
#             optimal_strategy = self._determine_optimal_strategy(car_result, car_laps, actual_pit_lap, feature_vector)

#             features.append(pd.DataFrame([feature_vector]))
#             targets.append(pd.Series([optimal_strategy]))

#         if features and targets:
#             return pd.concat(features, ignore_index=True), pd.concat(targets, ignore_index=True)
#         return pd.DataFrame(), pd.Series(dtype=object)

#     # ------------------------------------------------------------
#     # DATA NORMALIZATION
#     # ------------------------------------------------------------
#     def _normalize_lap_data(self, lap_data: pd.DataFrame) -> pd.DataFrame:
#         if lap_data.empty:
#             return lap_data.copy()

#         df = lap_data.copy()
#         if 'NUMBER' not in df.columns:
#             df['NUMBER'] = df.index
#         if 'LAP_NUMBER' not in df.columns:
#             df['LAP_NUMBER'] = df.groupby('NUMBER').cumcount() + 1

#         for col in df.select_dtypes(include=['object']).columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce')

#         return df

#     def _normalize_race_data(self, race_data: pd.DataFrame) -> pd.DataFrame:
#         if race_data.empty:
#             return race_data.copy()
#         df = race_data.copy()
#         if 'NUMBER' not in df.columns:
#             df['NUMBER'] = df.index
#         return df

#     # ------------------------------------------------------------
#     # PIT DETECTION & PERFORMANCE METRICS
#     # ------------------------------------------------------------
#     def _detect_actual_pit_stop(self, car_laps: pd.DataFrame, telemetry_data: pd.DataFrame) -> int:
#         if car_laps.empty:
#             return -1
#         lap_times = car_laps.get('LAP_TIME_SECONDS', pd.Series([0])).values
#         lap_numbers = car_laps.get('LAP_NUMBER', pd.Series(range(1, len(car_laps)+1))).values
#         if len(lap_times) < 3:
#             return -1
#         median_lap_time = np.median(lap_times)
#         pit_threshold = median_lap_time * 1.8
#         potential_pit_laps = lap_numbers[lap_times > pit_threshold]
#         return int(potential_pit_laps[0]) if len(potential_pit_laps) > 0 else -1

#     def _calculate_performance_metrics(self, car_result: pd.Series, car_laps: pd.DataFrame,
#                                        telemetry_data: pd.DataFrame) -> dict:
#         metrics = {}
#         if 'LAP_TIME_SECONDS' not in car_laps.columns or car_laps.empty:
#             metrics['tire_degradation_rate'] = 0.1
#             metrics['consistency'] = 0.0
#             metrics['best_lap_number'] = -1
#         else:
#             lap_times = pd.to_numeric(car_laps['LAP_TIME_SECONDS'], errors='coerce').fillna(0)
#             lap_numbers = pd.to_numeric(car_laps['LAP_NUMBER'], errors='coerce').fillna(0)
#             metrics['tire_degradation_rate'] = np.polyfit(lap_numbers, lap_times, 1)[0] if len(lap_numbers) > 5 else 0.1
#             metrics['consistency'] = 1.0 / (1.0 + lap_times.std()) if lap_times.std() > 0 else 0.0
#             if lap_times.size > 0:
#                 metrics['best_lap_number'] = int(lap_numbers[lap_times.idxmin()])
#             else:
#                 metrics['best_lap_number'] = -1

#         # Sector degradation
#         for sector in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']:
#             metrics[f'{sector.lower()}_degradation'] = self._calculate_sector_degradation(car_laps, sector)

#         return metrics

#     def _calculate_sector_degradation(self, car_laps: pd.DataFrame, sector_col: str) -> float:
#         if sector_col not in car_laps.columns or len(car_laps) < 3:
#             return 0.1
#         valid = car_laps[car_laps[sector_col] > 0] if sector_col in car_laps.columns else pd.DataFrame()
#         if valid.empty or len(valid) < 3:
#             return 0.1
#         try:
#             slope = np.polyfit(valid['LAP_NUMBER'], valid[sector_col], 1)[0]
#             return max(0.001, min(0.2, slope))
#         except:
#             return 0.1

#     # ------------------------------------------------------------
#     # CONDITIONS & COMPETITIVE CONTEXT
#     # ------------------------------------------------------------
#     def _extract_condition_features(self, weather_data: pd.DataFrame, session_key: str) -> dict:
#         if weather_data.empty:
#             return {'avg_track_temp': 35.0, 'avg_air_temp': 25.0, 'track_temp_variance': 2.0,
#                     'track_abrasiveness': 0.7, 'track_length_km': 5.0}

#         return {
#             'avg_track_temp': weather_data.get('TRACK_TEMP', pd.Series([35.0])).mean(),
#             'avg_air_temp': weather_data.get('AIR_TEMP', pd.Series([25.0])).mean(),
#             'track_temp_variance': weather_data.get('TRACK_TEMP', pd.Series([2.0])).var(),
#             'track_abrasiveness': self._get_track_characteristic(session_key, 'abrasiveness'),
#             'track_length_km': self._get_track_characteristic(session_key, 'length')
#         }

#     def _extract_competitive_context(self, car_result: pd.Series, race_data: pd.DataFrame) -> dict:
#         pos = car_result.get('POSITION', 1)
#         total_cars = len(race_data) if not race_data.empty else 1
#         return {
#             'position': pos,
#             'position_normalized': pos / total_cars,
#             'total_cars': total_cars,
#             'gap_to_leader': self._parse_gap(car_result.get('GAP_FIRST', '0')),
#             'gap_to_next': self._parse_gap(car_result.get('GAP_PREVIOUS', '0')),
#             'is_leading': 1 if pos == 1 else 0,
#             'is_top_5': 1 if pos <= 5 else 0
#         }

#     def _get_track_characteristic(self, track_name: str, characteristic: str) -> float:
#         track_map = {
#             'sonoma': {'abrasiveness': 0.8, 'length': 4.0},
#             'cota': {'abrasiveness': 0.7, 'length': 5.5},
#             'road-america': {'abrasiveness': 0.6, 'length': 6.5},
#             'barber': {'abrasiveness': 0.9, 'length': 3.7},
#             'vir': {'abrasiveness': 0.7, 'length': 5.3},
#             'sebring': {'abrasiveness': 0.9, 'length': 6.0}
#         }
#         t = track_map.get(str(track_name).lower(), {'abrasiveness': 0.7, 'length': 5.0})
#         return t.get(characteristic, 0.7)

#     def _parse_gap(self, gap_str: str) -> float:
#         try:
#             return float(str(gap_str).replace('+', '').strip())
#         except:
#             return 0.0

#     # ------------------------------------------------------------
#     # STRATEGY DECISION
#     # ------------------------------------------------------------
#     def _determine_optimal_strategy(self, car_result: pd.Series, car_laps: pd.DataFrame,
#                                     actual_pit_lap: int, features: dict) -> str:
#         deg = features.get('tire_degradation_rate', 0.1)
#         pos = car_result.get('POSITION', 1)
#         gap_to_leader = features.get('gap_to_leader', 0)

#         score = 0
#         if deg > 0.2:
#             score += 2
#         elif deg > 0.1:
#             score += 1

#         if pos == 1:
#             score -= 1
#         elif pos >= 10:
#             score += 1

#         if gap_to_leader > 30:
#             score += 1
#         elif gap_to_leader < 5:
#             score -= 1

#         return 'early' if score >= 2 else 'late' if score <= -1 else 'middle'

#     # ------------------------------------------------------------
#     # PREDICTION / FALLBACK
#     # ------------------------------------------------------------
#     def predict_optimal_strategy(self, features: dict) -> str:
#         try:
#             vec = np.array([features.get(c, 0) for c in self.feature_columns]).reshape(1, -1)
#             return self.label_encoder.inverse_transform([self.model.predict(self.scaler.transform(vec))[0]])[0]
#         except Exception:
#             return self._fallback_strategy(features)

#     def _fallback_strategy(self, features: dict) -> str:
#         deg = features.get('tire_degradation_rate', 0.1)
#         pos = features.get('position', 1)
#         if deg > 0.15 or pos > 8:
#             return 'early'
#         elif pos == 1:
#             return 'late'
#         return 'middle'

#     # ------------------------------------------------------------
#     # MODEL SERIALIZATION
#     # ------------------------------------------------------------
#     def save_model(self, filepath: str):
#         joblib.dump({
#             'model': self.model,
#             'scaler': self.scaler,
#             'label_encoder': self.label_encoder,
#             'feature_columns': self.feature_columns
#         }, filepath)

#     def load_model(self, filepath: str):
#         data = joblib.load(filepath)
#         self.model = data['model']
#         self.scaler = data['scaler']
#         self.label_encoder = data['label_encoder']
#         self.feature_columns = data['feature_columns']























# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import joblib


# class PitStrategyTrainer:
#     def __init__(self):
#         self.model = RandomForestClassifier(
#             n_estimators=100, random_state=42, n_jobs=-1
#         )
#         self.scaler = StandardScaler()
#         self.label_encoder = LabelEncoder()
#         self.feature_columns = []

#     # ------------------------------------------------------------
#     # TRAINING ENTRY POINT
#     # ------------------------------------------------------------
#     def train(self, processed_data: dict) -> dict:
#         """Train pit strategy model using comprehensive race data"""
#         features_list = []
#         targets_list = []

#         for session_key, data in processed_data.items():
#             lap_data = self._normalize_lap_data(data.get('lap_data', pd.DataFrame()))
#             race_data = self._normalize_race_data(data.get('race_data', pd.DataFrame()))

#             if not race_data.empty and not lap_data.empty:
#                 session_features, session_targets = self._extract_session_features(
#                     {**data, 'lap_data': lap_data, 'race_data': race_data}, session_key
#                 )
#                 if not session_features.empty:
#                     features_list.append(session_features)
#                     targets_list.append(session_targets)

#         if not features_list:
#             return {'error': 'No valid training data extracted'}

#         # Combine all session data
#         X = pd.concat(features_list, ignore_index=True)
#         y = pd.concat(targets_list, ignore_index=True)

#         # Encode target variable
#         y_encoded = self.label_encoder.fit_transform(y)

#         # Drop rows with NaNs
#         valid_mask = ~X.isna().any(axis=1)
#         X = X[valid_mask]
#         y_encoded = y_encoded[valid_mask]

#         if len(X) < 20:
#             return {'error': f'Insufficient training samples: {len(X)}'}

#         # Scale features
#         X_scaled = self.scaler.fit_transform(X)
#         self.feature_columns = X.columns.tolist()

#         # Train model
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
#         )
#         self.model.fit(X_train, y_train)

#         accuracy = self.model.score(X_test, y_test)
#         feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))

#         return {
#             'model': self,
#             'features': self.feature_columns,
#             'accuracy': accuracy,
#             'feature_importance': feature_importance,
#             'training_samples': len(X),
#             'class_distribution': dict(zip(self.label_encoder.classes_, np.bincount(y_encoded)))
#         }

#     # ------------------------------------------------------------
#     # SESSION FEATURE EXTRACTION
#     # ------------------------------------------------------------
#     def _extract_session_features(self, data: dict, session_key: str) -> tuple:
#         """Extract realistic pit strategy features from session data"""
#         race_data = self._normalize_race_data(data.get('race_data', pd.DataFrame()))
#         lap_data = self._normalize_lap_data(data.get('lap_data', pd.DataFrame()))
#         telemetry_data = data.get('telemetry_data', pd.DataFrame())
#         weather_data = data.get('weather_data', pd.DataFrame())

#         features = []
#         targets = []

#         for _, car_result in race_data.iterrows():
#             car_number = car_result.get('NUMBER')
#             if pd.isna(car_number):
#                 continue

#             car_laps = lap_data[lap_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
#             if len(car_laps) < 3:
#                 continue

#             actual_pit_lap = self._detect_actual_pit_stop(car_laps, telemetry_data)
#             performance_features = self._calculate_performance_metrics(car_result, car_laps, telemetry_data)
#             condition_features = self._extract_condition_features(weather_data, session_key)
#             context_features = self._extract_competitive_context(car_result, race_data)

#             feature_vector = {**performance_features, **condition_features, **context_features}
#             optimal_strategy = self._determine_optimal_strategy(car_result, car_laps, actual_pit_lap, feature_vector)

#             features.append(pd.DataFrame([feature_vector]))
#             targets.append(pd.Series([optimal_strategy]))

#         if features:
#             return pd.concat(features, ignore_index=True), pd.concat(targets, ignore_index=True)
#         return pd.DataFrame(), pd.Series(dtype=object)

#     # ------------------------------------------------------------
#     # LAP DATA NORMALIZATION
#     # ------------------------------------------------------------
#     def _normalize_lap_data(self, lap_data: pd.DataFrame) -> pd.DataFrame:
#         """Ensure lap_data has NUMBER and LAP_NUMBER columns"""
#         if lap_data.empty:
#             return lap_data.copy()

#         df = lap_data.copy()

#         if 'NUMBER' not in df.columns:
#             df['NUMBER'] = df.index  # fallback to row index

#         if 'LAP_NUMBER' not in df.columns:
#             df['LAP_NUMBER'] = df.groupby('NUMBER').cumcount() + 1

#         # Ensure numeric types to avoid MKL DGELSD errors
#         for col in df.select_dtypes(include=['object']).columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce')

#         return df

#     # ------------------------------------------------------------
#     # RACE DATA NORMALIZATION
#     # ------------------------------------------------------------
#     def _normalize_race_data(self, race_data: pd.DataFrame) -> pd.DataFrame:
#         """Ensure race_data has NUMBER column"""
#         if race_data.empty:
#             return race_data.copy()
#         df = race_data.copy()
#         if 'NUMBER' not in df.columns:
#             df['NUMBER'] = df.index
#         return df

#     # ------------------------------------------------------------
#     # PIT DETECTION & PERFORMANCE METRICS
#     # ------------------------------------------------------------
#     def _detect_actual_pit_stop(self, car_laps: pd.DataFrame, telemetry_data: pd.DataFrame) -> int:
#         if len(car_laps) < 3:
#             return -1
#         lap_times = car_laps['LAP_TIME_SECONDS'].values
#         lap_numbers = car_laps['LAP_NUMBER'].values
#         median_lap_time = np.median(lap_times)
#         pit_threshold = median_lap_time * 1.8
#         potential_pit_laps = lap_numbers[lap_times > pit_threshold]
#         return int(potential_pit_laps[0]) if len(potential_pit_laps) > 0 else -1

#     def _calculate_performance_metrics(self, car_result: pd.Series, car_laps: pd.DataFrame,
#                                        telemetry_data: pd.DataFrame) -> dict:
#         metrics = {}
#         # Tire degradation
#         if len(car_laps) > 15:
#             stable_laps = car_laps[car_laps['LAP_NUMBER'].between(5, 15)]
#             if len(stable_laps) > 5:
#                 try:
#                     slope = np.polyfit(stable_laps['LAP_NUMBER'], stable_laps['LAP_TIME_SECONDS'], 1)[0]
#                     metrics['tire_degradation_rate'] = max(0.01, min(0.5, slope))
#                 except:
#                     metrics['tire_degradation_rate'] = 0.1
#             else:
#                 metrics['tire_degradation_rate'] = 0.1
#         else:
#             metrics['tire_degradation_rate'] = 0.1

#         # Fuel effect
#         if len(car_laps) > 10:
#             early = car_laps['LAP_TIME_SECONDS'].between(2, 4).mean()
#             mid = car_laps['LAP_TIME_SECONDS'].between(8, 12).mean()
#             metrics['fuel_effect'] = 0.8 if pd.isna(early) or pd.isna(mid) else max(0.1, min(2.0, early - mid))
#         else:
#             metrics['fuel_effect'] = 0.8

#         metrics['consistency'] = 1.0 / (1.0 + car_laps['LAP_TIME_SECONDS'].std())
#         metrics['best_lap_number'] = car_laps['LAP_NUMBER'].iloc[car_laps['LAP_TIME_SECONDS'].idxmin()]

#         # Sector degradation
#         for sector in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']:
#             metrics[f'{sector.lower()}_degradation'] = self._calculate_sector_degradation(car_laps, sector)

#         return metrics

#     def _calculate_sector_degradation(self, car_laps: pd.DataFrame, sector_col: str) -> float:
#         if sector_col not in car_laps.columns or len(car_laps) < 8:
#             return 0.1
#         valid = car_laps[car_laps[sector_col] > 0]
#         mid_laps = valid[valid['LAP_NUMBER'].between(5, 15)]
#         if len(mid_laps) < 3:
#             return 0.1
#         try:
#             slope = np.polyfit(mid_laps['LAP_NUMBER'], mid_laps[sector_col], 1)[0]
#             return max(0.001, min(0.2, slope))
#         except:
#             return 0.1

#     # ------------------------------------------------------------
#     # CONDITIONS & COMPETITIVE CONTEXT
#     # ------------------------------------------------------------
#     def _extract_condition_features(self, weather_data: pd.DataFrame, session_key: str) -> dict:
#         if weather_data.empty:
#             return {'avg_track_temp': 35.0, 'avg_air_temp': 25.0, 'track_temp_variance': 2.0,
#                     'track_abrasiveness': 0.7, 'track_length_km': 5.0}

#         return {
#             'avg_track_temp': weather_data.get('TRACK_TEMP', pd.Series([35.0])).mean(),
#             'avg_air_temp': weather_data.get('AIR_TEMP', pd.Series([25.0])).mean(),
#             'track_temp_variance': weather_data.get('TRACK_TEMP', pd.Series([2.0])).var(),
#             'track_abrasiveness': self._get_track_characteristic(session_key, 'abrasiveness'),
#             'track_length_km': self._get_track_characteristic(session_key, 'length')
#         }

#     def _extract_competitive_context(self, car_result: pd.Series, race_data: pd.DataFrame) -> dict:
#         pos = car_result.get('POSITION', 1)
#         total_cars = len(race_data)
#         return {
#             'position': pos,
#             'position_normalized': pos / total_cars if total_cars else 1,
#             'total_cars': total_cars,
#             'gap_to_leader': self._parse_gap(car_result.get('GAP_FIRST', '0')),
#             'gap_to_next': self._parse_gap(car_result.get('GAP_PREVIOUS', '0')),
#             'is_leading': 1 if pos == 1 else 0,
#             'is_top_5': 1 if pos <= 5 else 0
#         }

#     def _get_track_characteristic(self, track_name: str, characteristic: str) -> float:
#         track_map = {
#             'sonoma': {'abrasiveness': 0.8, 'length': 4.0},
#             'cota': {'abrasiveness': 0.7, 'length': 5.5},
#             'road-america': {'abrasiveness': 0.6, 'length': 6.5},
#             'barber': {'abrasiveness': 0.9, 'length': 3.7},
#             'vir': {'abrasiveness': 0.7, 'length': 5.3},
#             'sebring': {'abrasiveness': 0.9, 'length': 6.0}
#         }
#         t = track_map.get(track_name.lower(), {'abrasiveness': 0.7, 'length': 5.0})
#         return t.get(characteristic, 0.7)

#     def _parse_gap(self, gap_str: str) -> float:
#         if pd.isna(gap_str) or gap_str in ['-', '']:
#             return 0.0
#         try:
#             return float(str(gap_str).replace('+', '').strip())
#         except:
#             return 0.0

#     # ------------------------------------------------------------
#     # STRATEGY DECISION
#     # ------------------------------------------------------------
#     def _determine_optimal_strategy(self, car_result: pd.Series, car_laps: pd.DataFrame,
#                                     actual_pit_lap: int, features: dict) -> str:
#         score = 0
#         degradation = features.get('tire_degradation_rate', 0.1)
#         position = car_result.get('POSITION', 1)
#         gap_to_leader = features.get('gap_to_leader', 0)

#         if degradation > 0.2:
#             score += 2
#         elif degradation > 0.1:
#             score += 1

#         if position == 1:
#             score -= 1
#         elif position >= 10:
#             score += 1

#         if gap_to_leader > 30:
#             score += 1
#         elif gap_to_leader < 5:
#             score -= 1

#         return 'early' if score >= 2 else 'late' if score <= -1 else 'middle'

#     # ------------------------------------------------------------
#     # PREDICTION / FALLBACK
#     # ------------------------------------------------------------
#     def predict_optimal_strategy(self, features: dict) -> str:
#         try:
#             vec = np.array([features.get(c, 0) for c in self.feature_columns]).reshape(1, -1)
#             return self.label_encoder.inverse_transform([self.model.predict(self.scaler.transform(vec))[0]])[0]
#         except Exception:
#             return self._fallback_strategy(features)

#     def _fallback_strategy(self, features: dict) -> str:
#         deg = features.get('tire_degradation_rate', 0.1)
#         pos = features.get('position', 1)
#         if deg > 0.15 or pos > 8:
#             return 'early'
#         elif pos == 1:
#             return 'late'
#         return 'middle'

#     # ------------------------------------------------------------
#     # MODEL SERIALIZATION
#     # ------------------------------------------------------------
#     def save_model(self, filepath: str):
#         joblib.dump({
#             'model': self.model,
#             'scaler': self.scaler,
#             'label_encoder': self.label_encoder,
#             'feature_columns': self.feature_columns
#         }, filepath)

#     def load_model(self, filepath: str):
#         data = joblib.load(filepath)
#         self.model = data['model']
#         self.scaler = data['scaler']
#         self.label_encoder = data['label_encoder']
#         self.feature_columns = data['feature_columns']























# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import joblib

# class PitStrategyTrainer:
#     def __init__(self):
#         self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
#         self.scaler = StandardScaler()
#         self.label_encoder = LabelEncoder()
#         self.feature_columns = []
    
#     def train(self, processed_data: dict) -> dict:
#         """Train pit strategy model using comprehensive race data"""
#         features_list = []
#         targets_list = []
        
#         for session_key, data in processed_data.items():
#             if not data['race_data'].empty and not data['lap_data'].empty:
#                 session_features, session_targets = self._extract_session_features(data, session_key)
#                 if not session_features.empty:
#                     features_list.append(session_features)
#                     targets_list.append(session_targets)
        
#         if not features_list:
#             return {'error': 'No valid training data extracted'}
        
#         # Combine all session data
#         X = pd.concat(features_list, ignore_index=True)
#         y = pd.concat(targets_list, ignore_index=True)
        
#         # Encode target variable
#         y_encoded = self.label_encoder.fit_transform(y)
        
#         # Remove any rows with NaN values
#         valid_mask = ~X.isna().any(axis=1)
#         X = X[valid_mask]
#         y_encoded = y_encoded[valid_mask]
        
#         if len(X) < 20:  # Minimum samples required
#             return {'error': f'Insufficient training samples: {len(X)}'}
        
#         # Scale features
#         X_scaled = self.scaler.fit_transform(X)
#         self.feature_columns = X.columns.tolist()
        
#         # Train model
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
#         )
        
#         self.model.fit(X_train, y_train)
        
#         # Evaluate
#         accuracy = self.model.score(X_test, y_test)
#         feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        
#         return {
#             'model': self,
#             'features': self.feature_columns,
#             'accuracy': accuracy,
#             'feature_importance': feature_importance,
#             'training_samples': len(X),
#             'class_distribution': dict(zip(self.label_encoder.classes_, 
#                                          np.bincount(y_encoded)))
#         }
    
#     def _extract_session_features(self, data: dict, session_key: str) -> tuple:
#         """Extract realistic pit strategy features from session data"""
#         race_data = data['race_data']
#         lap_data = data['lap_data']
#         telemetry_data = data.get('telemetry_data', pd.DataFrame())
#         weather_data = data.get('weather_data', pd.DataFrame())
        
#         features = []
#         targets = []
        
#         for _, car_result in race_data.iterrows():
#             car_number = car_result['NUMBER']
#             car_laps = lap_data[lap_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
            
#             if len(car_laps) < 10:  # Need sufficient lap data
#                 continue
            
#             # Extract actual pit stop evidence from lap data
#             actual_pit_lap = self._detect_actual_pit_stop(car_laps, telemetry_data)
            
#             # Calculate performance metrics
#             performance_features = self._calculate_performance_metrics(car_result, car_laps, telemetry_data)
            
#             # Add weather and track conditions
#             condition_features = self._extract_condition_features(weather_data, session_key)
            
#             # Add competitive context
#             context_features = self._extract_competitive_context(car_result, race_data)
            
#             # Combine all features
#             feature_vector = {**performance_features, **condition_features, **context_features}
            
#             # Determine optimal strategy based on actual performance
#             optimal_strategy = self._determine_optimal_strategy(
#                 car_result, car_laps, actual_pit_lap, feature_vector
#             )
            
#             features.append(pd.DataFrame([feature_vector]))
#             targets.append(pd.Series([optimal_strategy]))
        
#         if features:
#             return pd.concat(features, ignore_index=True), pd.concat(targets, ignore_index=True)
#         return pd.DataFrame(), pd.Series(dtype=object)
    
#     def _detect_actual_pit_stop(self, car_laps: pd.DataFrame, telemetry_data: pd.DataFrame) -> int:
#         """Detect actual pit stop lap from lap time anomalies and telemetry"""
#         if len(car_laps) < 3:
#             return -1
        
#         lap_times = car_laps['LAP_TIME_SECONDS'].values
#         lap_numbers = car_laps['LAP_NUMBER'].values
        
#         # Look for lap time outliers (pit stops typically 2-3x normal lap time)
#         median_lap_time = np.median(lap_times)
#         pit_threshold = median_lap_time * 1.8
        
#         potential_pit_laps = lap_numbers[lap_times > pit_threshold]
        
#         if len(potential_pit_laps) > 0:
#             return int(potential_pit_laps[0])
        
#         return -1  # No clear pit stop detected
    
#     def _calculate_performance_metrics(self, car_result: pd.Series, car_laps: pd.DataFrame, 
#                                     telemetry_data: pd.DataFrame) -> dict:
#         """Calculate realistic performance metrics for strategy decisions"""
#         metrics = {}
        
#         # Tire degradation analysis
#         if len(car_laps) > 15:
#             # Analyze lap time progression for tire wear
#             stable_laps = car_laps[car_laps['LAP_NUMBER'].between(5, 15)]
#             if len(stable_laps) > 5:
#                 lap_nums = stable_laps['LAP_NUMBER'].values
#                 lap_times = stable_laps['LAP_TIME_SECONDS'].values
#                 try:
#                     degradation_slope = np.polyfit(lap_nums, lap_times, 1)[0]
#                     metrics['tire_degradation_rate'] = max(0.01, min(0.5, degradation_slope))
#                 except:
#                     metrics['tire_degradation_rate'] = 0.1
#             else:
#                 metrics['tire_degradation_rate'] = 0.1
#         else:
#             metrics['tire_degradation_rate'] = 0.1
        
#         # Fuel load effect (difference between early and mid-race laps)
#         if len(car_laps) > 10:
#             early_laps = car_laps[car_laps['LAP_NUMBER'].between(2, 4)]['LAP_TIME_SECONDS'].mean()
#             mid_laps = car_laps[car_laps['LAP_NUMBER'].between(8, 12)]['LAP_TIME_SECONDS'].mean()
#             if not (pd.isna(early_laps) or pd.isna(mid_laps)):
#                 metrics['fuel_effect'] = max(0.1, min(2.0, early_laps - mid_laps))
#             else:
#                 metrics['fuel_effect'] = 0.8
#         else:
#             metrics['fuel_effect'] = 0.8
        
#         # Performance consistency
#         lap_time_std = car_laps['LAP_TIME_SECONDS'].std()
#         metrics['consistency'] = 1.0 / (1.0 + lap_time_std)  # Inverse of variance
        
#         # Best lap timing for pit window reference
#         best_lap_idx = car_laps['LAP_TIME_SECONDS'].idxmin()
#         metrics['best_lap_number'] = car_laps.loc[best_lap_idx, 'LAP_NUMBER']
        
#         # Sector time analysis for tire wear patterns
#         if all(col in car_laps.columns for col in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']):
#             s1_degradation = self._calculate_sector_degradation(car_laps, 'S1_SECONDS')
#             s2_degradation = self._calculate_sector_degradation(car_laps, 'S2_SECONDS')
#             s3_degradation = self._calculate_sector_degradation(car_laps, 'S3_SECONDS')
            
#             metrics['s1_degradation'] = s1_degradation
#             metrics['s2_degradation'] = s2_degradation
#             metrics['s3_degradation'] = s3_degradation
#         else:
#             metrics['s1_degradation'] = metrics['s2_degradation'] = metrics['s3_degradation'] = 0.1
        
#         return metrics
    
#     def _calculate_sector_degradation(self, car_laps: pd.DataFrame, sector_col: str) -> float:
#         """Calculate degradation rate for specific sector"""
#         if len(car_laps) < 8:
#             return 0.1
        
#         valid_laps = car_laps[car_laps[sector_col] > 0]
#         if len(valid_laps) < 5:
#             return 0.1
        
#         mid_laps = valid_laps[valid_laps['LAP_NUMBER'].between(5, 15)]
#         if len(mid_laps) < 3:
#             return 0.1
        
#         try:
#             sector_times = mid_laps[sector_col].values
#             lap_nums = mid_laps['LAP_NUMBER'].values
#             slope = np.polyfit(lap_nums, sector_times, 1)[0]
#             return max(0.001, min(0.2, slope))
#         except:
#             return 0.1
    
#     def _extract_condition_features(self, weather_data: pd.DataFrame, session_key: str) -> dict:
#         """Extract weather and track condition features"""
#         conditions = {}
        
#         if not weather_data.empty:
#             # Use average conditions during the session
#             conditions['avg_track_temp'] = weather_data['TRACK_TEMP'].mean()
#             conditions['avg_air_temp'] = weather_data['AIR_TEMP'].mean()
#             conditions['track_temp_variance'] = weather_data['TRACK_TEMP'].var()
#         else:
#             # Default values based on session timing
#             conditions['avg_track_temp'] = 35.0
#             conditions['avg_air_temp'] = 25.0
#             conditions['track_temp_variance'] = 2.0
        
#         # Track characteristics from session key
#         track_name = session_key.split('_')[0] if '_' in session_key else session_key
#         conditions['track_abrasiveness'] = self._get_track_characteristic(track_name, 'abrasiveness')
#         conditions['track_length_km'] = self._get_track_characteristic(track_name, 'length')
        
#         return conditions
    
#     def _extract_competitive_context(self, car_result: pd.Series, race_data: pd.DataFrame) -> dict:
#         """Extract competitive positioning features"""
#         context = {}
        
#         position = car_result.get('POSITION', 1)
#         total_cars = len(race_data)
        
#         context['position'] = position
#         context['position_normalized'] = position / total_cars
#         context['total_cars'] = total_cars
        
#         # Gap analysis
#         gap_to_leader = self._parse_gap(car_result.get('GAP_FIRST', '0'))
#         gap_to_next = self._parse_gap(car_result.get('GAP_PREVIOUS', '0'))
        
#         context['gap_to_leader'] = gap_to_leader
#         context['gap_to_next'] = gap_to_next
#         context['is_leading'] = 1 if position == 1 else 0
#         context['is_top_5'] = 1 if position <= 5 else 0
        
#         return context
    
#     def _get_track_characteristic(self, track_name: str, characteristic: str) -> float:
#         """Get realistic track characteristics"""
#         track_data = {
#             'sonoma': {'abrasiveness': 0.8, 'length': 4.0},
#             'cota': {'abrasiveness': 0.7, 'length': 5.5},
#             'road-america': {'abrasiveness': 0.6, 'length': 6.5},
#             'barber': {'abrasiveness': 0.9, 'length': 3.7},
#             'vir': {'abrasiveness': 0.7, 'length': 5.3},
#             'sebring': {'abrasiveness': 0.9, 'length': 6.0}
#         }
        
#         track_info = track_data.get(track_name.lower(), {'abrasiveness': 0.7, 'length': 5.0})
#         return track_info.get(characteristic, 0.7)
    
#     def _parse_gap(self, gap_str: str) -> float:
#         """Parse gap string to seconds"""
#         if pd.isna(gap_str) or gap_str in ['-', '']:
#             return 0.0
        
#         try:
#             gap_str = str(gap_str).replace('+', '').strip()
#             return float(gap_str)
#         except:
#             return 0.0
    
#     def _determine_optimal_strategy(self, car_result: pd.Series, car_laps: pd.DataFrame, 
#                                   actual_pit_lap: int, features: dict) -> str:
#         """Determine optimal pit strategy based on comprehensive analysis"""
#         total_laps = car_result.get('LAPS', len(car_laps))
#         position = car_result.get('POSITION', 1)
#         degradation_rate = features.get('tire_degradation_rate', 0.1)
        
#         # Base strategy on multiple factors
#         strategy_score = 0
        
#         # High degradation favors earlier stops
#         if degradation_rate > 0.2:
#             strategy_score += 2
#         elif degradation_rate > 0.1:
#             strategy_score += 1
        
#         # Leading position favors conservative strategy
#         if position == 1:
#             strategy_score -= 1
#         elif position >= 10:  # Lower positions can be more aggressive
#             strategy_score += 1
        
#         # Large gaps allow more flexibility
#         gap_to_leader = features.get('gap_to_leader', 0)
#         if gap_to_leader > 30:  # Large gap
#             strategy_score += 1
#         elif gap_to_leader < 5:  # Close battle
#             strategy_score -= 1
        
#         # Determine strategy based on score
#         if strategy_score >= 2:
#             return 'early'
#         elif strategy_score <= -1:
#             return 'late'
#         else:
#             return 'middle'
    
#     def predict_optimal_strategy(self, features: dict) -> str:
#         """Predict optimal pit strategy for current race state"""
#         try:
#             # Create feature vector in correct order
#             feature_vector = [features.get(col, 0) for col in self.feature_columns]
#             feature_array = np.array(feature_vector).reshape(1, -1)
            
#             # Scale features and predict
#             scaled_features = self.scaler.transform(feature_array)
#             prediction_encoded = self.model.predict(scaled_features)[0]
            
#             return self.label_encoder.inverse_transform([prediction_encoded])[0]
#         except Exception as e:
#             print(f"Strategy prediction error: {e}")
#             # Fallback to rule-based strategy
#             return self._fallback_strategy(features)
    
#     def _fallback_strategy(self, features: dict) -> str:
#         """Fallback strategy when model prediction fails"""
#         degradation = features.get('tire_degradation_rate', 0.1)
#         position = features.get('position', 1)
        
#         if degradation > 0.15 or position > 8:
#             return 'early'
#         elif position == 1:
#             return 'late'
#         else:
#             return 'middle'
    
#     def save_model(self, filepath: str):
#         """Save trained model, scaler, and encoders"""
#         model_data = {
#             'model': self.model,
#             'scaler': self.scaler,
#             'label_encoder': self.label_encoder,
#             'feature_columns': self.feature_columns
#         }
#         joblib.dump(model_data, filepath)
    
#     def load_model(self, filepath: str):
#         """Load trained model, scaler, and encoders"""
#         model_data = joblib.load(filepath)
#         self.model = model_data['model']
#         self.scaler = model_data['scaler']
#         self.label_encoder = model_data['label_encoder']
#         self.feature_columns = model_data['feature_columns']



















# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# import joblib

# class PitStrategyTrainer:
#     def __init__(self):
#         self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
#     def train(self, processed_data: dict) -> dict:
#         """Train pit strategy model using race results and lap data"""
#         # Extract features from multiple tracks
#         features_list = []
#         targets_list = []
        
#         for track_name, data in processed_data.items():
#             if not data['race_data'].empty and not data['lap_data'].empty:
#                 track_features, track_targets = self._extract_track_features(data, track_name)
#                 features_list.append(track_features)
#                 targets_list.append(track_targets)
        
#         if not features_list:
#             return {'model': self, 'features': [], 'accuracy': 0}
        
#         # Combine all track data
#         X = pd.concat(features_list, ignore_index=True)
#         y = pd.concat(targets_list, ignore_index=True)
        
#         # Train model
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         self.model.fit(X_train, y_train)
        
#         # Evaluate
#         accuracy = self.model.score(X_test, y_test)
#         feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        
#         return {
#             'model': self,
#             'features': X.columns.tolist(),
#             'accuracy': accuracy,
#             'feature_importance': feature_importance
#         }
    
#     def _extract_track_features(self, data: dict, track_name: str) -> tuple:
#         """Extract features for pit strategy prediction"""
#         race_data = data['race_data']
#         lap_data = data['lap_data']
        
#         features = []
#         targets = []
        
#         for _, car in race_data.iterrows():
#             car_number = car['NUMBER']
#             car_laps = lap_data[lap_data['NUMBER'] == car_number]
            
#             if len(car_laps) < 5:  # Need enough lap data
#                 continue
            
#             # Feature 1: Optimal pit lap based on best lap timing
#             best_lap_num = car.get('BEST_LAP_NUM', car_laps['LAP_NUMBER'].iloc[car_laps['LAP_TIME_SECONDS'].argmin()])
            
#             # Feature 2: Tire degradation rate (seconds per lap)
#             if len(car_laps) > 10:
#                 degradation_rate = self._calculate_degradation_rate(car_laps)
#             else:
#                 degradation_rate = 0.1  # Default
            
#             # Feature 3: Fuel effect (time improvement from lap 1 to lap 10)
#             fuel_effect = self._calculate_fuel_effect(car_laps)
            
#             # Feature 4: Position and gaps
#             position = car.get('POSITION', 1)
#             gap_to_leader = self._parse_gap(car.get('GAP_FIRST', '0'))
            
#             # Feature 5: Track characteristics
#             track_wear_factor = self._get_track_wear_factor(track_name)
            
#             # Create feature vector
#             feature_vector = pd.DataFrame([{
#                 'best_lap_num': best_lap_num,
#                 'degradation_rate': degradation_rate,
#                 'fuel_effect': fuel_effect,
#                 'position': position,
#                 'gap_to_leader': gap_to_leader,
#                 'track_wear_factor': track_wear_factor,
#                 'total_laps': car.get('LAPS', len(car_laps))
#             }])
            
#             # Target: Optimal pit window (early, middle, late)
#             optimal_window = self._determine_optimal_pit_window(car, car_laps)
            
#             features.append(feature_vector)
#             targets.append(pd.Series([optimal_window]))
        
#         if features:
#             return pd.concat(features, ignore_index=True), pd.concat(targets, ignore_index=True)
#         return pd.DataFrame(), pd.Series(dtype=object)
    
#     def _calculate_degradation_rate(self, car_laps: pd.DataFrame) -> float:
#         """Calculate tire degradation rate in seconds per lap"""
#         if len(car_laps) < 5:
#             return 0.1
        
#         # Use laps 5-15 for degradation calculation (after warm-up, before pits)
#         mid_laps = car_laps[(car_laps['LAP_NUMBER'] >= 5) & (car_laps['LAP_NUMBER'] <= 15)]
#         if len(mid_laps) < 3:
#             return 0.1
        
#         times = mid_laps['LAP_TIME_SECONDS'].values
#         laps = mid_laps['LAP_NUMBER'].values
        
#         # Linear regression for degradation rate
#         try:
#             slope = np.polyfit(laps, times, 1)[0]
#             return max(0.01, min(1.0, slope))  # Bound between 0.01 and 1.0
#         except:
#             return 0.1
    
#     def _calculate_fuel_effect(self, car_laps: pd.DataFrame) -> float:
#         """Calculate fuel burn effect (seconds improvement from start to mid-race)"""
#         if len(car_laps) < 10:
#             return 0.5
        
#         early_laps = car_laps[car_laps['LAP_NUMBER'] <= 3]['LAP_TIME_SECONDS'].mean()
#         mid_laps = car_laps[(car_laps['LAP_NUMBER'] >= 8) & (car_laps['LAP_NUMBER'] <= 12)]['LAP_TIME_SECONDS'].mean()
        
#         if pd.isna(early_laps) or pd.isna(mid_laps):
#             return 0.5
        
#         return max(0.1, min(2.0, early_laps - mid_laps))
    
#     def _parse_gap(self, gap_str: str) -> float:
#         """Parse gap string to seconds"""
#         if pd.isna(gap_str) or gap_str in ['-', '']:
#             return 0
        
#         try:
#             # Handle formats like "+0.234", "+16.306"
#             gap_str = str(gap_str).replace('+', '').strip()
#             return float(gap_str)
#         except:
#             return 0
    
#     def _get_track_wear_factor(self, track_name: str) -> float:
#         """Get track-specific tire wear factor"""
#         wear_factors = {
#             'barber-motorsports-park': 0.8,
#             'circuit-of-the-americas': 0.7,
#             'indianapolis': 0.5,
#             'road-america': 0.6,
#             'sebring': 0.9,
#             'sonoma': 0.8,
#             'virginia-international-raceway': 0.7
#         }
#         return wear_factors.get(track_name, 0.7)
    
#     def _determine_optimal_pit_window(self, car: pd.Series, car_laps: pd.DataFrame) -> str:
#         """Determine optimal pit strategy based on race data"""
#         total_laps = car.get('LAPS', len(car_laps))
#         position = car.get('POSITION', 1)
        
#         # Simple heuristic based on position and race length
#         if position <= 3:  # Front runners - conservative
#             return 'middle'
#         elif position >= 15:  # Back markers - aggressive
#             return 'early'
#         else:  # Mid-field - balanced
#             return 'middle'
    
#     def predict_optimal_window(self, features: dict) -> str:
#         """Predict optimal pit window for current race state"""
#         feature_df = pd.DataFrame([features])
#         return self.model.predict(feature_df)[0]
    
#     def save_model(self, filepath: str):
#         """Save trained model"""
#         joblib.dump(self.model, filepath)