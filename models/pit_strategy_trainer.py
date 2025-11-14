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

    # ------------------------------------------------------------
    # TRAINING ENTRY POINT
    # ------------------------------------------------------------
    def train(self, processed_data: dict) -> dict:
        """Train pit strategy model using comprehensive race data"""
        features_list = []
        targets_list = []

        for session_key, data in processed_data.items():
            lap_data = self._normalize_lap_data(data.get('lap_data', pd.DataFrame()))
            race_data = self._normalize_race_data(data.get('race_data', pd.DataFrame()))

            if not race_data.empty and not lap_data.empty:
                session_features, session_targets = self._extract_session_features(
                    {**data, 'lap_data': lap_data, 'race_data': race_data}, session_key
                )
                if not session_features.empty:
                    features_list.append(session_features)
                    targets_list.append(session_targets)

        if not features_list:
            return {'error': 'No valid training data extracted'}

        # Combine all session data
        X = pd.concat(features_list, ignore_index=True)
        y = pd.concat(targets_list, ignore_index=True)

        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)

        # Drop rows with NaNs
        valid_mask = ~X.isna().any(axis=1)
        X = X[valid_mask]
        y_encoded = y_encoded[valid_mask]

        if len(X) < 20:
            return {'error': f'Insufficient training samples: {len(X)}'}

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.feature_columns = X.columns.tolist()

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        self.model.fit(X_train, y_train)

        accuracy = self.model.score(X_test, y_test)
        feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))

        return {
            'model': self,
            'features': self.feature_columns,
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'training_samples': len(X),
            'class_distribution': dict(zip(self.label_encoder.classes_, np.bincount(y_encoded)))
        }

    # ------------------------------------------------------------
    # SESSION FEATURE EXTRACTION
    # ------------------------------------------------------------
    def _extract_session_features(self, data: dict, session_key: str) -> tuple:
        """Extract realistic pit strategy features from session data"""
        race_data = self._normalize_race_data(data.get('race_data', pd.DataFrame()))
        lap_data = self._normalize_lap_data(data.get('lap_data', pd.DataFrame()))
        telemetry_data = data.get('telemetry_data', pd.DataFrame())
        weather_data = data.get('weather_data', pd.DataFrame())

        features = []
        targets = []

        for _, car_result in race_data.iterrows():
            car_number = car_result.get('NUMBER')
            if pd.isna(car_number):
                continue

            car_laps = lap_data[lap_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
            if len(car_laps) < 3:
                continue

            actual_pit_lap = self._detect_actual_pit_stop(car_laps, telemetry_data)
            performance_features = self._calculate_performance_metrics(car_result, car_laps, telemetry_data)
            condition_features = self._extract_condition_features(weather_data, session_key)
            context_features = self._extract_competitive_context(car_result, race_data)

            feature_vector = {**performance_features, **condition_features, **context_features}
            optimal_strategy = self._determine_optimal_strategy(car_result, car_laps, actual_pit_lap, feature_vector)

            features.append(pd.DataFrame([feature_vector]))
            targets.append(pd.Series([optimal_strategy]))

        if features:
            return pd.concat(features, ignore_index=True), pd.concat(targets, ignore_index=True)
        return pd.DataFrame(), pd.Series(dtype=object)

    # ------------------------------------------------------------
    # LAP DATA NORMALIZATION
    # ------------------------------------------------------------
    def _normalize_lap_data(self, lap_data: pd.DataFrame) -> pd.DataFrame:
        """Ensure lap_data has NUMBER and LAP_NUMBER columns"""
        if lap_data.empty:
            return lap_data.copy()

        df = lap_data.copy()

        if 'NUMBER' not in df.columns:
            df['NUMBER'] = df.index  # fallback to row index

        if 'LAP_NUMBER' not in df.columns:
            df['LAP_NUMBER'] = df.groupby('NUMBER').cumcount() + 1

        # Ensure numeric types to avoid MKL DGELSD errors
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    # ------------------------------------------------------------
    # RACE DATA NORMALIZATION
    # ------------------------------------------------------------
    def _normalize_race_data(self, race_data: pd.DataFrame) -> pd.DataFrame:
        """Ensure race_data has NUMBER column"""
        if race_data.empty:
            return race_data.copy()
        df = race_data.copy()
        if 'NUMBER' not in df.columns:
            df['NUMBER'] = df.index
        return df

    # ------------------------------------------------------------
    # PIT DETECTION & PERFORMANCE METRICS
    # ------------------------------------------------------------
    def _detect_actual_pit_stop(self, car_laps: pd.DataFrame, telemetry_data: pd.DataFrame) -> int:
        if len(car_laps) < 3:
            return -1
        lap_times = car_laps['LAP_TIME_SECONDS'].values
        lap_numbers = car_laps['LAP_NUMBER'].values
        median_lap_time = np.median(lap_times)
        pit_threshold = median_lap_time * 1.8
        potential_pit_laps = lap_numbers[lap_times > pit_threshold]
        return int(potential_pit_laps[0]) if len(potential_pit_laps) > 0 else -1

    def _calculate_performance_metrics(self, car_result: pd.Series, car_laps: pd.DataFrame,
                                       telemetry_data: pd.DataFrame) -> dict:
        metrics = {}
        # Tire degradation
        if len(car_laps) > 15:
            stable_laps = car_laps[car_laps['LAP_NUMBER'].between(5, 15)]
            if len(stable_laps) > 5:
                try:
                    slope = np.polyfit(stable_laps['LAP_NUMBER'], stable_laps['LAP_TIME_SECONDS'], 1)[0]
                    metrics['tire_degradation_rate'] = max(0.01, min(0.5, slope))
                except:
                    metrics['tire_degradation_rate'] = 0.1
            else:
                metrics['tire_degradation_rate'] = 0.1
        else:
            metrics['tire_degradation_rate'] = 0.1

        # Fuel effect
        if len(car_laps) > 10:
            early = car_laps['LAP_TIME_SECONDS'].between(2, 4).mean()
            mid = car_laps['LAP_TIME_SECONDS'].between(8, 12).mean()
            metrics['fuel_effect'] = 0.8 if pd.isna(early) or pd.isna(mid) else max(0.1, min(2.0, early - mid))
        else:
            metrics['fuel_effect'] = 0.8

        metrics['consistency'] = 1.0 / (1.0 + car_laps['LAP_TIME_SECONDS'].std())
        metrics['best_lap_number'] = car_laps['LAP_NUMBER'].iloc[car_laps['LAP_TIME_SECONDS'].idxmin()]

        # Sector degradation
        for sector in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']:
            metrics[f'{sector.lower()}_degradation'] = self._calculate_sector_degradation(car_laps, sector)

        return metrics

    def _calculate_sector_degradation(self, car_laps: pd.DataFrame, sector_col: str) -> float:
        if sector_col not in car_laps.columns or len(car_laps) < 8:
            return 0.1
        valid = car_laps[car_laps[sector_col] > 0]
        mid_laps = valid[valid['LAP_NUMBER'].between(5, 15)]
        if len(mid_laps) < 3:
            return 0.1
        try:
            slope = np.polyfit(mid_laps['LAP_NUMBER'], mid_laps[sector_col], 1)[0]
            return max(0.001, min(0.2, slope))
        except:
            return 0.1

    # ------------------------------------------------------------
    # CONDITIONS & COMPETITIVE CONTEXT
    # ------------------------------------------------------------
    def _extract_condition_features(self, weather_data: pd.DataFrame, session_key: str) -> dict:
        if weather_data.empty:
            return {'avg_track_temp': 35.0, 'avg_air_temp': 25.0, 'track_temp_variance': 2.0,
                    'track_abrasiveness': 0.7, 'track_length_km': 5.0}

        return {
            'avg_track_temp': weather_data.get('TRACK_TEMP', pd.Series([35.0])).mean(),
            'avg_air_temp': weather_data.get('AIR_TEMP', pd.Series([25.0])).mean(),
            'track_temp_variance': weather_data.get('TRACK_TEMP', pd.Series([2.0])).var(),
            'track_abrasiveness': self._get_track_characteristic(session_key, 'abrasiveness'),
            'track_length_km': self._get_track_characteristic(session_key, 'length')
        }

    def _extract_competitive_context(self, car_result: pd.Series, race_data: pd.DataFrame) -> dict:
        pos = car_result.get('POSITION', 1)
        total_cars = len(race_data)
        return {
            'position': pos,
            'position_normalized': pos / total_cars if total_cars else 1,
            'total_cars': total_cars,
            'gap_to_leader': self._parse_gap(car_result.get('GAP_FIRST', '0')),
            'gap_to_next': self._parse_gap(car_result.get('GAP_PREVIOUS', '0')),
            'is_leading': 1 if pos == 1 else 0,
            'is_top_5': 1 if pos <= 5 else 0
        }

    def _get_track_characteristic(self, track_name: str, characteristic: str) -> float:
        track_map = {
            'sonoma': {'abrasiveness': 0.8, 'length': 4.0},
            'cota': {'abrasiveness': 0.7, 'length': 5.5},
            'road-america': {'abrasiveness': 0.6, 'length': 6.5},
            'barber': {'abrasiveness': 0.9, 'length': 3.7},
            'vir': {'abrasiveness': 0.7, 'length': 5.3},
            'sebring': {'abrasiveness': 0.9, 'length': 6.0}
        }
        t = track_map.get(track_name.lower(), {'abrasiveness': 0.7, 'length': 5.0})
        return t.get(characteristic, 0.7)

    def _parse_gap(self, gap_str: str) -> float:
        if pd.isna(gap_str) or gap_str in ['-', '']:
            return 0.0
        try:
            return float(str(gap_str).replace('+', '').strip())
        except:
            return 0.0

    # ------------------------------------------------------------
    # STRATEGY DECISION
    # ------------------------------------------------------------
    def _determine_optimal_strategy(self, car_result: pd.Series, car_laps: pd.DataFrame,
                                    actual_pit_lap: int, features: dict) -> str:
        score = 0
        degradation = features.get('tire_degradation_rate', 0.1)
        position = car_result.get('POSITION', 1)
        gap_to_leader = features.get('gap_to_leader', 0)

        if degradation > 0.2:
            score += 2
        elif degradation > 0.1:
            score += 1

        if position == 1:
            score -= 1
        elif position >= 10:
            score += 1

        if gap_to_leader > 30:
            score += 1
        elif gap_to_leader < 5:
            score -= 1

        return 'early' if score >= 2 else 'late' if score <= -1 else 'middle'

    # ------------------------------------------------------------
    # PREDICTION / FALLBACK
    # ------------------------------------------------------------
    def predict_optimal_strategy(self, features: dict) -> str:
        try:
            vec = np.array([features.get(c, 0) for c in self.feature_columns]).reshape(1, -1)
            return self.label_encoder.inverse_transform([self.model.predict(self.scaler.transform(vec))[0]])[0]
        except Exception:
            return self._fallback_strategy(features)

    def _fallback_strategy(self, features: dict) -> str:
        deg = features.get('tire_degradation_rate', 0.1)
        pos = features.get('position', 1)
        if deg > 0.15 or pos > 8:
            return 'early'
        elif pos == 1:
            return 'late'
        return 'middle'

    # ------------------------------------------------------------
    # MODEL SERIALIZATION
    # ------------------------------------------------------------
    def save_model(self, filepath: str):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }, filepath)

    def load_model(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']
        self.feature_columns = data['feature_columns']























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