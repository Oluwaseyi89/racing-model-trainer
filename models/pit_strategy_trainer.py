import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from typing import Dict, List, Tuple

class PitStrategyTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        
        # Updated to match EXACT column names from FirebaseDataLoader schemas
        self.required_pit_cols = ['NUMBER', 'DRIVER_NUMBER', 'LAP_NUMBER', 'LAP_TIME', 'LAP_IMPROVEMENT', 'CROSSING_FINISH_LINE_IN_PIT', 'S1', 'S1_IMPROVEMENT', 'S2', 'S2_IMPROVEMENT', 'S3', 'S3_IMPROVEMENT', 'KPH', 'ELAPSED', 'HOUR', 'S1_LARGE', 'S2_LARGE', 'S3_LARGE', 'TOP_SPEED', 'PIT_TIME', 'CLASS', 'GROUP', 'MANUFACTURER', 'FLAG_AT_FL', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS', 'IM1a_time', 'IM1a_elapsed', 'IM1_time', 'IM1_elapsed', 'IM2a_time', 'IM2a_elapsed', 'IM2_time', 'IM2_elapsed', 'IM3a_time', 'IM3a_elapsed', 'FL_time', 'FL_elapsed']
        self.required_race_cols = ['POSITION', 'NUMBER', 'STATUS', 'LAPS', 'TOTAL_TIME', 'GAP_FIRST', 'GAP_PREVIOUS', 'FL_LAPNUM', 'FL_TIME', 'FL_KPH', 'CLASS', 'GROUP', 'DIVISION', 'VEHICLE', 'TIRES', 'ECM Participant Id', 'ECM Team Id', 'ECM Category Id', 'ECM Car Id', 'ECM Brand Id', 'ECM Country Id', '*Extra 7', '*Extra 8', '*Extra 9', 'Sort Key', 'DRIVER_*Extra 3', 'DRIVER_*Extra 4', 'DRIVER_*Extra 5']
        self.required_weather_cols = ['TIME_UTC_SECONDS', 'TIME_UTC_STR', 'AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 'PRESSURE', 'WIND_SPEED', 'WIND_DIRECTION', 'RAIN']

    def train(self, track_data: Dict[str, Dict[str, pd.DataFrame]]) -> dict:
        """
        Train pit strategy model using structured data from Firebase loader across all tracks
        """
        try:
            # Validate input data structure
            if not isinstance(track_data, dict) or len(track_data) < 2:
                return {'error': 'Insufficient tracks for pit strategy model', 'status': 'error'}

            features_list = []
            targets_list = []

            # Process each track's data
            for track_name, data_dict in track_data.items():
                if not self._validate_track_data(data_dict):
                    continue
                    
                session_features, session_targets = self._extract_track_features(data_dict, track_name)
                if not session_features.empty and not session_targets.empty:
                    features_list.append(session_features)
                    targets_list.append(session_targets)

            if not features_list:
                return {'error': 'No valid training data extracted from tracks', 'status': 'error'}

            # Combine all track data
            X = pd.concat(features_list, ignore_index=True)
            y = pd.concat(targets_list, ignore_index=True)

            if X.empty or y.empty:
                return {'error': 'Empty feature or target matrices', 'status': 'error'}

            # Encode strategy targets
            try:
                y_encoded = self.label_encoder.fit_transform(y.astype(str))
            except Exception as e:
                return {'error': f'Target encoding failed: {e}', 'status': 'error'}

            # Clean data (should be minimal due to schema enforcement)
            valid_mask = ~X.isna().any(axis=1)
            X = X[valid_mask]
            y_encoded = y_encoded[valid_mask]

            if len(X) < 20:
                return {'error': f'Insufficient training samples: {len(X)}', 'status': 'error'}

            # Scale features and train
            X_scaled = self.scaler.fit_transform(X)
            self.feature_columns = X.columns.tolist()

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
                'tracks_used': len(track_data),
                'class_distribution': dict(zip(self.label_encoder.classes_, np.bincount(y_encoded))),
                'status': 'success'
            }
            
        except Exception as e:
            return {'error': f'Training failed: {str(e)}', 'status': 'error'}

    def _validate_track_data(self, data_dict: Dict[str, pd.DataFrame]) -> bool:
        """Validate that track data has required components using EXACT column names"""
        pit_data = data_dict.get('pit_data', pd.DataFrame())
        race_data = data_dict.get('race_data', pd.DataFrame())
        
        if pit_data.empty or race_data.empty:
            return False
            
        # Check for required columns using EXACT names
        missing_pit = [col for col in ['NUMBER', 'LAP_NUMBER', 'LAP_TIME', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'] if col not in pit_data.columns]
        missing_race = [col for col in ['POSITION', 'NUMBER', 'LAPS', 'TOTAL_TIME'] if col not in race_data.columns]
        
        return len(missing_pit) == 0 and len(missing_race) == 0

    def _extract_track_features(self, data_dict: Dict[str, pd.DataFrame], track_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract pit strategy features for all cars in a track using EXACT column names
        """
        pit_data = data_dict['pit_data']
        race_data = data_dict['race_data']
        weather_data = data_dict.get('weather_data', pd.DataFrame())
        telemetry_data = data_dict.get('telemetry_data', pd.DataFrame())

        features = []
        targets = []

        # Process each car in the race using EXACT column names
        for _, car_result in race_data.iterrows():
            car_number = car_result['NUMBER']
            
            # Get car's lap data using EXACT column names
            car_laps = pit_data[pit_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
            if car_laps.empty:
                continue

            # Extract features using EXACT column names
            performance_features = self._calculate_car_performance(car_result, car_laps, telemetry_data)
            condition_features = self._extract_track_conditions(weather_data, track_name)
            competitive_features = self._extract_competitive_position(car_result, race_data)
            strategy_features = self._analyze_strategy_patterns(car_laps, car_result)

            feature_vector = {
                **performance_features,
                **condition_features, 
                **competitive_features,
                **strategy_features,
                'track_name_encoded': hash(track_name) % 100
            }

            # Determine optimal strategy
            optimal_strategy = self._determine_optimal_pit_strategy(feature_vector, car_laps)
            
            features.append(pd.DataFrame([feature_vector]))
            targets.append(pd.Series([optimal_strategy]))

        if features and targets:
            return pd.concat(features, ignore_index=True), pd.concat(targets, ignore_index=True)
        return pd.DataFrame(), pd.Series(dtype=object)

    def _calculate_car_performance(self, car_result: pd.Series, car_laps: pd.DataFrame, 
                                 telemetry_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for strategy decisions using EXACT column names"""
        metrics = {}
        
        # Convert lap times to seconds for analysis using EXACT column names
        lap_times_seconds = car_laps['LAP_TIME'].apply(self._lap_time_to_seconds)
        lap_numbers = car_laps['LAP_NUMBER'].values
        
        # Tire degradation (lap time increase over race)
        if len(lap_times_seconds) > 2:
            degradation_slope = np.polyfit(lap_numbers, lap_times_seconds, 1)[0]
            metrics['tire_degradation_rate'] = max(0.0, degradation_slope)
        else:
            metrics['tire_degradation_rate'] = 0.1
            
        # Performance consistency
        metrics['lap_time_consistency'] = 1.0 / (1.0 + lap_times_seconds.std()) if lap_times_seconds.std() > 0 else 1.0
        
        # Sector performance analysis using EXACT column names
        for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
            if sector in car_laps.columns:
                sector_times = pd.to_numeric(car_laps[sector], errors='coerce').fillna(0)
                if len(sector_times) > 2:
                    sector_slope = np.polyfit(lap_numbers, sector_times, 1)[0]
                    metrics[f'sector_{i}_degradation'] = max(0.0, sector_slope)
                else:
                    metrics[f'sector_{i}_degradation'] = 0.05
            else:
                metrics[f'sector_{i}_degradation'] = 0.05
        
        # Additional performance metrics using EXACT column names
        if 'TOP_SPEED' in car_laps.columns:
            metrics['avg_top_speed'] = car_laps['TOP_SPEED'].mean()
        else:
            metrics['avg_top_speed'] = 150.0
            
        if 'KPH' in car_laps.columns:
            metrics['avg_kph'] = car_laps['KPH'].mean()
        else:
            metrics['avg_kph'] = 120.0
            
        if 'LAP_IMPROVEMENT' in car_laps.columns:
            metrics['lap_improvement_ratio'] = (car_laps['LAP_IMPROVEMENT'] > 0).mean()
        else:
            metrics['lap_improvement_ratio'] = 0.5
        
        # Best lap timing (when car was fastest)
        if not lap_times_seconds.empty:
            best_lap_idx = lap_times_seconds.idxmin()
            metrics['best_lap_occurrence'] = lap_numbers[best_lap_idx] / len(lap_numbers) if len(lap_numbers) > 0 else 0.5
        else:
            metrics['best_lap_occurrence'] = 0.5
            
        return metrics

    def _extract_track_conditions(self, weather_data: pd.DataFrame, track_name: str) -> Dict[str, float]:
        """Extract weather and track condition features using EXACT column names"""
        if weather_data.empty:
            return {
                'avg_track_temp': 35.0,
                'avg_air_temp': 25.0, 
                'temp_variance': 2.0,
                'humidity_level': 50.0,
                'pressure_level': 1013.0,
                'track_abrasiveness': self._get_track_abrasiveness(track_name)
            }
        
        return {
            'avg_track_temp': weather_data['TRACK_TEMP'].mean() if 'TRACK_TEMP' in weather_data.columns else 35.0,
            'avg_air_temp': weather_data['AIR_TEMP'].mean() if 'AIR_TEMP' in weather_data.columns else 25.0,
            'temp_variance': weather_data['TRACK_TEMP'].var() if 'TRACK_TEMP' in weather_data.columns else 2.0,
            'humidity_level': weather_data['HUMIDITY'].mean() if 'HUMIDITY' in weather_data.columns else 50.0,
            'pressure_level': weather_data['PRESSURE'].mean() if 'PRESSURE' in weather_data.columns else 1013.0,
            'track_abrasiveness': self._get_track_abrasiveness(track_name)
        }

    def _get_track_abrasiveness(self, track_name: str) -> float:
        """Estimate track abrasiveness based on track characteristics"""
        high_abrasion_tracks = ['barber', 'sonoma', 'sebring']
        medium_abrasion_tracks = ['cota', 'road_america', 'virginia']
        
        track_lower = track_name.lower()
        if any(t in track_lower for t in high_abrasion_tracks):
            return 0.8
        elif any(t in track_lower for t in medium_abrasion_tracks):
            return 0.5
        else:
            return 0.3

    def _extract_competitive_position(self, car_result: pd.Series, race_data: pd.DataFrame) -> Dict[str, float]:
        """Extract competitive context features using EXACT column names"""
        position = car_result['POSITION']
        total_cars = len(race_data)
        
        # Parse time gaps using EXACT column names
        gap_to_leader = self._parse_time_gap(car_result.get('GAP_FIRST', '0'))
        gap_to_next = self._parse_time_gap(car_result.get('GAP_PREVIOUS', '0'))
        
        # Additional competitive metrics
        if 'FL_LAPNUM' in car_result:
            fastest_lap_timing = car_result['FL_LAPNUM'] / car_result['LAPS'] if car_result['LAPS'] > 0 else 0.5
        else:
            fastest_lap_timing = 0.5
            
        if 'FL_KPH' in car_result:
            fastest_lap_speed = car_result['FL_KPH']
        else:
            fastest_lap_speed = 150.0
        
        return {
            'position_normalized': position / total_cars,
            'total_competitors': total_cars,
            'gap_to_leader_seconds': gap_to_leader,
            'gap_to_next_seconds': gap_to_next,
            'fastest_lap_timing': fastest_lap_timing,
            'fastest_lap_speed': fastest_lap_speed,
            'is_race_leader': 1.0 if position == 1 else 0.0,
            'is_top_5': 1.0 if position <= 5 else 0.0,
            'is_midfield': 1.0 if 6 <= position <= 15 else 0.0
        }

    def _analyze_strategy_patterns(self, car_laps: pd.DataFrame, car_result: pd.Series) -> Dict[str, float]:
        """Analyze historical strategy patterns using EXACT column names"""
        total_laps = car_result['LAPS']
        if total_laps == 0:
            return {'race_duration_factor': 0.5, 'stint_length_ratio': 0.3}
            
        # Analyze stint patterns using pit time data
        if 'PIT_TIME' in car_laps.columns:
            pit_stops = car_laps[car_laps['PIT_TIME'] > 0]
            stint_count = len(pit_stops) + 1
            avg_stint_length = total_laps / stint_count if stint_count > 0 else total_laps
        else:
            avg_stint_length = total_laps / 2.0
            
        stint_ratio = avg_stint_length / total_laps
        
        # Flag analysis
        if 'FLAG_AT_FL' in car_laps.columns:
            caution_flags = car_laps[car_laps['FLAG_AT_FL'].str.contains('FCY|SC', na=False)]
            caution_ratio = len(caution_flags) / len(car_laps)
        else:
            caution_ratio = 0.1
        
        return {
            'race_duration_factor': min(1.0, total_laps / 50.0),
            'stint_length_ratio': stint_ratio,
            'caution_flag_ratio': caution_ratio
        }

    def _determine_optimal_pit_strategy(self, features: Dict[str, float], car_laps: pd.DataFrame) -> str:
        """
        Determine optimal pit strategy based on features using enhanced logic
        Strategies: 'early' (aggressive), 'middle' (standard), 'late' (conservative), 'undercut' (early overtake), 'overcut' (late overtake)
        """
        degradation = features.get('tire_degradation_rate', 0.1)
        position = features.get('position_normalized', 0.5)
        gap_to_leader = features.get('gap_to_leader_seconds', 0)
        gap_to_next = features.get('gap_to_next_seconds', 0)
        track_abrasiveness = features.get('track_abrasiveness', 0.5)
        caution_ratio = features.get('caution_flag_ratio', 0.1)
        
        # Strategy scoring
        score = 0
        
        # High degradation favors early stops
        if degradation > 0.15:
            score += 2
        elif degradation > 0.08:
            score += 1
            
        # Abrasive tracks favor early stops
        if track_abrasiveness > 0.7:
            score += 1
            
        # High caution flag probability favors flexible strategy
        if caution_ratio > 0.2:
            return 'middle'  # Standard strategy for unpredictable conditions
            
        # Leading position favors conservative strategy
        if position <= 0.1:  # Top 10%
            score -= 2
        # Midfield favors aggressive strategy
        elif position >= 0.7:  # Bottom 30%
            score += 2
            
        # Close competition favors undercut/overcut strategies
        if 0 < gap_to_next < 5:  # Close battle with next car
            if position <= 0.3:  # In top positions
                return 'undercut' if degradation > 0.1 else 'overcut'
        
        # Large gap to leader favors aggressive strategy
        if gap_to_leader > 30:
            score += 1
            
        # Determine strategy based on score
        if score >= 3:
            return 'early'
        elif score <= -2:
            return 'late'
        else:
            return 'middle'

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

    def _parse_time_gap(self, gap_str: str) -> float:
        """Parse time gap string to seconds"""
        try:
            gap_clean = str(gap_str).replace('+', '').replace('-', '').strip()
            if ':' in gap_clean:
                parts = gap_clean.split(':')
                if len(parts) == 2:
                    return float(parts[0]) * 60 + float(parts[1])
            return float(gap_clean)
        except:
            return 0.0

    def predict_pit_strategy(self, features: Dict[str, float]) -> str:
        """Predict optimal pit strategy for given features"""
        try:
            if not self.feature_columns:
                return 'middle'  # Default strategy
                
            # Ensure all features are present
            feature_vector = [features.get(col, 0.0) for col in self.feature_columns]
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            prediction_encoded = self.model.predict(X_scaled)[0]
            return self.label_encoder.inverse_transform([prediction_encoded])[0]
            
        except Exception:
            return 'middle'  # Fallback strategy

    def get_strategy_confidence(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get confidence scores for each strategy option"""
        try:
            if not self.feature_columns:
                return {'early': 0.2, 'middle': 0.4, 'late': 0.2, 'undercut': 0.1, 'overcut': 0.1}
                
            feature_vector = [features.get(col, 0.0) for col in self.feature_columns]
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence_dict = {}
            
            for i, strategy in enumerate(self.label_encoder.classes_):
                confidence_dict[strategy] = float(probabilities[i])
                
            return confidence_dict
            
        except Exception:
            return {'early': 0.2, 'middle': 0.4, 'late': 0.2, 'undercut': 0.1, 'overcut': 0.1}

    def save_model(self, filepath: str):
        """Save trained model to file"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }, filepath)

    def load_model(self, filepath: str):
        """Load trained model from file"""
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
# from typing import Dict, List, Tuple

# class PitStrategyTrainer:
#     def __init__(self):
#         self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
#         self.scaler = StandardScaler()
#         self.label_encoder = LabelEncoder()
#         self.feature_columns = []
        
#         # Define expected data structures based on your schema
#         self.required_pit_cols = ['NUMBER', 'LAP_NUMBER', 'LAP_TIME', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']
#         self.required_race_cols = ['POSITION', 'NUMBER', 'LAPS', 'TOTAL_TIME', 'GAP_FIRST', 'GAP_PREVIOUS']
#         self.required_weather_cols = ['TIME_UTC_SECONDS', 'TRACK_TEMP', 'AIR_TEMP']

#     def train(self, processed_data: Dict[str, Dict[str, pd.DataFrame]]) -> dict:
#         """
#         Train pit strategy model using structured data from Firebase loader
#         """
#         try:
#             # Validate input data structure
#             if not isinstance(processed_data, dict) or len(processed_data) < 2:
#                 return {'error': 'Insufficient tracks for pit strategy model', 'status': 'error'}

#             features_list = []
#             targets_list = []

#             # Process each track's data
#             for track_name, track_data in processed_data.items():
#                 if not self._validate_track_data(track_data):
#                     continue
                    
#                 session_features, session_targets = self._extract_track_features(track_data, track_name)
#                 if not session_features.empty and not session_targets.empty:
#                     features_list.append(session_features)
#                     targets_list.append(session_targets)

#             if not features_list:
#                 return {'error': 'No valid training data extracted from tracks', 'status': 'error'}

#             # Combine all track data
#             X = pd.concat(features_list, ignore_index=True)
#             y = pd.concat(targets_list, ignore_index=True)

#             if X.empty or y.empty:
#                 return {'error': 'Empty feature or target matrices', 'status': 'error'}

#             # Encode strategy targets
#             try:
#                 y_encoded = self.label_encoder.fit_transform(y.astype(str))
#             except Exception as e:
#                 return {'error': f'Target encoding failed: {e}', 'status': 'error'}

#             # Clean data
#             valid_mask = ~X.isna().any(axis=1)
#             X = X[valid_mask]
#             y_encoded = y_encoded[valid_mask]

#             if len(X) < 20:
#                 return {'error': f'Insufficient training samples: {len(X)}', 'status': 'error'}

#             # Scale features and train
#             X_scaled = self.scaler.fit_transform(X)
#             self.feature_columns = X.columns.tolist()

#             X_train, X_test, y_train, y_test = train_test_split(
#                 X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
#             )
            
#             self.model.fit(X_train, y_train)
#             accuracy = self.model.score(X_test, y_test)

#             feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))

#             return {
#                 'model': self,
#                 'features': self.feature_columns,
#                 'accuracy': accuracy,
#                 'feature_importance': feature_importance,
#                 'training_samples': len(X),
#                 'class_distribution': dict(zip(self.label_encoder.classes_, np.bincount(y_encoded))),
#                 'status': 'success'
#             }
            
#         except Exception as e:
#             return {'error': f'Training failed: {str(e)}', 'status': 'error'}

#     def _validate_track_data(self, track_data: Dict[str, pd.DataFrame]) -> bool:
#         """Validate that track data has required components"""
#         pit_data = track_data.get('pit_data', pd.DataFrame())
#         race_data = track_data.get('race_data', pd.DataFrame())
        
#         if pit_data.empty or race_data.empty:
#             return False
            
#         # Check for required columns
#         missing_pit = [col for col in self.required_pit_cols if col not in pit_data.columns]
#         missing_race = [col for col in self.required_race_cols if col not in race_data.columns]
        
#         return len(missing_pit) == 0 and len(missing_race) == 0

#     def _extract_track_features(self, track_data: Dict[str, pd.DataFrame], track_name: str) -> Tuple[pd.DataFrame, pd.Series]:
#         """
#         Extract pit strategy features for all cars in a track
#         """
#         pit_data = track_data['pit_data']
#         race_data = track_data['race_data']
#         weather_data = track_data.get('weather_data', pd.DataFrame())
#         telemetry_data = track_data.get('telemetry_data', pd.DataFrame())

#         features = []
#         targets = []

#         # Process each car in the race
#         for _, car_result in race_data.iterrows():
#             car_number = car_result['NUMBER']
            
#             # Get car's lap data
#             car_laps = pit_data[pit_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
#             if car_laps.empty:
#                 continue

#             # Extract features
#             performance_features = self._calculate_car_performance(car_result, car_laps, telemetry_data)
#             condition_features = self._extract_track_conditions(weather_data, track_name)
#             competitive_features = self._extract_competitive_position(car_result, race_data)
#             strategy_features = self._analyze_strategy_patterns(car_laps, car_result)

#             feature_vector = {
#                 **performance_features,
#                 **condition_features, 
#                 **competitive_features,
#                 **strategy_features,
#                 'track_name_encoded': hash(track_name) % 100  # Simple track encoding
#             }

#             # Determine optimal strategy
#             optimal_strategy = self._determine_optimal_pit_strategy(feature_vector, car_laps)
            
#             features.append(pd.DataFrame([feature_vector]))
#             targets.append(pd.Series([optimal_strategy]))

#         if features and targets:
#             return pd.concat(features, ignore_index=True), pd.concat(targets, ignore_index=True)
#         return pd.DataFrame(), pd.Series(dtype=object)

#     def _calculate_car_performance(self, car_result: pd.Series, car_laps: pd.DataFrame, 
#                                  telemetry_data: pd.DataFrame) -> Dict[str, float]:
#         """Calculate performance metrics for strategy decisions"""
#         metrics = {}
        
#         # Convert lap times to seconds for analysis
#         lap_times_seconds = car_laps['LAP_TIME'].apply(self._lap_time_to_seconds)
#         lap_numbers = car_laps['LAP_NUMBER'].values
        
#         # Tire degradation (lap time increase over race)
#         if len(lap_times_seconds) > 2:
#             degradation_slope = np.polyfit(lap_numbers, lap_times_seconds, 1)[0]
#             metrics['tire_degradation_rate'] = max(0.0, degradation_slope)
#         else:
#             metrics['tire_degradation_rate'] = 0.1
            
#         # Performance consistency
#         metrics['lap_time_consistency'] = 1.0 / (1.0 + lap_times_seconds.std()) if lap_times_seconds.std() > 0 else 1.0
        
#         # Sector performance analysis
#         for i, sector in enumerate(['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'], 1):
#             if sector in car_laps.columns:
#                 sector_times = pd.to_numeric(car_laps[sector], errors='coerce').fillna(0)
#                 if len(sector_times) > 2:
#                     sector_slope = np.polyfit(lap_numbers, sector_times, 1)[0]
#                     metrics[f'sector_{i}_degradation'] = max(0.0, sector_slope)
#                 else:
#                     metrics[f'sector_{i}_degradation'] = 0.05
#             else:
#                 metrics[f'sector_{i}_degradation'] = 0.05
        
#         # Best lap timing (when car was fastest)
#         if not lap_times_seconds.empty:
#             best_lap_idx = lap_times_seconds.idxmin()
#             metrics['best_lap_occurrence'] = lap_numbers[best_lap_idx] / len(lap_numbers) if len(lap_numbers) > 0 else 0.5
#         else:
#             metrics['best_lap_occurrence'] = 0.5
            
#         return metrics

#     def _extract_track_conditions(self, weather_data: pd.DataFrame, track_name: str) -> Dict[str, float]:
#         """Extract weather and track condition features"""
#         if weather_data.empty:
#             return {
#                 'avg_track_temp': 35.0,
#                 'avg_air_temp': 25.0, 
#                 'temp_variance': 2.0,
#                 'track_abrasiveness': self._get_track_abrasiveness(track_name)
#             }
        
#         return {
#             'avg_track_temp': weather_data['TRACK_TEMP'].mean() if 'TRACK_TEMP' in weather_data.columns else 35.0,
#             'avg_air_temp': weather_data['AIR_TEMP'].mean() if 'AIR_TEMP' in weather_data.columns else 25.0,
#             'temp_variance': weather_data['TRACK_TEMP'].var() if 'TRACK_TEMP' in weather_data.columns else 2.0,
#             'track_abrasiveness': self._get_track_abrasiveness(track_name)
#         }

#     def _get_track_abrasiveness(self, track_name: str) -> float:
#         """Estimate track abrasiveness based on track characteristics"""
#         high_abrasion_tracks = ['monza', 'spa', 'silverstone', 'suzuka']
#         medium_abrasion_tracks = ['cota', 'interlagos', 'redbullring']
        
#         track_lower = track_name.lower()
#         if any(t in track_lower for t in high_abrasion_tracks):
#             return 0.8
#         elif any(t in track_lower for t in medium_abrasion_tracks):
#             return 0.5
#         else:
#             return 0.3

#     def _extract_competitive_position(self, car_result: pd.Series, race_data: pd.DataFrame) -> Dict[str, float]:
#         """Extract competitive context features"""
#         position = car_result['POSITION']
#         total_cars = len(race_data)
        
#         # Parse time gaps
#         gap_to_leader = self._parse_time_gap(car_result.get('GAP_FIRST', '0'))
#         gap_to_next = self._parse_time_gap(car_result.get('GAP_PREVIOUS', '0'))
        
#         return {
#             'position_normalized': position / total_cars,
#             'total_competitors': total_cars,
#             'gap_to_leader_seconds': gap_to_leader,
#             'gap_to_next_seconds': gap_to_next,
#             'is_race_leader': 1.0 if position == 1 else 0.0,
#             'is_top_5': 1.0 if position <= 5 else 0.0,
#             'is_midfield': 1.0 if 6 <= position <= 15 else 0.0
#         }

#     def _analyze_strategy_patterns(self, car_laps: pd.DataFrame, car_result: pd.Series) -> Dict[str, float]:
#         """Analyze historical strategy patterns"""
#         total_laps = car_result['LAPS']
#         if total_laps == 0:
#             return {'race_duration_factor': 0.5, 'stint_length_ratio': 0.3}
            
#         # Analyze stint patterns (simplified)
#         avg_stint_length = total_laps / 2.0  # Assume 2 pit stops
#         stint_ratio = avg_stint_length / total_laps
        
#         return {
#             'race_duration_factor': min(1.0, total_laps / 50.0),  # Normalize by typical race length
#             'stint_length_ratio': stint_ratio
#         }

#     def _determine_optimal_pit_strategy(self, features: Dict[str, float], car_laps: pd.DataFrame) -> str:
#         """
#         Determine optimal pit strategy based on features
#         Strategies: 'early' (aggressive), 'middle' (standard), 'late' (conservative)
#         """
#         degradation = features.get('tire_degradation_rate', 0.1)
#         position = features.get('position_normalized', 0.5)
#         gap_to_leader = features.get('gap_to_leader_seconds', 0)
#         track_abrasiveness = features.get('track_abrasiveness', 0.5)
        
#         # Strategy scoring
#         score = 0
        
#         # High degradation favors early stops
#         if degradation > 0.15:
#             score += 2
#         elif degradation > 0.08:
#             score += 1
            
#         # Abrasive tracks favor early stops
#         if track_abrasiveness > 0.7:
#             score += 1
            
#         # Leading position favors conservative strategy
#         if position <= 0.1:  # Top 10%
#             score -= 1
#         # Midfield favors aggressive strategy
#         elif position >= 0.7:  # Bottom 30%
#             score += 1
            
#         # Large gap to leader favors aggressive strategy
#         if gap_to_leader > 30:
#             score += 1
            
#         # Determine strategy based on score
#         if score >= 3:
#             return 'early'
#         elif score <= 0:
#             return 'late'
#         else:
#             return 'middle'

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

#     def _parse_time_gap(self, gap_str: str) -> float:
#         """Parse time gap string to seconds"""
#         try:
#             gap_clean = str(gap_str).replace('+', '').replace('-', '').strip()
#             if ':' in gap_clean:
#                 parts = gap_clean.split(':')
#                 if len(parts) == 2:
#                     return float(parts[0]) * 60 + float(parts[1])
#             return float(gap_clean)
#         except:
#             return 0.0

#     def predict_pit_strategy(self, features: Dict[str, float]) -> str:
#         """Predict optimal pit strategy for given features"""
#         try:
#             if not self.feature_columns:
#                 return 'middle'  # Default strategy
                
#             # Ensure all features are present
#             feature_vector = [features.get(col, 0.0) for col in self.feature_columns]
#             X = np.array(feature_vector).reshape(1, -1)
#             X_scaled = self.scaler.transform(X)
            
#             prediction_encoded = self.model.predict(X_scaled)[0]
#             return self.label_encoder.inverse_transform([prediction_encoded])[0]
            
#         except Exception:
#             return 'middle'  # Fallback strategy

#     def get_strategy_confidence(self, features: Dict[str, float]) -> Dict[str, float]:
#         """Get confidence scores for each strategy option"""
#         try:
#             if not self.feature_columns:
#                 return {'early': 0.33, 'middle': 0.34, 'late': 0.33}
                
#             feature_vector = [features.get(col, 0.0) for col in self.feature_columns]
#             X = np.array(feature_vector).reshape(1, -1)
#             X_scaled = self.scaler.transform(X)
            
#             probabilities = self.model.predict_proba(X_scaled)[0]
#             confidence_dict = {}
            
#             for i, strategy in enumerate(self.label_encoder.classes_):
#                 confidence_dict[strategy] = float(probabilities[i])
                
#             return confidence_dict
            
#         except Exception:
#             return {'early': 0.33, 'middle': 0.34, 'late': 0.33}

#     def save_model(self, filepath: str):
#         """Save trained model to file"""
#         joblib.dump({
#             'model': self.model,
#             'scaler': self.scaler,
#             'label_encoder': self.label_encoder,
#             'feature_columns': self.feature_columns
#         }, filepath)

#     def load_model(self, filepath: str):
#         """Load trained model from file"""
#         data = joblib.load(filepath)
#         self.model = data['model']
#         self.scaler = data['scaler']
#         self.label_encoder = data['label_encoder']
#         self.feature_columns = data['feature_columns']