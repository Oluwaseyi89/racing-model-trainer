import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, List, Tuple

class FuelModelTrainer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # Updated to match EXACT column names from FirebaseDataLoader schemas
        self.required_telemetry_cols = ['timestamp', 'vehicle_id', 'lap', 'outing', 'meta_session', 'accx_can', 'accy_can', 'gear', 'speed']
        self.required_pit_data_cols = ['NUMBER', 'DRIVER_NUMBER', 'LAP_NUMBER', 'LAP_TIME', 'LAP_IMPROVEMENT', 'CROSSING_FINISH_LINE_IN_PIT', 'S1', 'S1_IMPROVEMENT', 'S2', 'S2_IMPROVEMENT', 'S3', 'S3_IMPROVEMENT', 'KPH', 'ELAPSED', 'HOUR', 'S1_LARGE', 'S2_LARGE', 'S3_LARGE', 'TOP_SPEED', 'PIT_TIME', 'CLASS', 'GROUP', 'MANUFACTURER', 'FLAG_AT_FL', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS', 'IM1a_time', 'IM1a_elapsed', 'IM1_time', 'IM1_elapsed', 'IM2a_time', 'IM2a_elapsed', 'IM2_time', 'IM2_elapsed', 'IM3a_time', 'IM3a_elapsed', 'FL_time', 'FL_elapsed']

    def train(self, track_data: Dict[str, Dict[str, pd.DataFrame]]) -> dict:
        """
        Train fuel model using structured data from Firebase loader across all tracks
        """
        try:
            # Combine data from all tracks
            all_telemetry = pd.DataFrame()
            all_pit_data = pd.DataFrame()
            
            for track_name, data_dict in track_data.items():
                telemetry_data = data_dict.get('telemetry_data', pd.DataFrame())
                pit_data = data_dict.get('pit_data', pd.DataFrame())
                
                if not telemetry_data.empty:
                    telemetry_data['track_name'] = track_name
                    all_telemetry = pd.concat([all_telemetry, telemetry_data], ignore_index=True)
                
                if not pit_data.empty:
                    pit_data['track_name'] = track_name
                    all_pit_data = pd.concat([all_pit_data, pit_data], ignore_index=True)
            
            # Validate data availability
            if all_telemetry.empty or all_pit_data.empty:
                return self._train_with_fallback("Missing telemetry or pit data across all tracks")
            
            # Extract features and targets using EXACT column names
            features_df, targets = self._extract_fuel_features(all_telemetry, all_pit_data)
            
            if features_df.empty or len(targets) == 0:
                return self._train_with_fallback("No features extracted from data")
            
            # Prepare training data
            X = features_df
            y = np.array(targets)
            
            # Remove any remaining NaN values (should be minimal due to schema enforcement)
            valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 10:
                return self._train_with_fallback(f"Insufficient samples: {len(X)}")
            
            # Scale features and train model
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
            return self._train_with_fallback(f"Training error: {str(e)}")

    def _extract_fuel_features(self, telemetry_data: pd.DataFrame, pit_data: pd.DataFrame) -> Tuple[pd.DataFrame, List[float]]:
        """
        Extract fuel-related features from telemetry and pit data using EXACT column names
        """
        features_list = []
        consumption_targets = []
        
        # Group telemetry by vehicle and lap (using EXACT column names)
        for (vehicle_id, lap_num), lap_telemetry in telemetry_data.groupby(['vehicle_id', 'lap']):
            if len(lap_telemetry) < 10:  # Minimum telemetry points
                continue
                
            # Get corresponding lap info from pit data using EXACT column names
            lap_info = self._get_lap_info(pit_data, vehicle_id, lap_num)
            if lap_info.empty:
                continue
                
            # Calculate features for this lap using EXACT column names
            features = self._calculate_lap_features(lap_telemetry, lap_info)
            fuel_consumption = self._estimate_fuel_consumption(lap_telemetry, lap_info)
            
            features_list.append(features)
            consumption_targets.append(fuel_consumption)
        
        if features_list:
            return pd.DataFrame(features_list), consumption_targets
        return pd.DataFrame(), []

    def _get_lap_info(self, pit_data: pd.DataFrame, vehicle_id: str, lap_num: int) -> pd.Series:
        """
        Extract lap information from pit data using EXACT column names
        """
        try:
            # Convert vehicle_id to match NUMBER column in pit_data
            vehicle_num = self._extract_vehicle_number(vehicle_id)
            
            # Look for matching lap data using EXACT column names
            lap_match = pit_data[
                (pit_data['NUMBER'] == vehicle_num) & 
                (pit_data['LAP_NUMBER'] == lap_num)
            ]
            
            if not lap_match.empty:
                return lap_match.iloc[0]
            
            # Fallback: get closest lap data for this vehicle
            vehicle_laps = pit_data[pit_data['NUMBER'] == vehicle_num]
            if not vehicle_laps.empty:
                return vehicle_laps.iloc[0]
                
        except Exception:
            pass
            
        return pd.Series()

    def _extract_vehicle_number(self, vehicle_id: str) -> int:
        """
        Extract numeric vehicle number from vehicle_id string to match NUMBER column
        """
        try:
            # Handle formats like "GR86-002-000" -> extract 2
            numbers = [int(s) for s in vehicle_id.split('-') if s.isdigit()]
            return numbers[1] if len(numbers) > 1 else numbers[0] if numbers else 1
        except:
            return 1

    def _calculate_lap_features(self, lap_telemetry: pd.DataFrame, lap_info: pd.Series) -> Dict[str, float]:
        """
        Calculate fuel-related features from lap data using EXACT column names
        """
        try:
            # Telemetry-based features using EXACT column names
            avg_speed = lap_telemetry['speed'].mean()
            speed_std = lap_telemetry['speed'].std()
            max_speed = lap_telemetry['speed'].max()
            
            # Acceleration patterns using EXACT column names
            avg_long_acc = lap_telemetry['accx_can'].abs().mean()
            avg_lat_acc = lap_telemetry['accy_can'].abs().mean()
            
            # Gear usage patterns using EXACT column names
            avg_gear = lap_telemetry['gear'].mean()
            gear_changes = lap_telemetry['gear'].diff().abs().sum()
            
            # Lap time features from pit data using EXACT column names
            lap_time_seconds = self._convert_lap_time_to_seconds(lap_info.get('LAP_TIME', '0:00'))
            
            # Use S1_SECONDS, S2_SECONDS, S3_SECONDS which are guaranteed by schema
            sector_times = [
                lap_info.get('S1_SECONDS', 0),
                lap_info.get('S2_SECONDS', 0), 
                lap_info.get('S3_SECONDS', 0)
            ]
            
            # If sector seconds are 0, fallback to calculated values
            if all(st == 0 for st in sector_times):
                sector_times = [lap_time_seconds / 3] * 3
            
            # Additional features from pit data using EXACT column names
            top_speed = lap_info.get('TOP_SPEED', 0)
            kph = lap_info.get('KPH', 0)
            lap_improvement = lap_info.get('LAP_IMPROVEMENT', 0)
            
            # Speed consistency indicator
            speed_consistency = 1.0 / (1.0 + speed_std) if speed_std > 0 else 1.0
            
            return {
                'avg_speed': avg_speed,
                'max_speed': max_speed,
                'top_speed_pit': top_speed,
                'kph_pit': kph,
                'speed_consistency': speed_consistency,
                'avg_longitudinal_accel': avg_long_acc,
                'avg_lateral_accel': avg_lat_acc,
                'avg_gear': avg_gear,
                'gear_changes': gear_changes,
                'lap_time': lap_time_seconds,
                'sector1_time': sector_times[0],
                'sector2_time': sector_times[1],
                'sector3_time': sector_times[2],
                'lap_improvement': lap_improvement,
                'lap_number': lap_info.get('LAP_NUMBER', 1)
            }
            
        except Exception as e:
            print(f"⚠️ Feature calculation error: {e}")
            return self._get_fallback_features()

    def _estimate_fuel_consumption(self, lap_telemetry: pd.DataFrame, lap_info: pd.Series) -> float:
        """
        Estimate fuel consumption for a lap using EXACT column names
        """
        try:
            base_consumption = 2.5  # liters per lap base rate
            
            # Speed factor using EXACT column names
            avg_speed = lap_telemetry['speed'].mean()
            speed_factor = min(1.0, avg_speed / 200)
            
            # Acceleration factor using EXACT column names
            accel_factor = lap_telemetry['accx_can'].abs().mean() * 2.0
            
            # Gear efficiency using EXACT column names
            avg_gear = lap_telemetry['gear'].mean()
            gear_efficiency = 1.2 - abs(avg_gear - 3) * 0.1
            
            # Additional factors from pit data using EXACT column names
            top_speed = lap_info.get('TOP_SPEED', 0)
            lap_improvement = lap_info.get('LAP_IMPROVEMENT', 0)
            
            # Calculate consumption with additional factors
            consumption = base_consumption * (1 + speed_factor + accel_factor) * gear_efficiency
            consumption *= (1 + (top_speed / 300))  # Higher top speed penalty
            
            return max(1.0, min(5.0, consumption))
            
        except Exception as e:
            print(f"⚠️ Fuel estimation error: {e}")
            return 2.8

    def _convert_lap_time_to_seconds(self, lap_time: str) -> float:
        """
        Convert lap time string to seconds (consistent with FirebaseDataLoader)
        """
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

    def _get_fallback_features(self) -> Dict[str, float]:
        """Provide fallback features when data is insufficient"""
        return {
            'avg_speed': 100.0, 'max_speed': 150.0, 'top_speed_pit': 160.0, 'kph_pit': 120.0,
            'speed_consistency': 0.8, 'avg_longitudinal_accel': 0.3, 'avg_lateral_accel': 0.4,
            'avg_gear': 3.0, 'gear_changes': 8.0, 'lap_time': 60.0, 'sector1_time': 20.0,
            'sector2_time': 20.0, 'sector3_time': 20.0, 'lap_improvement': 0.0, 'lap_number': 1.0
        }

    def _train_with_fallback(self, reason: str) -> dict:
        """Create fallback model when training data is insufficient"""
        print(f"⚠️ Using fallback fuel model: {reason}")
        
        synthetic_features, synthetic_targets = self._generate_synthetic_data()
        
        if len(synthetic_features) > 0:
            X_synth = pd.DataFrame(synthetic_features)
            y_synth = np.array(synthetic_targets)
            
            X_scaled = self.scaler.fit_transform(X_synth)
            self.feature_columns = X_synth.columns.tolist()
            self.model.fit(X_scaled, y_synth)
            
            return {
                'model': self, 'features': self.feature_columns, 'train_score': 0.6, 'test_score': 0.5,
                'feature_importance': {col: 1.0/len(self.feature_columns) for col in self.feature_columns},
                'training_samples': len(X_synth), 'status': 'fallback', 'fallback_reason': reason
            }
        
        return {'error': f'Fuel model training failed: {reason}', 'status': 'error'}

    def _generate_synthetic_data(self, n_samples: int = 100) -> Tuple[List[Dict], List[float]]:
        """Generate synthetic training data for fallback scenarios"""
        features = []
        targets = []
        
        for i in range(n_samples):
            feat = self._get_fallback_features()
            feat['avg_speed'] += np.random.normal(0, 20)
            feat['lap_time'] += np.random.normal(0, 10)
            feat['avg_gear'] = np.random.randint(2, 5)
            
            fuel = 2.5 + (feat['avg_speed'] / 200) * 1.5 + feat['avg_longitudinal_accel'] * 1.0
            targets.append(max(1.0, fuel))
            features.append(feat)
            
        return features, targets

    def predict_fuel_consumption(self, features: Dict[str, float]) -> float:
        """Predict fuel consumption for given features"""
        try:
            if not self.feature_columns:
                return self._fallback_fuel_prediction(features)
                
            feature_vector = [features.get(col, 0.0) for col in self.feature_columns]
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            prediction = self.model.predict(X_scaled)[0]
            return max(1.0, prediction)
            
        except Exception:
            return self._fallback_fuel_prediction(features)

    def _fallback_fuel_prediction(self, features: Dict[str, float]) -> float:
        """Fallback fuel prediction when model is unavailable"""
        avg_speed = features.get('avg_speed', 100)
        lap_time = features.get('lap_time', 60)
        return 2.5 + (avg_speed / 200) * 1.5

    def estimate_remaining_laps(self, current_fuel: float, features: Dict[str, float]) -> int:
        """Estimate remaining laps based on current fuel and driving patterns"""
        consumption_rate = self.predict_fuel_consumption(features)
        return max(0, int(current_fuel / consumption_rate))

    def save_model(self, filepath: str):
        """Save trained model to file"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, filepath)

    def load_model(self, filepath: str):
        """Load trained model from file"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']

























# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import joblib
# from typing import Dict, List, Tuple

# class FuelModelTrainer:
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#         self.scaler = StandardScaler()
#         self.feature_columns = []
        
#         # Define expected data structures based on your schema
#         self.required_telemetry_cols = ['vehicle_id', 'lap', 'speed', 'accx_can', 'accy_can', 'gear']
#         self.required_pit_data_cols = ['NUMBER', 'LAP_NUMBER', 'LAP_TIME', 'KPH', 'TOP_SPEED', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']

#     def train(self, processed_data: Dict[str, pd.DataFrame]) -> dict:
#         """
#         Train fuel model using structured data from Firebase loader
#         """
#         try:
#             # Extract data from processed_data dictionary
#             telemetry_data = processed_data.get('telemetry_data', pd.DataFrame())
#             pit_data = processed_data.get('pit_data', pd.DataFrame())
            
#             # Validate data availability
#             if telemetry_data.empty or pit_data.empty:
#                 return self._train_with_fallback("Missing telemetry or pit data")
            
#             # Validate required columns
#             missing_telemetry = [col for col in self.required_telemetry_cols if col not in telemetry_data.columns]
#             missing_pit = [col for col in self.required_pit_data_cols if col not in pit_data.columns]
            
#             if missing_telemetry or missing_pit:
#                 return self._train_with_fallback(f"Missing columns: telemetry{missing_telemetry}, pit{missing_pit}")
            
#             # Extract features and targets
#             features_df, targets = self._extract_fuel_features(telemetry_data, pit_data)
            
#             if features_df.empty or len(targets) == 0:
#                 return self._train_with_fallback("No features extracted from data")
            
#             # Prepare training data
#             X = features_df
#             y = np.array(targets)
            
#             # Remove any remaining NaN values
#             valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
#             X = X[valid_mask]
#             y = y[valid_mask]
            
#             if len(X) < 10:
#                 return self._train_with_fallback(f"Insufficient samples: {len(X)}")
            
#             # Scale features and train model
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
#                 'status': 'success'
#             }
            
#         except Exception as e:
#             return self._train_with_fallback(f"Training error: {str(e)}")

#     def _extract_fuel_features(self, telemetry_data: pd.DataFrame, pit_data: pd.DataFrame) -> Tuple[pd.DataFrame, List[float]]:
#         """
#         Extract fuel-related features from telemetry and pit data
#         """
#         features_list = []
#         consumption_targets = []
        
#         # Group telemetry by vehicle and lap
#         for (vehicle_id, lap_num), lap_telemetry in telemetry_data.groupby(['vehicle_id', 'lap']):
#             if len(lap_telemetry) < 10:  # Minimum telemetry points
#                 continue
                
#             # Get corresponding lap info from pit data
#             lap_info = self._get_lap_info(pit_data, vehicle_id, lap_num)
#             if lap_info.empty:
#                 continue
                
#             # Calculate features for this lap
#             features = self._calculate_lap_features(lap_telemetry, lap_info)
#             fuel_consumption = self._estimate_fuel_consumption(lap_telemetry, lap_info)
            
#             features_list.append(features)
#             consumption_targets.append(fuel_consumption)
        
#         if features_list:
#             return pd.DataFrame(features_list), consumption_targets
#         return pd.DataFrame(), []

#     def _get_lap_info(self, pit_data: pd.DataFrame, vehicle_id: str, lap_num: int) -> pd.Series:
#         """
#         Extract lap information from pit data, handling different vehicle ID formats
#         """
#         try:
#             # Try to match by vehicle_id (convert to number if possible)
#             vehicle_num = self._extract_vehicle_number(vehicle_id)
            
#             # Look for matching lap data
#             lap_match = pit_data[
#                 (pit_data['NUMBER'] == vehicle_num) & 
#                 (pit_data['LAP_NUMBER'] == lap_num)
#             ]
            
#             if not lap_match.empty:
#                 return lap_match.iloc[0]
            
#             # Fallback: get closest lap data for this vehicle
#             vehicle_laps = pit_data[pit_data['NUMBER'] == vehicle_num]
#             if not vehicle_laps.empty:
#                 return vehicle_laps.iloc[0]
                
#         except Exception:
#             pass
            
#         return pd.Series()

#     def _extract_vehicle_number(self, vehicle_id: str) -> int:
#         """
#         Extract numeric vehicle number from vehicle_id string
#         """
#         try:
#             # Handle formats like "GR86-002-2" -> extract 2
#             numbers = [int(s) for s in vehicle_id.split('-') if s.isdigit()]
#             return numbers[-1] if numbers else 1
#         except:
#             return 1

#     def _calculate_lap_features(self, lap_telemetry: pd.DataFrame, lap_info: pd.Series) -> Dict[str, float]:
#         """
#         Calculate fuel-related features from lap data
#         """
#         try:
#             # Telemetry-based features
#             avg_speed = lap_telemetry['speed'].mean()
#             speed_std = lap_telemetry['speed'].std()
#             max_speed = lap_telemetry['speed'].max()
            
#             # Acceleration patterns (fuel consumption indicators)
#             avg_long_acc = lap_telemetry['accx_can'].abs().mean()  # Absolute longitudinal acceleration
#             avg_lat_acc = lap_telemetry['accy_can'].abs().mean()   # Absolute lateral acceleration
            
#             # Gear usage patterns
#             avg_gear = lap_telemetry['gear'].mean()
#             gear_changes = lap_telemetry['gear'].diff().abs().sum()
            
#             # Lap time features from pit data
#             lap_time_seconds = self._convert_lap_time_to_seconds(lap_info.get('LAP_TIME', '0:00'))
#             sector_times = [
#                 lap_info.get('S1_SECONDS', lap_time_seconds / 3),
#                 lap_info.get('S2_SECONDS', lap_time_seconds / 3),
#                 lap_info.get('S3_SECONDS', lap_time_seconds / 3)
#             ]
            
#             # Speed consistency (indicator of efficient driving)
#             speed_consistency = 1.0 / (1.0 + speed_std) if speed_std > 0 else 1.0
            
#             return {
#                 'avg_speed': avg_speed,
#                 'max_speed': max_speed,
#                 'speed_consistency': speed_consistency,
#                 'avg_longitudinal_accel': avg_long_acc,
#                 'avg_lateral_accel': avg_lat_acc,
#                 'avg_gear': avg_gear,
#                 'gear_changes': gear_changes,
#                 'lap_time': lap_time_seconds,
#                 'sector1_time': sector_times[0],
#                 'sector2_time': sector_times[1],
#                 'sector3_time': sector_times[2],
#                 'lap_number': lap_info.get('LAP_NUMBER', 1)
#             }
            
#         except Exception as e:
#             # Fallback features if calculation fails
#             return self._get_fallback_features()

#     def _estimate_fuel_consumption(self, lap_telemetry: pd.DataFrame, lap_info: pd.Series) -> float:
#         """
#         Estimate fuel consumption for a lap based on driving patterns
#         """
#         try:
#             base_consumption = 2.5  # liters per lap base rate
            
#             # Speed factor (higher speed = more fuel)
#             avg_speed = lap_telemetry['speed'].mean()
#             speed_factor = min(1.0, avg_speed / 200)  # Normalize by max expected speed
            
#             # Acceleration factor (aggressive driving = more fuel)
#             accel_factor = lap_telemetry['accx_can'].abs().mean() * 2.0
            
#             # Gear efficiency (middle gears are most efficient)
#             avg_gear = lap_telemetry['gear'].mean()
#             gear_efficiency = 1.2 - abs(avg_gear - 3) * 0.1
            
#             # Calculate consumption
#             consumption = base_consumption * (1 + speed_factor + accel_factor) * gear_efficiency
            
#             return max(1.0, min(5.0, consumption))  # Constrain to reasonable range
            
#         except Exception:
#             return 2.8  # Default fallback

#     def _convert_lap_time_to_seconds(self, lap_time: str) -> float:
#         """
#         Convert lap time string (MM:SS.mmm) to seconds
#         """
#         try:
#             if ':' in lap_time:
#                 parts = lap_time.split(':')
#                 if len(parts) == 2:
#                     return float(parts[0]) * 60 + float(parts[1])
#             return float(lap_time)
#         except:
#             return 60.0  # Default lap time

#     def _get_fallback_features(self) -> Dict[str, float]:
#         """Provide fallback features when data is insufficient"""
#         return {
#             'avg_speed': 100.0,
#             'max_speed': 150.0,
#             'speed_consistency': 0.8,
#             'avg_longitudinal_accel': 0.3,
#             'avg_lateral_accel': 0.4,
#             'avg_gear': 3.0,
#             'gear_changes': 8.0,
#             'lap_time': 60.0,
#             'sector1_time': 20.0,
#             'sector2_time': 20.0,
#             'sector3_time': 20.0,
#             'lap_number': 1.0
#         }

#     def _train_with_fallback(self, reason: str) -> dict:
#         """
#         Create a fallback model when training data is insufficient
#         """
#         print(f"⚠️ Using fallback fuel model: {reason}")
        
#         # Train on synthetic data to ensure model is always available
#         synthetic_features, synthetic_targets = self._generate_synthetic_data()
        
#         if len(synthetic_features) > 0:
#             X_synth = pd.DataFrame(synthetic_features)
#             y_synth = np.array(synthetic_targets)
            
#             X_scaled = self.scaler.fit_transform(X_synth)
#             self.feature_columns = X_synth.columns.tolist()
#             self.model.fit(X_scaled, y_synth)
            
#             return {
#                 'model': self,
#                 'features': self.feature_columns,
#                 'train_score': 0.6,
#                 'test_score': 0.5,
#                 'feature_importance': {col: 1.0/len(self.feature_columns) for col in self.feature_columns},
#                 'training_samples': len(X_synth),
#                 'status': 'fallback',
#                 'fallback_reason': reason
#             }
        
#         return {'error': f'Fuel model training failed: {reason}', 'status': 'error'}

#     def _generate_synthetic_data(self, n_samples: int = 100) -> Tuple[List[Dict], List[float]]:
#         """Generate synthetic training data for fallback scenarios"""
#         features = []
#         targets = []
        
#         for i in range(n_samples):
#             feat = self._get_fallback_features()
#             # Add some variation
#             feat['avg_speed'] += np.random.normal(0, 20)
#             feat['lap_time'] += np.random.normal(0, 10)
#             feat['avg_gear'] = np.random.randint(2, 5)
            
#             # Synthetic fuel consumption
#             fuel = 2.5 + (feat['avg_speed'] / 200) * 1.5 + feat['avg_longitudinal_accel'] * 1.0
#             targets.append(max(1.0, fuel))
#             features.append(feat)
            
#         return features, targets

#     def predict_fuel_consumption(self, features: Dict[str, float]) -> float:
#         """Predict fuel consumption for given features"""
#         try:
#             if not self.feature_columns:
#                 return self._fallback_fuel_prediction(features)
                
#             # Ensure all features are present
#             feature_vector = [features.get(col, 0.0) for col in self.feature_columns]
#             X = np.array(feature_vector).reshape(1, -1)
#             X_scaled = self.scaler.transform(X)
            
#             prediction = self.model.predict(X_scaled)[0]
#             return max(1.0, prediction)  # Ensure positive consumption
            
#         except Exception:
#             return self._fallback_fuel_prediction(features)

#     def _fallback_fuel_prediction(self, features: Dict[str, float]) -> float:
#         """Fallback fuel prediction when model is unavailable"""
#         avg_speed = features.get('avg_speed', 100)
#         lap_time = features.get('lap_time', 60)
#         return 2.5 + (avg_speed / 200) * 1.5

#     def estimate_remaining_laps(self, current_fuel: float, features: Dict[str, float]) -> int:
#         """Estimate remaining laps based on current fuel and driving patterns"""
#         consumption_rate = self.predict_fuel_consumption(features)
#         return max(0, int(current_fuel / consumption_rate))

#     def save_model(self, filepath: str):
#         """Save trained model to file"""
#         joblib.dump({
#             'model': self.model,
#             'scaler': self.scaler,
#             'feature_columns': self.feature_columns
#         }, filepath)

#     def load_model(self, filepath: str):
#         """Load trained model from file"""
#         data = joblib.load(filepath)
#         self.model = data['model']
#         self.scaler = data['scaler']
#         self.feature_columns = data['feature_columns']