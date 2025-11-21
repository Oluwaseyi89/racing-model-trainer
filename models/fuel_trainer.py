import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, List, Tuple
import logging

class FuelModelTrainer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.logger = logging.getLogger(__name__)
        
        # Minimum required columns for flexibility
        self.minimum_telemetry_cols = ['vehicle_id', 'lap', 'speed']
        self.minimum_pit_cols = ['NUMBER', 'LAP_NUMBER', 'LAP_TIME']

    def train(self, track_data: Dict[str, Dict[str, pd.DataFrame]]) -> dict:
        """
        Train fuel model with enhanced error handling and vehicle_id mapping
        """
        try:
            self.logger.info("🚀 Starting fuel model training...")
            
            # Process each track individually for better debugging
            all_features_list = []
            all_targets_list = []
            processed_tracks = []
            
            for track_name, data_dict in track_data.items():
                self.logger.info(f"📊 Processing track: {track_name}")
                
                if not self._validate_track_data(data_dict):
                    self.logger.warning(f"⚠️ Skipping {track_name}: validation failed")
                    continue
                    
                track_features, track_targets = self._extract_track_fuel_features(data_dict, track_name)
                
                if not track_features.empty and len(track_targets) > 0:
                    all_features_list.append(track_features)
                    all_targets_list.extend(track_targets)
                    processed_tracks.append(track_name)
                    self.logger.info(f"✅ {track_name}: extracted {len(track_features)} samples")
                else:
                    self.logger.warning(f"⚠️ No fuel features extracted from {track_name}")

            if not all_features_list:
                return self._train_with_fallback("No valid fuel features extracted from any track", processed_tracks)
            
            # Combine all track data
            X = pd.concat(all_features_list, ignore_index=True)
            y = np.array(all_targets_list)
            
            if X.empty or len(y) == 0:
                return self._train_with_fallback("Empty feature or target matrices after processing", processed_tracks)

            self.logger.info(f"📈 Final dataset: {len(X)} samples, {len(X.columns)} features")

            # Clean data
            valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            if len(X_clean) < 10:
                return self._train_with_fallback(f"Insufficient samples: {len(X_clean)}", processed_tracks)
            
            # Scale features and train model
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

            self.logger.info(f"✅ Fuel model trained successfully - Test Score: {test_score:.3f}")

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
            error_msg = f'Training error: {str(e)}'
            self.logger.error(f"❌ {error_msg}")
            return self._train_with_fallback(error_msg, [])

    def _validate_track_data(self, data_dict: Dict[str, pd.DataFrame]) -> bool:
        """Enhanced validation with better logging"""
        telemetry_data = data_dict.get('telemetry_data', pd.DataFrame())
        pit_data = data_dict.get('pit_data', pd.DataFrame())
        
        if telemetry_data.empty or pit_data.empty:
            self.logger.debug("⚠️ Missing telemetry_data or pit_data")
            return False
            
        # Check for minimum required columns
        missing_telemetry = [col for col in self.minimum_telemetry_cols if col not in telemetry_data.columns]
        missing_pit = [col for col in self.minimum_pit_cols if col not in pit_data.columns]
        
        if missing_telemetry or missing_pit:
            self.logger.debug(f"⚠️ Missing required columns - telemetry: {missing_telemetry}, pit: {missing_pit}")
            return False
            
        # Check for minimum data volume
        if len(telemetry_data) < 5 or len(pit_data) < 3:
            self.logger.debug(f"⚠️ Insufficient data - telemetry: {len(telemetry_data)}, pit: {len(pit_data)}")
            return False
            
        self.logger.debug(f"✅ Track data validated: {len(telemetry_data)} telemetry rows, {len(pit_data)} pit rows")
        return True

    def _extract_track_fuel_features(self, data_dict: Dict[str, pd.DataFrame], track_name: str) -> Tuple[pd.DataFrame, List[float]]:
        """
        Extract fuel-related features with enhanced vehicle_id mapping and error handling
        """
        telemetry_data = data_dict['telemetry_data']
        pit_data = data_dict['pit_data']
        
        features_list = []
        consumption_targets = []
        
        # Add vehicle_id mapping to pit_data for consistent matching
        pit_data = self._add_vehicle_id_mapping(pit_data)
        
        self.logger.debug(f"🔧 Processing {len(telemetry_data['vehicle_id'].unique())} vehicles in {track_name}")

        # Group telemetry by vehicle and lap
        for (vehicle_id, lap_num), lap_telemetry in telemetry_data.groupby(['vehicle_id', 'lap']):
            if len(lap_telemetry) < 5:  # Reduced minimum telemetry points
                self.logger.debug(f"⚠️ Vehicle {vehicle_id} lap {lap_num}: insufficient telemetry points ({len(lap_telemetry)})")
                continue
                
            # Get corresponding lap info from pit data
            lap_info = self._get_lap_info(pit_data, vehicle_id, lap_num)
            if lap_info.empty:
                self.logger.debug(f"⚠️ Vehicle {vehicle_id} lap {lap_num}: no matching pit data")
                continue
                
            try:
                # Calculate features for this lap
                features = self._calculate_lap_features(lap_telemetry, lap_info, track_name)
                fuel_consumption = self._estimate_fuel_consumption(lap_telemetry, lap_info)
                
                features_list.append(pd.DataFrame([features]))
                consumption_targets.append(fuel_consumption)
                
                self.logger.debug(f"✅ Vehicle {vehicle_id} lap {lap_num}: extracted features")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Feature extraction failed for vehicle {vehicle_id} lap {lap_num}: {e}")
                continue
        
        if features_list:
            result_features = pd.concat(features_list, ignore_index=True)
            self.logger.debug(f"✅ {track_name}: extracted {len(result_features)} fuel samples")
            return result_features, consumption_targets
        
        self.logger.debug(f"❌ {track_name}: no fuel features extracted")
        return pd.DataFrame(), []

    def _add_vehicle_id_mapping(self, pit_data: pd.DataFrame) -> pd.DataFrame:
        """Create vehicle_id mapping for consistent telemetry integration"""
        pit_data = pit_data.copy()
        
        # Create consistent vehicle_id format matching telemetry data
        pit_data['vehicle_id'] = 'GR86-' + pit_data['NUMBER'].astype(str).str.zfill(3) + '-000'
        
        return pit_data

    def _get_lap_info(self, pit_data: pd.DataFrame, vehicle_id: str, lap_num: int) -> pd.Series:
        """
        Extract lap information from pit data with enhanced matching
        """
        try:
            # Look for exact match first
            lap_match = pit_data[
                (pit_data['vehicle_id'] == vehicle_id) & 
                (pit_data['LAP_NUMBER'] == lap_num)
            ]
            
            if not lap_match.empty:
                return lap_match.iloc[0]
            
            # Fallback: get closest lap data for this vehicle
            vehicle_laps = pit_data[pit_data['vehicle_id'] == vehicle_id]
            if not vehicle_laps.empty:
                # Find closest lap number
                lap_diffs = abs(vehicle_laps['LAP_NUMBER'] - lap_num)
                closest_idx = lap_diffs.idxmin()
                if lap_diffs[closest_idx] <= 2:  # Allow 2 lap difference
                    return vehicle_laps.loc[closest_idx]
                    
        except Exception as e:
            self.logger.debug(f"⚠️ Lap info lookup failed for {vehicle_id} lap {lap_num}: {e}")
            
        return pd.Series()

    def _calculate_lap_features(self, lap_telemetry: pd.DataFrame, lap_info: pd.Series, track_name: str) -> Dict[str, float]:
        """
        Calculate fuel-related features with robust error handling
        """
        try:
            # Telemetry-based features
            avg_speed = lap_telemetry['speed'].mean()
            speed_std = lap_telemetry['speed'].std()
            max_speed = lap_telemetry['speed'].max()
            
            # Acceleration patterns
            avg_long_acc = lap_telemetry['accx_can'].abs().mean() if 'accx_can' in lap_telemetry.columns else 0.3
            avg_lat_acc = lap_telemetry['accy_can'].abs().mean() if 'accy_can' in lap_telemetry.columns else 0.4
            
            # Gear usage patterns
            avg_gear = lap_telemetry['gear'].mean() if 'gear' in lap_telemetry.columns else 3.0
            gear_changes = lap_telemetry['gear'].diff().abs().sum() if 'gear' in lap_telemetry.columns else 8.0
            
            # Lap time features from pit data
            lap_time_seconds = self._convert_lap_time_to_seconds(lap_info.get('LAP_TIME', '0:00'))
            
            # Sector times with fallbacks
            sector_times = [
                lap_info.get('S1_SECONDS', 0),
                lap_info.get('S2_SECONDS', 0), 
                lap_info.get('S3_SECONDS', 0)
            ]
            
            # If sector seconds are invalid, use calculated values
            if all(st == 0 for st in sector_times) or any(pd.isna(st) for st in sector_times):
                sector_times = [lap_time_seconds / 3] * 3
            
            # Additional features from pit data with fallbacks
            top_speed = lap_info.get('TOP_SPEED', 0)
            kph = lap_info.get('KPH', 0)
            lap_improvement = lap_info.get('LAP_IMPROVEMENT', 0)
            
            # Speed consistency indicator
            speed_consistency = 1.0 / (1.0 + speed_std) if speed_std > 0 else 1.0
            
            # Track-specific factors
            track_factor = self._get_track_fuel_factor(track_name)
            
            features = {
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
                'lap_number': lap_info.get('LAP_NUMBER', 1),
                'track_fuel_factor': track_factor,
                'is_high_speed_track': 1.0 if avg_speed > 150 else 0.0,
                'acceleration_intensity': (avg_long_acc + avg_lat_acc) / 2
            }
            
            return features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature calculation failed: {e}")
            return self._get_fallback_features(track_name)

    def _get_track_fuel_factor(self, track_name: str) -> float:
        """Get track-specific fuel consumption factor"""
        high_fuel_tracks = ['road_america', 'cota']  # Long tracks with high speeds
        medium_fuel_tracks = ['indianapolis', 'virginia']
        low_fuel_tracks = ['barber', 'sonoma', 'sebring']  # Technical tracks
        
        track_lower = track_name.lower()
        if any(track in track_lower for track in high_fuel_tracks):
            return 1.2
        elif any(track in track_lower for track in medium_fuel_tracks):
            return 1.0
        else:
            return 0.8

    def _estimate_fuel_consumption(self, lap_telemetry: pd.DataFrame, lap_info: pd.Series) -> float:
        """
        Estimate fuel consumption for a lap with enhanced calculation
        """
        try:
            base_consumption = 2.5  # liters per lap base rate
            
            # Speed factor
            avg_speed = lap_telemetry['speed'].mean()
            speed_factor = min(1.5, avg_speed / 150)  # Adjusted threshold
            
            # Acceleration factor
            accel_factor = 0.0
            if 'accx_can' in lap_telemetry.columns:
                accel_factor = lap_telemetry['accx_can'].abs().mean() * 1.5
            
            # Gear efficiency
            gear_efficiency = 1.0
            if 'gear' in lap_telemetry.columns:
                avg_gear = lap_telemetry['gear'].mean()
                # Optimal gear range is 3-4 for fuel efficiency
                gear_efficiency = 1.3 - abs(avg_gear - 3.5) * 0.15
            
            # Additional factors from pit data
            top_speed = lap_info.get('TOP_SPEED', 0)
            speed_penalty = (top_speed / 250) if top_speed > 0 else 0.2
            
            # Calculate consumption with multiple factors
            consumption = base_consumption * (1 + speed_factor) * (1 + accel_factor) * gear_efficiency
            consumption *= (1 + speed_penalty)
            
            return max(1.0, min(6.0, consumption))  # Wider range
            
        except Exception as e:
            self.logger.warning(f"⚠️ Fuel estimation failed: {e}")
            return 2.8  # Conservative default

    def _convert_lap_time_to_seconds(self, lap_time: str) -> float:
        """Convert lap time string to seconds with robust parsing"""
        try:
            if pd.isna(lap_time) or lap_time == 0:
                return 90.0  # More realistic default
                
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
            return 90.0

    def _get_fallback_features(self, track_name: str) -> Dict[str, float]:
        """Provide fallback features when data is insufficient"""
        track_factor = self._get_track_fuel_factor(track_name)
        
        return {
            'avg_speed': 120.0, 'max_speed': 160.0, 'top_speed_pit': 170.0, 'kph_pit': 130.0,
            'speed_consistency': 0.7, 'avg_longitudinal_accel': 0.4, 'avg_lateral_accel': 0.5,
            'avg_gear': 3.5, 'gear_changes': 10.0, 'lap_time': 90.0, 'sector1_time': 30.0,
            'sector2_time': 30.0, 'sector3_time': 30.0, 'lap_improvement': 0.0, 'lap_number': 5.0,
            'track_fuel_factor': track_factor, 'is_high_speed_track': 0.0, 'acceleration_intensity': 0.45
        }

    def _train_with_fallback(self, reason: str, processed_tracks: List[str]) -> dict:
        """Create fallback model with enhanced synthetic data"""
        self.logger.warning(f"⚠️ Using fallback fuel model: {reason}")
        
        synthetic_features, synthetic_targets = self._generate_enhanced_synthetic_data()
        
        if len(synthetic_features) > 0:
            X_synth = pd.DataFrame(synthetic_features)
            y_synth = np.array(synthetic_targets)
            
            X_scaled = self.scaler.fit_transform(X_synth)
            self.feature_columns = X_synth.columns.tolist()
            self.model.fit(X_scaled, y_synth)
            
            self.logger.info("✅ Fallback fuel model trained with synthetic data")
            
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
            'error': f'Fuel model training failed: {reason}', 
            'status': 'error',
            'tracks_used': len(processed_tracks),
            'processed_tracks': processed_tracks
        }

    def _generate_enhanced_synthetic_data(self, n_samples: int = 150) -> Tuple[List[Dict], List[float]]:
        """Generate enhanced synthetic training data"""
        features = []
        targets = []
        
        track_types = ['high_speed', 'technical', 'balanced']
        
        for i in range(n_samples):
            track_type = np.random.choice(track_types)
            
            if track_type == 'high_speed':
                base_speed = 160
                base_consumption = 3.0
            elif track_type == 'technical':
                base_speed = 100
                base_consumption = 2.2
            else:  # balanced
                base_speed = 130
                base_consumption = 2.5
            
            feat = self._get_fallback_features(track_type)
            feat['avg_speed'] = base_speed + np.random.normal(0, 15)
            feat['max_speed'] = feat['avg_speed'] + np.random.normal(20, 5)
            feat['lap_time'] = 60 + (180 - feat['avg_speed']) * 0.5 + np.random.normal(0, 10)
            feat['avg_gear'] = np.random.randint(2, 5)
            feat['gear_changes'] = np.random.randint(5, 15)
            feat['avg_longitudinal_accel'] = 0.3 + np.random.random() * 0.4
            feat['avg_lateral_accel'] = 0.4 + np.random.random() * 0.3
            
            # More realistic fuel calculation
            fuel = base_consumption + (feat['avg_speed'] / 200) * 1.0 + feat['avg_longitudinal_accel'] * 0.8
            fuel += feat['avg_lateral_accel'] * 0.5 + (feat['gear_changes'] / 20) * 0.3
            
            targets.append(max(1.5, min(5.5, fuel)))
            features.append(feat)
            
        return features, targets

    def predict_fuel_consumption(self, features: Dict[str, float]) -> float:
        """Predict fuel consumption for given features"""
        try:
            if not self.feature_columns:
                self.logger.warning("⚠️ No trained model available, using fallback")
                return self._fallback_fuel_prediction(features)
                
            # Ensure all features are present
            feature_vector = [features.get(col, 0.0) for col in self.feature_columns]
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            prediction = self.model.predict(X_scaled)[0]
            return max(1.0, min(6.0, prediction))
            
        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {e}")
            return self._fallback_fuel_prediction(features)

    def _fallback_fuel_prediction(self, features: Dict[str, float]) -> float:
        """Enhanced fallback fuel prediction"""
        avg_speed = features.get('avg_speed', 120)
        lap_time = features.get('lap_time', 90)
        accel_intensity = features.get('acceleration_intensity', 0.4)
        
        base_consumption = 2.5
        speed_factor = (avg_speed / 150) * 1.2
        time_factor = (lap_time / 100) * 0.8
        accel_factor = accel_intensity * 1.5
        
        return max(1.5, base_consumption + speed_factor + time_factor + accel_factor)

    def estimate_remaining_laps(self, current_fuel: float, features: Dict[str, float]) -> int:
        """Estimate remaining laps based on current fuel and driving patterns"""
        try:
            consumption_rate = self.predict_fuel_consumption(features)
            if consumption_rate <= 0:
                return 0
            remaining_laps = current_fuel / consumption_rate
            return max(0, int(remaining_laps))
        except Exception as e:
            self.logger.error(f"❌ Remaining laps estimation failed: {e}")
            return max(0, int(current_fuel / 2.5))  # Conservative fallback

    def get_fuel_efficiency_rating(self, features: Dict[str, float]) -> str:
        """Get fuel efficiency rating based on consumption"""
        try:
            consumption = self.predict_fuel_consumption(features)
            if consumption < 2.0:
                return "Excellent"
            elif consumption < 3.0:
                return "Good"
            elif consumption < 4.0:
                return "Average"
            else:
                return "Poor"
        except:
            return "Unknown"

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
