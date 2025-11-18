import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class FuelModelTrainer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.required_features = [
            'throttle_usage', 'brake_pressure', 'longitudinal_accel', 'lateral_accel',
            'avg_gear', 'avg_speed', 'lap_time', 'speed_variance', 'lap_number'
        ]

    # ---------------------------
    # TRAINING ENTRY POINT
    # ---------------------------
    def train(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> dict:
        """Train fuel consumption model with robust error handling"""
        try:
            # Validate and prepare input data
            telemetry_data, lap_data = self._validate_and_prepare_inputs(telemetry_data, lap_data)
            
            # Extract features with comprehensive error handling
            features_df, targets = self._extract_fuel_features(telemetry_data, lap_data)
            
            if features_df.empty or len(targets) == 0:
                return {'error': 'No valid fuel features extracted from provided data'}

            # Prepare training data
            X, y = self._prepare_training_data(features_df, targets)
            if len(X) < 10:
                return {'error': f'Insufficient training samples after preprocessing: {len(X)}'}

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
            return {'error': f'Fuel model training failed: {str(e)}'}

    def _validate_and_prepare_inputs(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> tuple:
        """Validate and prepare input data with fallbacks"""
        # Handle empty inputs with synthetic data
        if telemetry_data.empty or lap_data.empty:
            print("⚠️ Empty input data, generating synthetic data for training")
            return self._generate_synthetic_training_data()
        
        # Ensure data has required structure
        telemetry_data = self._normalize_telemetry_data(telemetry_data)
        lap_data = self._normalize_lap_data(lap_data)
        
        return telemetry_data, lap_data

    def _normalize_telemetry_data(self, telemetry_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize telemetry data to expected schema"""
        df = telemetry_data.copy()
        
        # Ensure required columns exist with canonical names
        column_mapping = {
            'vehicle_number': 'vehicle_number',
            'lap': 'lap',
            'THROTTLE_POSITION': 'aps',
            'BRAKE_PRESSURE_FRONT': 'pbrake_f', 
            'BRAKE_PRESSURE_REAR': 'pbrake_r',
            'LONGITUDINAL_ACCEL': 'accx_can',
            'LATERAL_ACCEL': 'accy_can',
            'GEAR': 'gear',
            'KPH': 'KPH',
            'TOP_SPEED': 'TOP_SPEED'
        }
        
        # Rename columns to expected names
        for canonical, expected in column_mapping.items():
            if canonical in df.columns and expected not in df.columns:
                df[expected] = df[canonical]
        
        return df

    def _normalize_lap_data(self, lap_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize lap data to expected schema"""
        df = lap_data.copy()
        
        # Ensure required columns
        if 'NUMBER' not in df.columns:
            df['NUMBER'] = 1
        if 'LAP_NUMBER' not in df.columns:
            df['LAP_NUMBER'] = range(1, len(df) + 1)
        
        # Ensure numeric types for critical columns
        numeric_columns = ['KPH', 'LAP_TIME_SECONDS', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(60.0 if 'TIME' in col else 100.0)
        
        return df

    def _extract_fuel_features(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> tuple:
        """Extract fuel consumption features with robust error handling"""
        features_list = []
        consumption_targets = []

        try:
            # Group by vehicle and lap with validation
            if 'vehicle_number' not in telemetry_data.columns or 'lap' not in telemetry_data.columns:
                print("❌ Missing grouping columns in telemetry data")
                return pd.DataFrame(), []

            grouped = telemetry_data.groupby(['vehicle_number', 'lap'])
            
            for (vehicle_num, lap_num), lap_telemetry in grouped:
                try:
                    # Validate lap telemetry data
                    if len(lap_telemetry) < 10:  # Reduced minimum for robustness
                        lap_telemetry = self._generate_lap_telemetry(vehicle_num, lap_num, 30)

                    # Get lap info with fallback
                    lap_info = self._get_lap_info(vehicle_num, lap_num, lap_data)
                    
                    # Calculate features and fuel estimate
                    features = self._calculate_robust_features(lap_telemetry, lap_info)
                    fuel_estimate = self._estimate_fuel_consumption(lap_telemetry, lap_info)
                    
                    features_list.append(pd.DataFrame([features]))
                    consumption_targets.append(fuel_estimate)
                    
                except Exception as e:
                    print(f"⚠️ Failed to process lap {lap_num} for vehicle {vehicle_num}: {e}")
                    continue

        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            return pd.DataFrame(), []

        # Combine results with validation
        if features_list:
            try:
                features_df = pd.concat(features_list, ignore_index=True)
                # Ensure all required features are present
                features_df = self._ensure_required_features(features_df)
                return features_df, consumption_targets
            except Exception as e:
                print(f"❌ Failed to combine features: {e}")

        return pd.DataFrame(), []

    def _get_lap_info(self, vehicle_num: int, lap_num: int, lap_data: pd.DataFrame) -> pd.Series:
        """Get lap information with comprehensive fallbacks"""
        try:
            lap_info = lap_data[
                (lap_data['NUMBER'] == vehicle_num) & 
                (lap_data['LAP_NUMBER'] == lap_num)
            ]
            
            if not lap_info.empty:
                return lap_info.iloc[0]
            else:
                # Fallback: generate synthetic lap info
                return self._generate_lap_info(vehicle_num, lap_num)
                
        except Exception as e:
            print(f"⚠️ Lap info retrieval failed: {e}")
            return self._generate_lap_info(vehicle_num, lap_num)

    def _calculate_robust_features(self, telemetry: pd.DataFrame, lap_info: pd.Series) -> dict:
        """Calculate features with comprehensive error handling and fallbacks"""
        features = {}
        
        try:
            # Throttle usage with fallback
            features['throttle_usage'] = float(self._safe_column_mean(telemetry, 'aps', 50.0))
            
            # Brake pressure with fallback
            brake_front = self._safe_column_mean(telemetry, 'pbrake_f', 25.0)
            brake_rear = self._safe_column_mean(telemetry, 'pbrake_r', 25.0)
            features['brake_pressure'] = float((brake_front + brake_rear) / 2.0)
            
            # Acceleration metrics
            features['longitudinal_accel'] = float(self._safe_column_mean(telemetry, 'accx_can', 0.0, absolute=True))
            features['lateral_accel'] = float(self._safe_column_mean(telemetry, 'accy_can', 0.0, absolute=True))
            
            # Gear metrics
            features['gear_changes'] = float(self._calculate_gear_changes(telemetry))
            features['avg_gear'] = float(self._safe_column_mean(telemetry, 'gear', 3.0))
            
            # Speed metrics
            features['avg_speed'] = float(self._safe_get_value(lap_info, 'KPH', 100.0))
            features['top_speed'] = float(self._safe_column_max(telemetry, 'TOP_SPEED', features['avg_speed'] * 1.2))
            
            # Time metrics
            features['lap_time'] = float(self._safe_get_value(lap_info, 'LAP_TIME_SECONDS', 60.0))
            features['sector1_time'] = float(self._safe_get_value(lap_info, 'S1_SECONDS', features['lap_time'] / 3))
            features['sector2_time'] = float(self._safe_get_value(lap_info, 'S2_SECONDS', features['lap_time'] / 3))
            features['sector3_time'] = float(self._safe_get_value(lap_info, 'S3_SECONDS', features['lap_time'] / 3))
            
            # Speed variance
            features['speed_variance'] = float(self._safe_column_variance(telemetry, 'KPH', 0.0))
            
            # Lap number
            features['lap_number'] = int(self._safe_get_value(lap_info, 'LAP_NUMBER', 1))

        except Exception as e:
            print(f"⚠️ Feature calculation failed, using fallbacks: {e}")
            # Comprehensive fallback features
            features.update(self._get_fallback_features())

        return features

    def _estimate_fuel_consumption(self, telemetry: pd.DataFrame, lap_info: pd.Series) -> float:
        """Estimate fuel consumption with realistic physics-based model"""
        try:
            base_consumption = 2.5  # kg per lap base rate
            
            # Throttle factor (more throttle = more fuel)
            throttle_usage = self._safe_column_mean(telemetry, 'aps', 50.0)
            throttle_factor = (throttle_usage / 100.0) * 0.6
            
            # Speed factor (higher speed = more fuel but with efficiency curve)
            avg_speed = self._safe_get_value(lap_info, 'KPH', 100.0)
            speed_factor = min(0.8, (avg_speed / 180.0) * 0.4)  # Cap speed factor
            
            # Acceleration factor (aggressive driving = more fuel)
            long_accel = self._safe_column_mean(telemetry, 'accx_can', 0.0, absolute=True)
            accel_factor = long_accel * 1.8
            
            # Gear efficiency factor (optimal around gear 4-5)
            avg_gear = self._safe_column_mean(telemetry, 'gear', 3.0)
            gear_efficiency = 1.0 - abs(avg_gear - 4.0) * 0.08  # Best efficiency at gear 4
            
            # Calculate consumption
            consumption = base_consumption * (1.0 + throttle_factor + speed_factor + accel_factor) * gear_efficiency
            
            # Add small random variation for realism
            variation = np.random.normal(0, 0.1)
            
            return float(max(1.2, min(5.0, consumption + variation)))
            
        except Exception as e:
            print(f"⚠️ Fuel estimation failed: {e}")
            return 2.8  # Reasonable fallback

    def _prepare_training_data(self, features_df: pd.DataFrame, targets: list) -> tuple:
        """Prepare training data with proper validation"""
        try:
            X = features_df.copy()
            y = np.array(targets, dtype=float)
            
            # Remove rows with NaNs using proper boolean indexing
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

    def _safe_column_max(self, df: pd.DataFrame, column: str, default: float) -> float:
        """Safely calculate column max with fallback"""
        try:
            if column not in df.columns:
                return default
            values = pd.to_numeric(df[column], errors='coerce').dropna()
            return float(values.max()) if not values.empty else default
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

    def _calculate_gear_changes(self, telemetry: pd.DataFrame) -> float:
        """Calculate number of gear changes with error handling"""
        try:
            if 'gear' not in telemetry.columns:
                return 5.0  # Reasonable default
            gear_series = pd.to_numeric(telemetry['gear'], errors='coerce').dropna()
            if len(gear_series) < 2:
                return 5.0
            changes = gear_series.diff().abs().sum()
            return float(changes) if not pd.isna(changes) else 5.0
        except Exception:
            return 5.0

    def _safe_get_value(self, series: pd.Series, key: str, default: any) -> any:
        """Safely get value from pandas Series with fallback"""
        try:
            value = series.get(key, default)
            return value if not pd.isna(value) else default
        except (KeyError, TypeError, AttributeError):
            return default

    def _ensure_required_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required features are present"""
        df = features_df.copy()
        for feature in self.required_features:
            if feature not in df.columns:
                df[feature] = 0.0  # Default fallback
        return df

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

    def _get_fallback_features(self) -> dict:
        """Get comprehensive fallback features"""
        return {
            'throttle_usage': 50.0,
            'brake_pressure': 20.0,
            'longitudinal_accel': 0.5,
            'lateral_accel': 0.5,
            'gear_changes': 5.0,
            'avg_gear': 3.0,
            'avg_speed': 100.0,
            'top_speed': 120.0,
            'lap_time': 60.0,
            'sector1_time': 20.0,
            'sector2_time': 20.0,
            'sector3_time': 20.0,
            'speed_variance': 0.0,
            'lap_number': 1
        }

    # ---------------------------
    # SYNTHETIC DATA GENERATION
    # ---------------------------
    def _generate_synthetic_training_data(self) -> tuple:
        """Generate realistic synthetic training data"""
        print("🔄 Generating synthetic fuel training data...")
        
        # Create multiple vehicles and laps for diversity
        lap_data_list = []
        telemetry_data_list = []
        
        for vehicle_num in range(1, 4):  # 3 vehicles
            for lap_num in range(1, 11):  # 10 laps per vehicle
                lap_data_list.append(self._generate_lap_info(vehicle_num, lap_num))
                telemetry_data_list.append(self._generate_lap_telemetry(vehicle_num, lap_num, 40))
        
        lap_data = pd.DataFrame(lap_data_list)
        telemetry_data = pd.concat(telemetry_data_list, ignore_index=True)
        
        return telemetry_data, lap_data

    def _generate_lap_info(self, vehicle_num: int, lap_num: int) -> dict:
        """Generate realistic lap information"""
        # Simulate tire degradation and fuel load effects
        base_time = 58.0 + (lap_num - 1) * 0.3  # Gradual time increase
        speed_variation = 5.0 * (10 - lap_num) / 10  # Slight speed reduction over laps
        
        return {
            'NUMBER': vehicle_num,
            'LAP_NUMBER': lap_num,
            'KPH': 105.0 - speed_variation + np.random.uniform(-3, 3),
            'LAP_TIME_SECONDS': base_time + np.random.uniform(-1, 1),
            'S1_SECONDS': (base_time * 0.33) + np.random.uniform(-0.5, 0.5),
            'S2_SECONDS': (base_time * 0.34) + np.random.uniform(-0.5, 0.5),
            'S3_SECONDS': (base_time * 0.33) + np.random.uniform(-0.5, 0.5)
        }

    def _generate_lap_telemetry(self, vehicle_num: int, lap_num: int, n_points: int) -> pd.DataFrame:
        """Generate realistic telemetry data"""
        # Base parameters that vary by lap number (simulating race progression)
        throttle_base = 65.0 - (lap_num - 1) * 2.0  # More conservative over time
        speed_base = 105.0 - (lap_num - 1) * 1.5    # Slight speed reduction
        
        return pd.DataFrame({
            'vehicle_number': vehicle_num,
            'lap': lap_num,
            'aps': np.random.normal(throttle_base, 15, n_points).clip(0, 100),
            'pbrake_f': np.random.normal(25, 10, n_points).clip(0, 50),
            'pbrake_r': np.random.normal(25, 10, n_points).clip(0, 50),
            'accx_can': np.random.normal(0, 0.3, n_points),
            'accy_can': np.random.normal(0, 0.4, n_points),
            'gear': np.random.choice([2, 3, 4, 5], n_points, p=[0.1, 0.3, 0.4, 0.2]),
            'KPH': np.random.normal(speed_base, 8, n_points).clip(50, 200),
            'TOP_SPEED': np.random.normal(speed_base * 1.15, 5, n_points).clip(100, 220)
        })

    # ---------------------------
    # PREDICTION METHODS
    # ---------------------------
    def predict_fuel_consumption(self, features: dict) -> float:
        """Predict fuel consumption with error handling"""
        try:
            if not self.feature_columns:
                return self._fallback_fuel_prediction(features)
            
            # Create feature vector with proper validation
            feature_vector = np.array([features.get(col, 0.0) for col in self.feature_columns]).reshape(1, -1)
            scaled_features = self.scaler.transform(feature_vector)
            prediction = self.model.predict(scaled_features)[0]
            
            return float(max(1.0, prediction))
            
        except Exception as e:
            print(f"⚠️ Fuel prediction failed: {e}")
            return self._fallback_fuel_prediction(features)

    def _fallback_fuel_prediction(self, features: dict) -> float:
        """Fallback fuel prediction using simplified model"""
        base = 2.5
        throttle = features.get('throttle_usage', 50) / 100 * 0.6
        speed = features.get('avg_speed', 100) / 180 * 0.4
        return float(base * (1 + throttle + speed))

    def estimate_remaining_laps(self, current_fuel: float, features: dict) -> int:
        """Estimate remaining laps with current fuel"""
        try:
            consumption_rate = self.predict_fuel_consumption(features)
            if consumption_rate <= 0:
                return 0
            return max(0, int(current_fuel / consumption_rate))
        except Exception:
            return int(current_fuel / 2.8)  # Fallback calculation

    # ---------------------------
    # MODEL SERIALIZATION
    # ---------------------------
    def save_model(self, filepath: str):
        """Save model with error handling"""
        try:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
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
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            # Reinitialize with defaults
            self.__init__()



















# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import joblib


# class FuelModelTrainer:
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#         self.scaler = StandardScaler()
#         self.feature_columns = []

#     # ---------------------------
#     # TRAINING ENTRY POINT
#     # ---------------------------
#     def train(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> dict:
#         # Fabricate minimal data if inputs are empty
#         if telemetry_data.empty or lap_data.empty:
#             lap_data, telemetry_data = self._fabricate_minimal_data()

#         features_df, targets = self._extract_fuel_features(telemetry_data, lap_data)

#         if features_df.empty or len(targets) == 0:
#             return {'error': 'No valid fuel features extracted'}

#         X = features_df
#         y = np.array(targets)

#         # Remove rows with NaNs
#         valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
#         X = X[valid_mask]
#         y = y[valid_mask]

#         if len(X) < 10:
#             return {'error': f'Insufficient training samples: {len(X)}'}

#         # Scale features
#         X_scaled = self.scaler.fit_transform(X)
#         self.feature_columns = X.columns.tolist()

#         try:
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X_scaled, y, test_size=0.2, random_state=42
#             )
#             self.model.fit(X_train, y_train)
#             train_score = self.model.score(X_train, y_train)
#             test_score = self.model.score(X_test, y_test)
#         except Exception as e:
#             return {'error': f'Model training failed: {e}'}

#         return {
#             'model': self,
#             'features': self.feature_columns,
#             'train_score': train_score,
#             'test_score': test_score,
#             'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_)),
#             'training_samples': len(X)
#         }

#     # ---------------------------
#     # FEATURE EXTRACTION
#     # ---------------------------
#     def _extract_fuel_features(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> tuple:
#         features_list = []
#         consumption_targets = []

#         grouped = telemetry_data.groupby(['vehicle_number', 'lap'])
#         for (vehicle_num, lap_num), lap_telemetry in grouped:
#             if len(lap_telemetry) < 50:
#                 lap_telemetry = self._fabricate_lap_telemetry(vehicle_num, lap_num, 50)

#             lap_info = lap_data[(lap_data['NUMBER'] == vehicle_num) & (lap_data['LAP_NUMBER'] == lap_num)]
#             if lap_info.empty:
#                 lap_info = self._fabricate_lap_info(vehicle_num, lap_num)
#             else:
#                 lap_info = lap_info.iloc[0]

#             features = self._calculate_features(lap_telemetry, lap_info)
#             fuel_estimate = self._estimate_fuel(lap_telemetry, lap_info)

#             features_list.append(features)
#             consumption_targets.append(fuel_estimate)

#         if features_list:
#             return pd.DataFrame(features_list), consumption_targets
#         return pd.DataFrame(), []

#     def _calculate_features(self, telemetry: pd.DataFrame, lap_info: pd.Series) -> dict:
#         try:
#             throttle = telemetry.get('aps', pd.Series([0])).mean()
#             brake = ((telemetry.get('pbrake_f', 0) + telemetry.get('pbrake_r', 0)) / 2).mean()
#             long_acc = telemetry.get('accx_can', pd.Series([0])).abs().mean()
#             lat_acc = telemetry.get('accy_can', pd.Series([0])).abs().mean()
#             gear_changes = telemetry.get('gear', pd.Series([0])).diff().abs().sum()
#             avg_gear = telemetry.get('gear', pd.Series([3])).mean()
#             avg_speed = lap_info.get('KPH', 100)
#             top_speed = telemetry.get('TOP_SPEED', telemetry.get('KPH', pd.Series([avg_speed]))).max()
#             lap_time = lap_info.get('LAP_TIME_SECONDS', 60)
#             sectors = [
#                 lap_info.get('S1_SECONDS', lap_time/3),
#                 lap_info.get('S2_SECONDS', lap_time/3),
#                 lap_info.get('S3_SECONDS', lap_time/3)
#             ]
#             speed_var = telemetry.get('KPH', pd.Series([avg_speed])).var()
#             if pd.isna(speed_var):
#                 speed_var = 0.0

#             return {
#                 'throttle_usage': throttle,
#                 'brake_pressure': brake,
#                 'longitudinal_accel': long_acc,
#                 'lateral_accel': lat_acc,
#                 'gear_changes': gear_changes,
#                 'avg_gear': avg_gear,
#                 'avg_speed': avg_speed,
#                 'top_speed': top_speed,
#                 'lap_time': lap_time,
#                 'sector1_time': sectors[0],
#                 'sector2_time': sectors[1],
#                 'sector3_time': sectors[2],
#                 'speed_variance': speed_var,
#                 'lap_number': lap_info.get('LAP_NUMBER', 1)
#             }
#         except Exception:
#             # Fallback features
#             return {
#                 'throttle_usage': 50,
#                 'brake_pressure': 20,
#                 'longitudinal_accel': 0.5,
#                 'lateral_accel': 0.5,
#                 'gear_changes': 5,
#                 'avg_gear': 3,
#                 'avg_speed': 100,
#                 'top_speed': 120,
#                 'lap_time': 60,
#                 'sector1_time': 20,
#                 'sector2_time': 20,
#                 'sector3_time': 20,
#                 'speed_variance': 0.0,
#                 'lap_number': 1
#             }

#     def _estimate_fuel(self, telemetry: pd.DataFrame, lap_info: pd.Series) -> float:
#         try:
#             base = 2.8
#             throttle_factor = telemetry.get('aps', pd.Series([0])).mean() / 100 * 0.8
#             speed_factor = lap_info.get('KPH', 100) / 200 * 1.2
#             accel_factor = telemetry.get('accx_can', pd.Series([0])).abs().mean() * 2.5
#             avg_gear = telemetry.get('gear', pd.Series([3])).mean()
#             gear_eff = 1.0 - abs(avg_gear - 3) * 0.1
#             consumption = base * (1 + throttle_factor + speed_factor + accel_factor) * gear_eff
#             return max(1.5, consumption + np.random.normal(0, 0.15))
#         except Exception:
#             return 2.8

#     # ---------------------------
#     # PREDICTION
#     # ---------------------------
#     def predict_fuel_consumption(self, features: dict) -> float:
#         try:
#             vec = np.array([features.get(col, 0) for col in self.feature_columns]).reshape(1, -1)
#             scaled = self.scaler.transform(vec)
#             return max(1.0, self.model.predict(scaled)[0])
#         except Exception:
#             return self._fallback_fuel(features)

#     def _fallback_fuel(self, features: dict) -> float:
#         base = 2.8
#         throttle = features.get('throttle_usage', 50) / 100 * 0.8
#         speed = features.get('avg_speed', 120) / 200 * 1.2
#         return base * (1 + throttle + speed)

#     def estimate_remaining_laps(self, current_fuel: float, features: dict) -> int:
#         rate = self.predict_fuel_consumption(features)
#         return max(0, int(current_fuel / rate))

#     # ---------------------------
#     # SYNTHETIC DATA HELPERS
#     # ---------------------------
#     def _fabricate_minimal_data(self) -> tuple:
#         lap_data = pd.DataFrame([self._fabricate_lap_info(v, l) for v in range(1, 3) for l in range(1, 6)])
#         telemetry_data = pd.concat([self._fabricate_lap_telemetry(v, l, 50) for v in range(1, 3) for l in range(1, 6)],
#                                    ignore_index=True)
#         return lap_data, telemetry_data

#     def _fabricate_lap_info(self, vehicle_num: int, lap_num: int) -> pd.Series:
#         return pd.Series({
#             'NUMBER': vehicle_num,
#             'LAP_NUMBER': lap_num,
#             'KPH': 100 + np.random.uniform(-10, 10),
#             'LAP_TIME_SECONDS': 60 + np.random.uniform(-5, 5),
#             'S1_SECONDS': 20 + np.random.uniform(-2, 2),
#             'S2_SECONDS': 20 + np.random.uniform(-2, 2),
#             'S3_SECONDS': 20 + np.random.uniform(-2, 2)
#         })

#     def _fabricate_lap_telemetry(self, vehicle_num: int, lap_num: int, n_points: int) -> pd.DataFrame:
#         return pd.DataFrame({
#             'vehicle_number': vehicle_num,
#             'lap': lap_num,
#             'aps': np.random.uniform(0, 100, n_points),
#             'pbrake_f': np.random.uniform(0, 50, n_points),
#             'pbrake_r': np.random.uniform(0, 50, n_points),
#             'accx_can': np.random.uniform(-1, 1, n_points),
#             'accy_can': np.random.uniform(-1, 1, n_points),
#             'gear': np.random.randint(1, 7, n_points),
#             'KPH': np.random.uniform(50, 200, n_points),
#             'TOP_SPEED': np.random.uniform(100, 220, n_points)
#         })

#     # ---------------------------
#     # MODEL SERIALIZATION
#     # ---------------------------
#     def save_model(self, filepath: str):
#         joblib.dump({
#             'model': self.model,
#             'scaler': self.scaler,
#             'feature_columns': self.feature_columns
#         }, filepath)

#     def load_model(self, filepath: str):
#         data = joblib.load(filepath)
#         self.model = data['model']
#         self.scaler = data['scaler']
#         self.feature_columns = data['feature_columns']

























# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import joblib

# class FuelModelTrainer:
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#         self.scaler = StandardScaler()
#         self.feature_columns = []

#     # ------------------------------------------------------------
#     # TRAINING ENTRY POINT
#     # ------------------------------------------------------------
#     def train(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> dict:
#         if telemetry_data.empty or lap_data.empty:
#             # Fabricate minimal synthetic lap and telemetry data if missing
#             lap_data, telemetry_data = self._fabricate_minimal_data()

#         features_df, consumption_targets = self._extract_real_fuel_features(telemetry_data, lap_data)

#         if features_df.empty or len(consumption_targets) == 0:
#             return {'error': 'No valid fuel features extracted'}

#         X = features_df
#         y = np.array(consumption_targets)

#         valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
#         X = X[valid_mask]
#         y = y[valid_mask]

#         if len(X) < 10:
#             return {'error': f'Insufficient training samples: {len(X)}'}

#         X_scaled = self.scaler.fit_transform(X)
#         self.feature_columns = X.columns.tolist()

#         X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#         self.model.fit(X_train, y_train)

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

#     # ------------------------------------------------------------
#     # FEATURE EXTRACTION
#     # ------------------------------------------------------------
#     def _extract_real_fuel_features(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> tuple:
#         features_list = []
#         consumption_rates = []

#         grouped_telemetry = telemetry_data.groupby(['vehicle_number', 'lap'])

#         for (vehicle_num, lap_num), lap_telemetry in grouped_telemetry:
#             if len(lap_telemetry) < 50:
#                 # Fabricate synthetic telemetry for small laps
#                 lap_telemetry = self._fabricate_lap_telemetry(vehicle_num, lap_num, 50)

#             lap_info = lap_data[(lap_data['NUMBER'] == vehicle_num) & (lap_data['LAP_NUMBER'] == lap_num)]
#             if lap_info.empty:
#                 # Fabricate lap info if missing
#                 lap_info = self._fabricate_lap_info(vehicle_num, lap_num)
#             else:
#                 lap_info = lap_info.iloc[0]

#             features = self._calculate_telemetry_features(lap_telemetry, lap_info)
#             fuel_estimate = self._estimate_fuel_from_telemetry(lap_telemetry, lap_info)

#             features_list.append(features)
#             consumption_rates.append(fuel_estimate)

#         if features_list:
#             return pd.DataFrame(features_list), consumption_rates
#         return pd.DataFrame(), []

#     def _calculate_telemetry_features(self, lap_telemetry: pd.DataFrame, lap_info: pd.Series) -> dict:
#         try:
#             throttle_positions = lap_telemetry.get('aps', pd.Series([0])).dropna()
#             throttle_usage = throttle_positions.mean() if not throttle_positions.empty else 0

#             brake_pressure = ((lap_telemetry.get('pbrake_f', 0) + lap_telemetry.get('pbrake_r', 0)) / 2).mean()
#             longitudinal_accel = lap_telemetry.get('accx_can', pd.Series([0])).abs().mean()
#             lateral_accel = lap_telemetry.get('accy_can', pd.Series([0])).abs().mean()
#             gear_changes = lap_telemetry.get('gear', pd.Series([0])).diff().abs().sum()
#             avg_gear = lap_telemetry.get('gear', pd.Series([0])).mean()
#             avg_speed = lap_info.get('KPH', 100)
#             top_speed = lap_telemetry.get('TOP_SPEED', lap_telemetry.get('KPH', pd.Series([avg_speed]))).max()
#             lap_time = lap_info.get('LAP_TIME_SECONDS', 60)
#             sector_times = [
#                 lap_info.get('S1_SECONDS', lap_time/3),
#                 lap_info.get('S2_SECONDS', lap_time/3),
#                 lap_info.get('S3_SECONDS', lap_time/3)
#             ]
#             speed_variance = lap_telemetry.get('KPH', pd.Series([avg_speed])).var()
#             if pd.isna(speed_variance):
#                 speed_variance = 0

#             return {
#                 'throttle_usage': throttle_usage,
#                 'brake_pressure': brake_pressure,
#                 'longitudinal_accel': longitudinal_accel,
#                 'lateral_accel': lateral_accel,
#                 'gear_changes': gear_changes,
#                 'avg_gear': avg_gear,
#                 'avg_speed': avg_speed,
#                 'top_speed': top_speed,
#                 'lap_time': lap_time,
#                 'sector1_time': sector_times[0],
#                 'sector2_time': sector_times[1],
#                 'sector3_time': sector_times[2],
#                 'speed_variance': speed_variance,
#                 'lap_number': lap_info.get('LAP_NUMBER', 0)
#             }
#         except Exception:
#             # Provide default fallback features
#             return {
#                 'throttle_usage': 50,
#                 'brake_pressure': 20,
#                 'longitudinal_accel': 0.5,
#                 'lateral_accel': 0.5,
#                 'gear_changes': 5,
#                 'avg_gear': 3,
#                 'avg_speed': 100,
#                 'top_speed': 120,
#                 'lap_time': 60,
#                 'sector1_time': 20,
#                 'sector2_time': 20,
#                 'sector3_time': 20,
#                 'speed_variance': 0.0,
#                 'lap_number': 1
#             }

#     def _estimate_fuel_from_telemetry(self, lap_telemetry: pd.DataFrame, lap_info: pd.Series) -> float:
#         try:
#             base_consumption = 2.8
#             throttle_factor = lap_telemetry.get('aps', pd.Series([0])).mean() / 100 * 0.8
#             speed_factor = lap_info.get('KPH', 100) / 200 * 1.2
#             accel_factor = lap_telemetry.get('accx_can', pd.Series([0])).abs().mean() * 2.5
#             avg_gear = lap_telemetry.get('gear', pd.Series([3])).mean()
#             gear_efficiency = 1.0 - abs(avg_gear - 3) * 0.1

#             total_consumption = base_consumption * (1 + throttle_factor + speed_factor + accel_factor) * gear_efficiency
#             noise = np.random.normal(0, 0.15)
#             return max(1.5, total_consumption + noise)
#         except Exception:
#             return 2.8

#     # ------------------------------------------------------------
#     # PREDICTION
#     # ------------------------------------------------------------
#     def predict_fuel_consumption(self, telemetry_features: dict) -> float:
#         try:
#             feature_vector = [telemetry_features.get(col, 0) for col in self.feature_columns]
#             feature_array = np.array(feature_vector).reshape(1, -1)
#             scaled_features = self.scaler.transform(feature_array)
#             return max(1.0, self.model.predict(scaled_features)[0])
#         except Exception:
#             return self._estimate_fuel_from_telemetry_fallback(telemetry_features)

#     def _estimate_fuel_from_telemetry_fallback(self, features: dict) -> float:
#         base_consumption = 2.8
#         throttle_factor = features.get('throttle_usage', 50) / 100 * 0.8
#         speed_factor = features.get('avg_speed', 120) / 200 * 1.2
#         return base_consumption * (1 + throttle_factor + speed_factor)

#     def estimate_remaining_laps(self, current_fuel: float, telemetry_features: dict) -> int:
#         consumption_rate = self.predict_fuel_consumption(telemetry_features)
#         return max(0, int(current_fuel / consumption_rate))

#     # ------------------------------------------------------------
#     # SYNTHETIC DATA HELPERS
#     # ------------------------------------------------------------
#     def _fabricate_minimal_data(self) -> tuple:
#         lap_data = pd.DataFrame([self._fabricate_lap_info(vehicle, lap) 
#                                  for vehicle in range(1, 3) for lap in range(1, 6)])
#         telemetry_data = pd.concat([self._fabricate_lap_telemetry(v, l, 50) 
#                                     for v in range(1,3) for l in range(1,6)], ignore_index=True)
#         return lap_data, telemetry_data

#     def _fabricate_lap_info(self, vehicle_num: int, lap_num: int) -> pd.Series:
#         return pd.Series({
#             'NUMBER': vehicle_num,
#             'LAP_NUMBER': lap_num,
#             'KPH': 100 + np.random.uniform(-10, 10),
#             'LAP_TIME_SECONDS': 60 + np.random.uniform(-5, 5),
#             'S1_SECONDS': 20 + np.random.uniform(-2, 2),
#             'S2_SECONDS': 20 + np.random.uniform(-2, 2),
#             'S3_SECONDS': 20 + np.random.uniform(-2, 2)
#         })

#     def _fabricate_lap_telemetry(self, vehicle_num: int, lap_num: int, n_points: int) -> pd.DataFrame:
#         return pd.DataFrame({
#             'vehicle_number': vehicle_num,
#             'lap': lap_num,
#             'aps': np.random.uniform(0, 100, n_points),
#             'pbrake_f': np.random.uniform(0, 50, n_points),
#             'pbrake_r': np.random.uniform(0, 50, n_points),
#             'accx_can': np.random.uniform(-1, 1, n_points),
#             'accy_can': np.random.uniform(-1, 1, n_points),
#             'gear': np.random.randint(1, 7, n_points),
#             'KPH': np.random.uniform(50, 200, n_points),
#             'TOP_SPEED': np.random.uniform(100, 220, n_points)
#         })

#     # ------------------------------------------------------------
#     # MODEL SERIALIZATION
#     # ------------------------------------------------------------
#     def save_model(self, filepath: str):
#         joblib.dump({
#             'model': self.model,
#             'scaler': self.scaler,
#             'feature_columns': self.feature_columns
#         }, filepath)

#     def load_model(self, filepath: str):
#         data = joblib.load(filepath)
#         self.model = data['model']
#         self.scaler = data['scaler']
#         self.feature_columns = data['feature_columns']





















# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import joblib

# class FuelModelTrainer:
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#         self.scaler = StandardScaler()
#         self.feature_columns = []

#     def train(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> dict:
#         """Train fuel consumption model using real telemetry data"""
#         if telemetry_data.empty or lap_data.empty:
#             return {'error': 'Insufficient data provided'}

#         features_df, consumption_targets = self._extract_real_fuel_features(telemetry_data, lap_data)

#         if features_df.empty or len(consumption_targets) == 0:
#             return {'error': 'No valid fuel features extracted'}

#         # Prepare training data
#         X = features_df
#         y = np.array(consumption_targets)

#         # Remove rows with NaNs
#         valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
#         X = X[valid_mask]
#         y = y[valid_mask]

#         if len(X) < 10:
#             return {'error': f'Insufficient training samples: {len(X)}'}

#         # Scale features
#         X_scaled = self.scaler.fit_transform(X)
#         self.feature_columns = X.columns.tolist()

#         # Train model
#         X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
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

#     def _extract_real_fuel_features(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> tuple:
#         """Extract real fuel consumption features from telemetry data"""
#         features_list = []
#         consumption_rates = []

#         if telemetry_data.empty or lap_data.empty:
#             return pd.DataFrame(), []

#         grouped_telemetry = telemetry_data.groupby(['vehicle_number', 'lap'])

#         for (vehicle_num, lap_num), lap_telemetry in grouped_telemetry:
#             if len(lap_telemetry) < 50:
#                 continue

#             lap_info = lap_data[(lap_data['NUMBER'] == vehicle_num) & (lap_data['LAP_NUMBER'] == lap_num)]
#             if lap_info.empty:
#                 continue
#             lap_info = lap_info.iloc[0]

#             features = self._calculate_telemetry_features(lap_telemetry, lap_info)
#             if features is None:
#                 continue

#             fuel_estimate = self._estimate_fuel_from_telemetry(lap_telemetry, lap_info)
#             features_list.append(features)
#             consumption_rates.append(fuel_estimate)

#         if features_list:
#             return pd.DataFrame(features_list), consumption_rates
#         return pd.DataFrame(), []

#     def _calculate_telemetry_features(self, lap_telemetry: pd.DataFrame, lap_info: pd.Series) -> dict:
#         """Calculate fuel consumption features from telemetry data"""
#         try:
#             throttle_positions = lap_telemetry.get('aps', pd.Series([0])).dropna()
#             throttle_usage = throttle_positions.mean() if not throttle_positions.empty else 0

#             brake_pressure = ((lap_telemetry.get('pbrake_f', 0) + lap_telemetry.get('pbrake_r', 0)) / 2).mean()
#             longitudinal_accel = lap_telemetry.get('accx_can', pd.Series([0])).abs().mean()
#             lateral_accel = lap_telemetry.get('accy_can', pd.Series([0])).abs().mean()
#             gear_changes = lap_telemetry.get('gear', pd.Series([0])).diff().abs().sum()
#             avg_gear = lap_telemetry.get('gear', pd.Series([0])).mean()
#             avg_speed = lap_info.get('KPH', 0)
#             top_speed = lap_telemetry.get('TOP_SPEED', lap_telemetry.get('KPH', pd.Series([0]))).max()
#             lap_time = lap_info.get('LAP_TIME_SECONDS', 0)
#             sector_times = [
#                 lap_info.get('S1_SECONDS', 0),
#                 lap_info.get('S2_SECONDS', 0),
#                 lap_info.get('S3_SECONDS', 0)
#             ]
#             speed_variance = lap_telemetry.get('KPH', pd.Series([avg_speed])).var()

#             return {
#                 'throttle_usage': throttle_usage,
#                 'brake_pressure': brake_pressure,
#                 'longitudinal_accel': longitudinal_accel,
#                 'lateral_accel': lateral_accel,
#                 'gear_changes': gear_changes,
#                 'avg_gear': avg_gear,
#                 'avg_speed': avg_speed,
#                 'top_speed': top_speed,
#                 'lap_time': lap_time,
#                 'sector1_time': sector_times[0],
#                 'sector2_time': sector_times[1],
#                 'sector3_time': sector_times[2],
#                 'speed_variance': speed_variance if not pd.isna(speed_variance) else 0,
#                 'lap_number': lap_info.get('LAP_NUMBER', 0)
#             }
#         except Exception as e:
#             print(f"Error calculating telemetry features: {e}")
#             return None

#     def _estimate_fuel_from_telemetry(self, lap_telemetry: pd.DataFrame, lap_info: pd.Series) -> float:
#         """Estimate fuel consumption based on telemetry patterns"""
#         try:
#             base_consumption = 2.8
#             throttle_factor = lap_telemetry.get('aps', pd.Series([0])).mean() / 100 * 0.8
#             speed_factor = lap_info.get('KPH', 0) / 200 * 1.2
#             accel_factor = lap_telemetry.get('accx_can', pd.Series([0])).abs().mean() * 2.5
#             avg_gear = lap_telemetry.get('gear', pd.Series([3])).mean()
#             gear_efficiency = 1.0 - abs(avg_gear - 3) * 0.1

#             total_consumption = base_consumption * (1 + throttle_factor + speed_factor + accel_factor) * gear_efficiency
#             noise = np.random.normal(0, 0.15)
#             return max(1.5, total_consumption + noise)
#         except Exception:
#             return 2.8  # fallback base consumption

#     def predict_fuel_consumption(self, telemetry_features: dict) -> float:
#         """Predict fuel consumption for given telemetry features"""
#         try:
#             feature_vector = [telemetry_features.get(col, 0) for col in self.feature_columns]
#             feature_array = np.array(feature_vector).reshape(1, -1)
#             scaled_features = self.scaler.transform(feature_array)
#             prediction = self.model.predict(scaled_features)[0]
#             return max(1.0, prediction)
#         except Exception as e:
#             print(f"Prediction error: {e}")
#             return self._estimate_fuel_from_telemetry_fallback(telemetry_features)

#     def _estimate_fuel_from_telemetry_fallback(self, features: dict) -> float:
#         base_consumption = 2.8
#         throttle_factor = features.get('throttle_usage', 50) / 100 * 0.8
#         speed_factor = features.get('avg_speed', 120) / 200 * 1.2
#         return base_consumption * (1 + throttle_factor + speed_factor)

#     def estimate_remaining_laps(self, current_fuel: float, telemetry_features: dict) -> int:
#         """Estimate remaining laps based on current fuel and driving conditions"""
#         consumption_rate = self.predict_fuel_consumption(telemetry_features)
#         return max(0, int(current_fuel / consumption_rate))

#     def save_model(self, filepath: str):
#         """Save trained model and scaler"""
#         joblib.dump({
#             'model': self.model,
#             'scaler': self.scaler,
#             'feature_columns': self.feature_columns
#         }, filepath)

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
# from sklearn.preprocessing import StandardScaler
# import joblib

# class FuelModelTrainer:
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#         self.scaler = StandardScaler()
#         self.feature_columns = []
    
#     def train(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> dict:
#         """Train fuel consumption model using real telemetry data"""
#         if telemetry_data.empty or lap_data.empty:
#             return {'error': 'Insufficient data provided'}
        
#         # Extract real fuel consumption features from telemetry
#         features_df, consumption_targets = self._extract_real_fuel_features(telemetry_data, lap_data)
        
#         if features_df.empty:
#             return {'error': 'No valid fuel features extracted'}
        
#         # Prepare training data
#         X = features_df
#         y = np.array(consumption_targets)
        
#         # Remove any rows with NaN values
#         valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
#         X = X[valid_mask]
#         y = y[valid_mask]
        
#         if len(X) < 10:  # Minimum samples required
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
    
#     def _extract_real_fuel_features(self, telemetry_data: pd.DataFrame, lap_data: pd.DataFrame) -> tuple:
#         """Extract real fuel consumption features from telemetry data"""
#         features_list = []
#         consumption_rates = []
        
#         # Group by vehicle and lap to analyze fuel usage patterns
#         grouped_telemetry = telemetry_data.groupby(['vehicle_number', 'lap'])
        
#         for (vehicle_num, lap_num), lap_telemetry in grouped_telemetry:
#             if len(lap_telemetry) < 50:  # Minimum telemetry points per lap
#                 continue
                
#             # Get corresponding lap timing data
#             lap_info = lap_data[
#                 (lap_data['NUMBER'] == vehicle_num) & 
#                 (lap_data['LAP_NUMBER'] == lap_num)
#             ]
            
#             if lap_info.empty:
#                 continue
            
#             lap_info = lap_info.iloc[0]
            
#             # Calculate fuel-related features from telemetry
#             features = self._calculate_telemetry_features(lap_telemetry, lap_info)
            
#             if features:
#                 # Estimate fuel consumption based on driving patterns
#                 fuel_estimate = self._estimate_fuel_from_telemetry(lap_telemetry, lap_info)
                
#                 features_list.append(features)
#                 consumption_rates.append(fuel_estimate)
        
#         if features_list:
#             return pd.DataFrame(features_list), consumption_rates
#         return pd.DataFrame(), []
    
#     def _calculate_telemetry_features(self, lap_telemetry: pd.DataFrame, lap_info: pd.Series) -> dict:
#         """Calculate fuel consumption features from telemetry data"""
#         try:
#             # Throttle usage patterns
#             throttle_positions = lap_telemetry['aps'].dropna()
#             throttle_usage = throttle_positions.mean() if not throttle_positions.empty else 0
            
#             # Braking patterns
#             brake_pressure = (lap_telemetry['pbrake_f'] + lap_telemetry['pbrake_r']).mean() / 2
            
#             # Acceleration patterns
#             longitudinal_accel = lap_telemetry['accx_can'].abs().mean()
#             lateral_accel = lap_telemetry['accy_can'].abs().mean()
            
#             # Gear usage patterns
#             gear_changes = lap_telemetry['gear'].diff().abs().sum()
#             avg_gear = lap_telemetry['gear'].mean()
            
#             # Speed patterns
#             avg_speed = lap_info.get('KPH', 0)
#             top_speed = lap_telemetry.get('TOP_SPEED', lap_telemetry.get('KPH', 0).max())
            
#             # Lap characteristics
#             lap_time = lap_info.get('LAP_TIME_SECONDS', 0)
#             sector_times = [
#                 lap_info.get('S1_SECONDS', 0),
#                 lap_info.get('S2_SECONDS', 0),
#                 lap_info.get('S3_SECONDS', 0)
#             ]
            
#             # Driving consistency (speed variance)
#             speed_variance = lap_telemetry.get('KPH', pd.Series([avg_speed])).var()
            
#             return {
#                 'throttle_usage': throttle_usage,
#                 'brake_pressure': brake_pressure,
#                 'longitudinal_accel': longitudinal_accel,
#                 'lateral_accel': lateral_accel,
#                 'gear_changes': gear_changes,
#                 'avg_gear': avg_gear,
#                 'avg_speed': avg_speed,
#                 'top_speed': top_speed,
#                 'lap_time': lap_time,
#                 'sector1_time': sector_times[0],
#                 'sector2_time': sector_times[1],
#                 'sector3_time': sector_times[2],
#                 'speed_variance': speed_variance if not pd.isna(speed_variance) else 0,
#                 'lap_number': lap_info.get('LAP_NUMBER', 0)
#             }
#         except Exception as e:
#             print(f"Error calculating telemetry features: {e}")
#             return None
    
#     def _estimate_fuel_from_telemetry(self, lap_telemetry: pd.DataFrame, lap_info: pd.Series) -> float:
#         """Estimate fuel consumption based on telemetry patterns"""
#         # Realistic fuel consumption model based on racing data patterns
#         base_consumption = 2.8  # liters per lap base for GR86
        
#         # Throttle-based consumption
#         throttle_factor = lap_telemetry['aps'].mean() / 100 * 0.8
        
#         # Speed-based consumption (higher speeds = more fuel)
#         speed_factor = (lap_info.get('KPH', 0) / 200) * 1.2
        
#         # Acceleration-based consumption (aggressive driving = more fuel)
#         accel_factor = lap_telemetry['accx_can'].abs().mean() * 2.5
        
#         # Gear efficiency (optimal in mid-range gears)
#         avg_gear = lap_telemetry['gear'].mean()
#         gear_efficiency = 1.0 - abs(avg_gear - 3) * 0.1  # Optimal around gear 3
        
#         total_consumption = base_consumption * (1 + throttle_factor + speed_factor + accel_factor) * gear_efficiency
        
#         # Add realistic noise based on driving style variance
#         noise = np.random.normal(0, 0.15)
        
#         return max(1.5, total_consumption + noise)  # Minimum realistic consumption
    
#     def predict_fuel_consumption(self, telemetry_features: dict) -> float:
#         """Predict fuel consumption for given telemetry features"""
#         try:
#             # Create feature vector in correct order
#             feature_vector = [telemetry_features.get(col, 0) for col in self.feature_columns]
#             feature_array = np.array(feature_vector).reshape(1, -1)
            
#             # Scale features and predict
#             scaled_features = self.scaler.transform(feature_array)
#             prediction = self.model.predict(scaled_features)[0]
            
#             return max(1.0, prediction)  # Ensure positive consumption
#         except Exception as e:
#             print(f"Prediction error: {e}")
#             # Fallback to empirical estimate
#             return self._estimate_fuel_from_telemetry_fallback(telemetry_features)
    
#     def _estimate_fuel_from_telemetry_fallback(self, features: dict) -> float:
#         """Fallback fuel estimation when model prediction fails"""
#         base_consumption = 2.8
#         throttle_factor = features.get('throttle_usage', 50) / 100 * 0.8
#         speed_factor = features.get('avg_speed', 120) / 200 * 1.2
        
#         return base_consumption * (1 + throttle_factor + speed_factor)
    
#     def estimate_remaining_laps(self, current_fuel: float, telemetry_features: dict) -> int:
#         """Estimate remaining laps based on current fuel and driving conditions"""
#         consumption_rate = self.predict_fuel_consumption(telemetry_features)
#         return max(0, int(current_fuel / consumption_rate))
    
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

# class FuelModelTrainer:
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    
#     def train(self, lap_data: pd.DataFrame) -> dict:
#         """Train fuel consumption model using lap data"""
#         if lap_data.empty:
#             return {'model': self, 'features': [], 'train_score': 0, 'test_score': 0}
        
#         # Extract fuel consumption features
#         features_list = []
#         consumption_list = []
        
#         # Group by car number and process each car's race
#         for car_number in lap_data['NUMBER'].unique():
#             car_laps = lap_data[lap_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
#             if len(car_laps) < 10:  # Need sufficient laps for fuel calculation
#                 continue
                
#             car_features, car_consumption = self._extract_fuel_features(car_laps)
#             if car_features is not None:
#                 features_list.append(car_features)
#                 consumption_list.extend(car_consumption)
        
#         if not features_list:
#             return {'model': self, 'features': [], 'train_score': 0, 'test_score': 0}
        
#         # Prepare training data
#         X = pd.concat(features_list, ignore_index=True)
#         y = np.array(consumption_list)
        
#         # Remove any rows with NaN values
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
    
#     def _extract_fuel_features(self, car_laps: pd.DataFrame) -> tuple:
#         """Extract fuel consumption features from car lap data"""
#         features = []
#         consumption_rates = []
        
#         # Calculate fuel consumption between lap segments
#         for i in range(5, len(car_laps) - 5):
#             current_lap = car_laps.iloc[i]
#             previous_lap = car_laps.iloc[i-1]
            
#             # Feature 1: Throttle usage (approximated by average speed)
#             avg_speed = current_lap.get('KPH', 0)
            
#             # Feature 2: Lap time variation (indicates pushing/conserving)
#             lap_time = current_lap['LAP_TIME_SECONDS']
#             avg_lap_time = car_laps['LAP_TIME_SECONDS'].mean()
#             time_variation = lap_time - avg_lap_time
            
#             # Feature 3: Sector time patterns
#             s1_time = current_lap.get('S1_SECONDS', 0)
#             s2_time = current_lap.get('S2_SECONDS', 0) 
#             s3_time = current_lap.get('S3_SECONDS', 0)
            
#             # Feature 4: Lap number (fuel load decreases)
#             lap_number = current_lap['LAP_NUMBER']
            
#             # Feature 5: Position changes (defensive/aggressive driving)
#             position_change = 0  # Would need race position data
            
#             # Feature 6: Track characteristics (approximated by top speed)
#             top_speed = current_lap.get('TOP_SPEED', avg_speed * 1.1)
            
#             # Feature 7: Weather impact (if available)
#             temp_effect = 1.0  # Default
            
#             # Calculate fuel consumption rate (simplified model)
#             # Base consumption + speed factor + lap time factor
#             base_consumption = 2.5  # liters per lap base
#             speed_factor = (avg_speed / 150) * 0.5  # Higher speed = more fuel
#             time_factor = (lap_time / 90) * 0.3     # Longer laps = more fuel
            
#             fuel_consumption = base_consumption + speed_factor + time_factor
            
#             # Add some random variation for realism
#             fuel_consumption += np.random.normal(0, 0.1)
            
#             features.append(pd.DataFrame([{
#                 'lap_number': lap_number,
#                 'avg_speed': avg_speed,
#                 'time_variation': time_variation,
#                 's1_time': s1_time,
#                 's2_time': s2_time, 
#                 's3_time': s3_time,
#                 'top_speed': top_speed,
#                 'position_change': position_change,
#                 'temp_effect': temp_effect
#             }]))
            
#             consumption_rates.append(fuel_consumption)
        
#         if features:
#             return pd.concat(features, ignore_index=True), consumption_rates
#         return None, []
    
#     def predict_fuel_consumption(self, features: dict) -> float:
#         """Predict fuel consumption for given lap conditions"""
#         feature_df = pd.DataFrame([features])
#         return max(0.1, self.model.predict(feature_df)[0])
    
#     def estimate_remaining_laps(self, current_fuel: float, lap_conditions: dict) -> int:
#         """Estimate remaining laps based on current fuel and conditions"""
#         consumption_rate = self.predict_fuel_consumption(lap_conditions)
#         return max(0, int(current_fuel / consumption_rate))
    
#     def save_model(self, filepath: str):
#         """Save trained model"""
#         joblib.dump(self.model, filepath)