from models.tire_trainer import TireModelTrainer
from models.fuel_trainer import FuelModelTrainer
from models.pit_strategy_trainer import PitStrategyTrainer
from models.weather_trainer import WeatherModelTrainer
from data.firebase_loader import FirebaseDataLoader
from data.preprocessor import DataPreprocessor
from data.feature_engineer import FeatureEngineer
import pandas as pd
import logging
import os
from typing import Dict, Any


class TrainingOrchestrator:
    def __init__(self, storage):
        self.storage = storage
        self.logger = logging.getLogger(__name__)
        self.models_output_dir = "outputs/models"
        os.makedirs(self.models_output_dir, exist_ok=True)

    def train_all_models(self) -> dict:
        """Orchestrate training of all models with proper data integration"""
        self.logger.info("🚀 Starting comprehensive model training pipeline...")

        # Load available tracks dynamically
        available_tracks = self.storage.list_available_tracks()
        if not available_tracks:
            self.logger.error("❌ No tracks found in storage")
            return {}

        self.logger.info(f"📥 Loading data for {len(available_tracks)} tracks: {available_tracks}")

        # Load data from Firebase
        all_data = self.storage.load_all_tracks(available_tracks)

        # Preprocess and engineer features
        preprocessor = DataPreprocessor()
        feature_engineer = FeatureEngineer()

        processed_data = {}
        for track, data in all_data.items():
            self.logger.info(f"🔄 Processing {track}...")

            processed_track_data = {
                'lap_data': preprocessor.preprocess_lap_data(data.get('lap_data', pd.DataFrame())),
                'race_data': preprocessor.preprocess_race_data(data.get('race_data', pd.DataFrame())),
                'weather_data': preprocessor.preprocess_weather_data(data.get('weather_data', pd.DataFrame())),
                'telemetry_data': preprocessor.preprocess_telemetry_data(data.get('telemetry_data', pd.DataFrame()))
            }

            # Engineer features safely
            enhanced_data = feature_engineer.create_composite_features({track: processed_track_data})
            processed_data[track] = enhanced_data.get(track, processed_track_data)

            self._log_data_quality(track, processed_data[track])

        # Train models
        self.logger.info("🏃 Training models with integrated data...")
        models = {}

        # Tire Model
        tire_trainer = TireModelTrainer()
        tire_data = self._prepare_tire_training_data(processed_data)
        if tire_data:
            try:
                models['tire_degradation'] = tire_trainer.train(
                    tire_data['lap_data'],
                    tire_data['telemetry_data'],
                    tire_data['weather_data']
                )
            except Exception as e:
                self.logger.error(f"❌ Tire model training failed: {e}")
        else:
            self.logger.warning("⚠️ Insufficient data for tire model training")

        # Fuel Model
        fuel_trainer = FuelModelTrainer()
        fuel_data = self._prepare_fuel_training_data(processed_data)
        if fuel_data:
            try:
                models['fuel_consumption'] = fuel_trainer.train(
                    fuel_data['lap_data'],
                    fuel_data['telemetry_data']
                )
            except Exception as e:
                self.logger.error(f"❌ Fuel model training failed: {e}")
        else:
            self.logger.warning("⚠️ Insufficient data for fuel model training")

        # Pit Strategy Model
        pit_trainer = PitStrategyTrainer()
        if len(processed_data) >= 2:
            try:
                pit_result = pit_trainer.train(processed_data)
                if 'error' not in pit_result:
                    models['pit_strategy'] = pit_result
                else:
                    self.logger.warning(f"⚠️ Pit strategy training skipped: {pit_result['error']}")
            except Exception as e:
                self.logger.error(f"❌ Pit strategy model training failed: {e}")
        else:
            self.logger.warning("⚠️ Insufficient tracks for pit strategy model")

        # Weather Model
        weather_trainer = WeatherModelTrainer()
        weather_data = self._prepare_weather_training_data(processed_data)
        if weather_data:
            try:
                models['weather_impact'] = weather_trainer.train(weather_data)
            except Exception as e:
                self.logger.error(f"❌ Weather model training failed: {e}")
        else:
            self.logger.warning("⚠️ Insufficient data for weather model training")

        # Save models
        self.logger.info("💾 Saving trained models...")
        self._save_models(models)

        self.logger.info(f"✅ Training completed: {len(models)} models processed")
        return models

    # -------------------------------
    # Data preparation helpers
    # -------------------------------
    def _prepare_tire_training_data(self, processed_data: Dict) -> Dict[str, pd.DataFrame]:
        lap_list, telemetry_list, weather_list = [], [], []
        for track, data in processed_data.items():
            lap_data = data.get('lap_data', pd.DataFrame())
            telemetry = data.get('telemetry_data', pd.DataFrame())
            weather = data.get('weather_data', pd.DataFrame())
            if not lap_data.empty and len(lap_data) >= 20 and not telemetry.empty and len(telemetry) >= 100:
                lap_copy = lap_data.copy()
                lap_copy['TRACK'] = track
                lap_list.append(lap_copy)
                telemetry_list.append(telemetry)
                weather_list.append(weather)
        if not lap_list:
            return {}
        return {
            'lap_data': pd.concat(lap_list, ignore_index=True),
            'telemetry_data': pd.concat(telemetry_list, ignore_index=True),
            'weather_data': pd.concat(weather_list, ignore_index=True)
        }

    def _prepare_fuel_training_data(self, processed_data: Dict) -> Dict[str, pd.DataFrame]:
        lap_list, telemetry_list = [], []
        for track, data in processed_data.items():
            lap_data = data.get('lap_data', pd.DataFrame())
            telemetry = data.get('telemetry_data', pd.DataFrame())
            if not lap_data.empty and len(lap_data) >= 15 and not telemetry.empty and 'THROTTLE_POSITION' in telemetry.columns:
                lap_copy = lap_data.copy()
                lap_copy['TRACK'] = track
                lap_list.append(lap_copy)
                telemetry_list.append(telemetry)
        if not lap_list:
            return {}
        return {
            'lap_data': pd.concat(lap_list, ignore_index=True),
            'telemetry_data': pd.concat(telemetry_list, ignore_index=True)
        }

    def _prepare_weather_training_data(self, processed_data: Dict) -> Dict:
        valid_tracks = {}
        for track, data in processed_data.items():
            lap_data = data.get('lap_data', pd.DataFrame())
            weather_data = data.get('weather_data', pd.DataFrame())
            if not lap_data.empty and len(lap_data) >= 10 and not weather_data.empty and len(weather_data) >= 5:
                valid_tracks[track] = data
        return valid_tracks if len(valid_tracks) >= 2 else {}

    # -------------------------------
    # Model saving and logging
    # -------------------------------
    def _save_models(self, models: Dict[str, Any]) -> Dict[str, str]:
        saved = {}
        for name, result in models.items():
            try:
                if isinstance(result, dict) and 'model' in result and hasattr(result['model'], 'save_model'):
                    filepath = os.path.join(self.models_output_dir, f"{name}_model.pkl")
                    result['model'].save_model(filepath)
                    saved[name] = filepath
                    self.logger.info(f"💾 Saved {name} model to {filepath}")
            except Exception as e:
                self.logger.error(f"❌ Failed to save {name} model: {e}")
        return saved

    def _log_data_quality(self, track: str, data: Dict[str, pd.DataFrame]):
        report = {}
        for dt, df in data.items():
            if df.empty:
                report[dt] = "EMPTY"
            else:
                report[dt] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'null_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                }
        self.logger.info(f"📊 {track} data quality: {report}")

    # -------------------------------
    # Validation helper
    # -------------------------------
    def validate_training_results(self, models: Dict) -> Dict[str, Any]:
        results = {}
        for name, result in models.items():
            if isinstance(result, dict) and 'error' in result:
                results[name] = {'status': 'FAILED', 'error': result['error']}
            elif isinstance(result, dict) and 'accuracy' in result:
                acc = result['accuracy']
                status = 'GOOD' if acc > 0.8 else 'FAIR' if acc > 0.6 else 'POOR'
                results[name] = {'status': status, 'accuracy': acc, 'training_samples': result.get('training_samples', 0)}
            elif isinstance(result, dict) and 'test_score' in result:
                score = result['test_score']
                status = 'GOOD' if score > 0.7 else 'FAIR' if score > 0.5 else 'POOR'
                results[name] = {'status': status, 'test_score': score, 'training_samples': result.get('training_samples', 0)}
            else:
                results[name] = {'status': 'UNKNOWN', 'result': result}
        return results



























# from models.tire_trainer import TireModelTrainer
# from models.fuel_trainer import FuelModelTrainer
# from models.pit_strategy_trainer import PitStrategyTrainer
# from models.weather_trainer import WeatherModelTrainer
# from data.firebase_loader import FirebaseDataLoader
# from data.preprocessor import DataPreprocessor
# from data.feature_engineer import FeatureEngineer
# import pandas as pd
# import logging
# import os
# from typing import Dict, Any

# class TrainingOrchestrator:
#     def __init__(self, storage):
#         self.storage = storage
#         self.logger = logging.getLogger(__name__)
#         self.models_output_dir = "outputs/models"
#         os.makedirs(self.models_output_dir, exist_ok=True)
    
#     def train_all_models(self) -> dict:
#         """Orchestrate training of all models with proper data integration"""
#         self.logger.info("🚀 Starting comprehensive model training pipeline...")
        
#         # Load available tracks dynamically
#         available_tracks = self.storage.list_available_tracks()
#         if not available_tracks:
#             self.logger.error("❌ No tracks found in storage")
#             return {}
        
#         self.logger.info(f"📥 Loading data for {len(available_tracks)} tracks: {available_tracks}")
        
#         # Load data from Firebase with telemetry support
#         all_data = self.storage.load_all_tracks(available_tracks)
        
#         # Preprocess and engineer features for each track
#         preprocessor = DataPreprocessor()
#         feature_engineer = FeatureEngineer()
        
#         processed_data = {}
#         for track, data in all_data.items():
#             self.logger.info(f"🔄 Processing {track}...")
            
#             # Preprocess all data types
#             processed_track_data = {
#                 'lap_data': preprocessor.preprocess_lap_data(data['lap_data']),
#                 'race_data': preprocessor.preprocess_race_data(data['race_data']),
#                 'weather_data': preprocessor.preprocess_weather_data(data['weather_data']),
#                 'telemetry_data': preprocessor.preprocess_telemetry_data(data['telemetry_data'])
#             }
            
#             # Engineer advanced features
#             enhanced_data = feature_engineer.create_composite_features({track: processed_track_data})
#             processed_data[track] = enhanced_data[track]
            
#             # Log data quality
#             self._log_data_quality(track, processed_data[track])
        
#         # Train models with appropriate data integration
#         self.logger.info("🏃 Training models with integrated data...")
        
#         models = {}
        
#         # Train Tire Model (requires lap data + telemetry)
#         tire_trainer = TireModelTrainer()
#         valid_tire_data = self._prepare_tire_training_data(processed_data)
#         if valid_tire_data:
#             models['tire_degradation'] = tire_trainer.train(
#                 valid_tire_data['lap_data'], 
#                 valid_tire_data['telemetry_data'],
#                 valid_tire_data['weather_data']
#             )
#         else:
#             self.logger.warning("⚠️ Insufficient data for tire model training")
        
#         # Train Fuel Model (requires lap data + telemetry)
#         fuel_trainer = FuelModelTrainer()
#         valid_fuel_data = self._prepare_fuel_training_data(processed_data)
#         if valid_fuel_data:
#             models['fuel_consumption'] = fuel_trainer.train(
#                 valid_fuel_data['lap_data'],
#                 valid_fuel_data['telemetry_data']
#             )
#         else:
#             self.logger.warning("⚠️ Insufficient data for fuel model training")
        
#         # Train Pit Strategy Model (requires processed multi-track data)
#         pit_trainer = PitStrategyTrainer()
#         if len(processed_data) >= 2:  # Need multiple tracks for strategy patterns
#             models['pit_strategy'] = pit_trainer.train(processed_data)
#         else:
#             self.logger.warning("⚠️ Insufficient tracks for pit strategy model")
        
#         # Train Weather Impact Model
#         weather_trainer = WeatherModelTrainer()
#         valid_weather_data = self._prepare_weather_training_data(processed_data)
#         if valid_weather_data:
#             models['weather_impact'] = weather_trainer.train(valid_weather_data)
#         else:
#             self.logger.warning("⚠️ Insufficient data for weather model training")
        
#         # Save models and log results
#         self.logger.info("💾 Saving trained models...")
#         successful_models = self._save_models(models)
        
#         self.logger.info(f"✅ Training completed: {len(successful_models)}/{len(models)} models trained successfully")
#         return models
    
#     def _prepare_tire_training_data(self, processed_data: Dict) -> Dict[str, pd.DataFrame]:
#         """Prepare combined data for tire model training"""
#         lap_data_list = []
#         telemetry_data_list = []
#         weather_data_list = []
        
#         for track, data in processed_data.items():
#             lap_data = data['lap_data']
#             telemetry_data = data['telemetry_data']
#             weather_data = data['weather_data']
            
#             # Only include tracks with sufficient lap and telemetry data
#             if (not lap_data.empty and len(lap_data) >= 20 and 
#                 not telemetry_data.empty and len(telemetry_data) >= 100):
                
#                 # Add track identifier
#                 lap_data = lap_data.copy()
#                 lap_data['TRACK'] = track
                
#                 lap_data_list.append(lap_data)
#                 telemetry_data_list.append(telemetry_data)
#                 weather_data_list.append(weather_data)
        
#         if not lap_data_list:
#             return {}
        
#         return {
#             'lap_data': pd.concat(lap_data_list, ignore_index=True),
#             'telemetry_data': pd.concat(telemetry_data_list, ignore_index=True),
#             'weather_data': pd.concat(weather_data_list, ignore_index=True)
#         }
    
#     def _prepare_fuel_training_data(self, processed_data: Dict) -> Dict[str, pd.DataFrame]:
#         """Prepare combined data for fuel model training"""
#         lap_data_list = []
#         telemetry_data_list = []
        
#         for track, data in processed_data.items():
#             lap_data = data['lap_data']
#             telemetry_data = data['telemetry_data']
            
#             # Filter for tracks with throttle/brake telemetry
#             if (not lap_data.empty and not telemetry_data.empty and 
#                 'THROTTLE_POSITION' in telemetry_data.columns and
#                 len(lap_data) >= 15):
                
#                 lap_data = lap_data.copy()
#                 lap_data['TRACK'] = track
                
#                 lap_data_list.append(lap_data)
#                 telemetry_data_list.append(telemetry_data)
        
#         if not lap_data_list:
#             return {}
        
#         return {
#             'lap_data': pd.concat(lap_data_list, ignore_index=True),
#             'telemetry_data': pd.concat(telemetry_data_list, ignore_index=True)
#         }
    
#     def _prepare_weather_training_data(self, processed_data: Dict) -> Dict:
#         """Prepare data for weather model training"""
#         valid_tracks = {}
        
#         for track, data in processed_data.items():
#             # Only include tracks with both weather and lap data
#             if (not data['weather_data'].empty and not data['lap_data'].empty and
#                 len(data['lap_data']) >= 10 and len(data['weather_data']) >= 5):
#                 valid_tracks[track] = data
        
#         return valid_tracks if len(valid_tracks) >= 2 else {}  # Need multiple tracks for patterns
    
#     def _save_models(self, models: Dict[str, Any]) -> Dict[str, str]:
#         """Save trained models and return success status"""
#         successful_saves = {}
        
#         for name, result in models.items():
#             try:
#                 if 'model' in result and hasattr(result['model'], 'save_model'):
#                     filepath = os.path.join(self.models_output_dir, f"{name}_model.pkl")
#                     result['model'].save_model(filepath)
#                     successful_saves[name] = filepath
#                     self.logger.info(f"💾 Saved {name} model to {filepath}")
#                 else:
#                     self.logger.warning(f"⚠️ Model {name} cannot be saved - invalid result structure")
#             except Exception as e:
#                 self.logger.error(f"❌ Failed to save {name} model: {e}")
        
#         return successful_saves
    
#     def _log_data_quality(self, track: str, data: Dict[str, pd.DataFrame]):
#         """Log data quality metrics for each track"""
#         quality_report = {}
        
#         for data_type, df in data.items():
#             if df.empty:
#                 quality_report[data_type] = "EMPTY"
#             else:
#                 quality_report[data_type] = {
#                     'rows': len(df),
#                     'columns': len(df.columns),
#                     'null_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
#                 }
        
#         self.logger.info(f"📊 {track} data quality: {quality_report}")
    
#     def validate_training_results(self, models: Dict) -> Dict[str, Any]:
#         """Validate model training results and performance"""
#         validation_results = {}
        
#         for name, result in models.items():
#             if 'error' in result:
#                 validation_results[name] = {'status': 'FAILED', 'error': result['error']}
#             elif 'test_score' in result:
#                 score = result['test_score']
#                 status = 'GOOD' if score > 0.7 else 'FAIR' if score > 0.5 else 'POOR'
#                 validation_results[name] = {
#                     'status': status,
#                     'test_score': score,
#                     'training_samples': result.get('training_samples', 0)
#                 }
#             elif 'accuracy' in result:
#                 accuracy = result['accuracy']
#                 status = 'GOOD' if accuracy > 0.8 else 'FAIR' if accuracy > 0.6 else 'POOR'
#                 validation_results[name] = {
#                     'status': status,
#                     'accuracy': accuracy,
#                     'training_samples': result.get('training_samples', 0)
#                 }
#             else:
#                 validation_results[name] = {'status': 'UNKNOWN', 'result': result}
        
#         return validation_results

















# from models.tire_trainer import TireModelTrainer
# from models.fuel_trainer import FuelModelTrainer
# from models.pit_strategy_trainer import PitStrategyTrainer
# from data.firebase_loader import FirebaseDataLoader
# from data.preprocessor import DataPreprocessor
# import pandas as pd

# import logging

# class TrainingOrchestrator:
#     def __init__(self, storage):
#         self.storage = storage
#         self.logger = logging.getLogger(__name__)
    
#     def train_all_models(self) -> dict:
#         """Orchestrate training of all models"""
#         self.logger.info("📥 Loading training data...")
        
#         # Load data from Firebase
#         tracks = [
#             'barber-motorsports-park',
#             'circuit-of-the-americas', 
#             'indianapolis',
#             'road-america',
#             'sebring', 
#             'sonoma',
#             'virginia-international-raceway'
#         ]
#         all_data = self.storage.load_all_tracks(tracks)
        
#         # Preprocess data
#         preprocessor = DataPreprocessor()
#         processed_data = {}
        
#         for track, data in all_data.items():
#             processed_data[track] = {
#                 'lap_data': preprocessor.preprocess_lap_data(data['lap_data']),
#                 'race_data': data['race_data'],
#                 'weather_data': data['weather_data']
#             }
        
#         # Train models
#         self.logger.info("🏃 Training models...")
        
#         tire_trainer = TireModelTrainer()
#         fuel_trainer = FuelModelTrainer()
#         pit_trainer = PitStrategyTrainer()
        
#         # Combine data from all tracks
#         combined_lap_data = pd.concat([data['lap_data'] for data in processed_data.values()])
        
#         models = {
#             'tire_degradation': tire_trainer.train(combined_lap_data),
#             'fuel_consumption': fuel_trainer.train(combined_lap_data),
#             'pit_strategy': pit_trainer.train(processed_data)
#         }
        
#         # Save models
#         self.logger.info("💾 Saving models...")
#         for name, result in models.items():
#             result['model'].save_model(f"outputs/models/{name}.pkl")
        
#         self.logger.info(f"✅ Trained {len(models)} models successfully")
#         return models