import os
import logging
import joblib
import pandas as pd
from typing import Dict, Any

from models.tire_trainer import TireModelTrainer
from models.fuel_trainer import FuelModelTrainer
from models.pit_strategy_trainer import PitStrategyTrainer
from models.weather_trainer import WeatherModelTrainer
from data.preprocessor import DataPreprocessor
from data.feature_engineer import FeatureEngineer


class TrainingOrchestrator:
    def __init__(self, storage):
        self.storage = storage
        self.logger = logging.getLogger(__name__)
        self.models_output_dir = "outputs/models"
        self.training_state_dir = "outputs/training_state"
        os.makedirs(self.models_output_dir, exist_ok=True)
        os.makedirs(self.training_state_dir, exist_ok=True)

    # -------------------------------
    # MAIN TRAINING PIPELINE
    # -------------------------------
    def train_all_models(self) -> dict:
        self.logger.info("🚀 Starting robust model training pipeline...")

        state = self._load_training_state()
        if state.get("completed", False):
            self.logger.info("✅ Training already completed. Loading models...")
            return self._load_existing_models()

        available_tracks = self.storage.list_available_tracks()
        if not available_tracks:
            self.logger.error("❌ No tracks available in storage")
            return {}

        processed_tracks = state.get("processed_tracks", [])
        remaining_tracks = [t for t in available_tracks if t not in processed_tracks]

        if not remaining_tracks:
            self.logger.info("📊 All tracks processed. Finalizing models...")
            return self._finalize_training()

        self.logger.info(f"📥 Processing {len(remaining_tracks)} tracks: {remaining_tracks}")
        all_processed_data = self._load_processed_data_state()

        for idx, track in enumerate(remaining_tracks, 1):
            try:
                self.logger.info(f"🔄 Processing track {idx}/{len(available_tracks)}: {track}")
                track_data = self._process_single_track(track)
                if track_data:
                    all_processed_data[track] = track_data
                    processed_tracks.append(track)
                    self._update_training_state({
                        "processed_tracks": processed_tracks,
                        "processed_data_keys": list(all_processed_data.keys())
                    })
                    self._save_track_data(track, track_data)
                    self.logger.info(f"✅ Track processed: {track}")
                else:
                    self.logger.warning(f"⚠️ Skipped {track}: insufficient data")
            except Exception as e:
                self.logger.error(f"❌ Failed to process {track}: {e}")
                continue

        if all_processed_data:
            models = self._train_models_incrementally(all_processed_data, state.get("models", {}))
            self._upload_models_to_firebase(models)
            self._update_training_state({
                "processed_tracks": processed_tracks,
                "processed_data_keys": list(all_processed_data.keys()),
                "models": list(models.keys()),
                "completed": len(processed_tracks) == len(available_tracks)
            })
            return models
        else:
            self.logger.error("❌ No valid data processed for training")
            return {}

    # -------------------------------
    # SINGLE TRACK PROCESSING
    # -------------------------------
    def _process_single_track(self, track: str) -> Dict[str, pd.DataFrame]:
        cache_file = os.path.join(self.training_state_dir, f"{track}_processed.pkl")
        if os.path.exists(cache_file):
            self.logger.info(f"📂 Loading cached processed data for {track}")
            try:
                return joblib.load(cache_file)
            except Exception:
                self.logger.warning(f"⚠️ Cached file corrupted. Reprocessing {track}")

        track_raw = self.storage.load_track_data(track)
        preprocessor = DataPreprocessor()
        feature_engineer = FeatureEngineer()

        processed = {
            "lap_data": preprocessor.preprocess_lap_data(track_raw.get("lap_data", pd.DataFrame())),
            "race_data": preprocessor.preprocess_race_data(track_raw.get("race_data", pd.DataFrame())),
            "weather_data": preprocessor.preprocess_weather_data(track_raw.get("weather_data", pd.DataFrame())),
            "telemetry_data": preprocessor.preprocess_telemetry_data(track_raw.get("telemetry_data", pd.DataFrame())),
        }

        processed = feature_engineer.create_composite_features({track: processed}).get(track, processed)
        self._log_data_quality(track, processed)

        try:
            joblib.dump(processed, cache_file)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to cache processed data for {track}: {e}")
        return processed

    # -------------------------------
    # INCREMENTAL MODEL TRAINING
    # -------------------------------
    def _train_models_incrementally(self, processed_data: Dict, existing_models: Dict = None) -> Dict[str, Any]:
        self.logger.info("🏃 Training models incrementally...")
        models = existing_models or {}

        # Tire Model
        tire_data = self._prepare_tire_training_data(processed_data)
        if tire_data:
            try:
                tire_trainer = TireModelTrainer()
                result = tire_trainer.train(
                    tire_data["lap_data"], tire_data["telemetry_data"], tire_data["weather_data"]
                )
                models["tire_degradation"] = result
            except Exception as e:
                self.logger.error(f"❌ Tire model training failed: {e}")
                models["tire_degradation"] = {"error": str(e)}

        # Fuel Model
        fuel_data = self._prepare_fuel_training_data(processed_data)
        if fuel_data:
            try:
                fuel_trainer = FuelModelTrainer()
                result = fuel_trainer.train(fuel_data["lap_data"], fuel_data["telemetry_data"])
                models["fuel_consumption"] = result
            except Exception as e:
                self.logger.error(f"❌ Fuel model training failed: {e}")
                models["fuel_consumption"] = {"error": str(e)}

        # Pit Strategy Model
        if len(processed_data) >= 2:
            try:
                pit_trainer = PitStrategyTrainer()
                result = pit_trainer.train(processed_data)
                models["pit_strategy"] = result if "error" not in result else {"error": "Pit strategy training failed"}
            except Exception as e:
                self.logger.error(f"❌ Pit strategy model training failed: {e}")
                models["pit_strategy"] = {"error": str(e)}

        # Weather Model
        weather_data = self._prepare_weather_training_data(processed_data)
        if weather_data:
            try:
                weather_trainer = WeatherModelTrainer()
                result = weather_trainer.train(weather_data)
                models["weather_impact"] = result
            except Exception as e:
                self.logger.error(f"❌ Weather model training failed: {e}")
                models["weather_impact"] = {"error": str(e)}

        self._save_models(models)
        self.logger.info(f"✅ Incremental training completed: {len(models)} models processed")
        return models

    # -------------------------------
    # DATA PREPARATION
    # -------------------------------
    def _prepare_tire_training_data(self, processed_data: Dict) -> Dict[str, pd.DataFrame]:
        laps, telemetry, weather = [], [], []
        for track, data in processed_data.items():
            lap, telem, weath = data.get("lap_data", pd.DataFrame()), data.get("telemetry_data", pd.DataFrame()), data.get("weather_data", pd.DataFrame())
            if len(lap) >= 3 and len(telem) >= 10:
                laps.append(lap.assign(TRACK=track))
                telemetry.append(telem)
                weather.append(weath)
        if not laps: return {}
        return {"lap_data": pd.concat(laps, ignore_index=True),
                "telemetry_data": pd.concat(telemetry, ignore_index=True),
                "weather_data": pd.concat(weather, ignore_index=True)}

    def _prepare_fuel_training_data(self, processed_data: Dict) -> Dict[str, pd.DataFrame]:
        laps, telemetry = [], []
        for track, data in processed_data.items():
            lap, telem = data.get("lap_data", pd.DataFrame()), data.get("telemetry_data", pd.DataFrame())
            if len(lap) >= 2 and not telem.empty:
                laps.append(lap.assign(TRACK=track))
                telemetry.append(telem)
        if not laps: return {}
        return {"lap_data": pd.concat(laps, ignore_index=True),
                "telemetry_data": pd.concat(telemetry, ignore_index=True)}

    def _prepare_weather_training_data(self, processed_data: Dict) -> Dict:
        valid = {}
        for track, data in processed_data.items():
            lap, weather = data.get("lap_data", pd.DataFrame()), data.get("weather_data", pd.DataFrame())
            if len(lap) >= 3 and len(weather) >= 2:
                valid[track] = data
        return valid if len(valid) >= 2 else {}

    # -------------------------------
    # MODEL & STATE MANAGEMENT
    # -------------------------------
    def _load_training_state(self) -> Dict:
        state_file = os.path.join(self.training_state_dir, "training_state.pkl")
        if os.path.exists(state_file):
            try: return joblib.load(state_file)
            except: pass
        return {"processed_tracks": [], "models": {}}

    def _update_training_state(self, updates: Dict):
        state_file = os.path.join(self.training_state_dir, "training_state.pkl")
        state = self._load_training_state()
        state.update(updates)
        try: joblib.dump(state, state_file)
        except Exception: pass

    def _load_processed_data_state(self) -> Dict:
        processed = {}
        for track in self._load_training_state().get("processed_data_keys", []):
            cache_file = os.path.join(self.training_state_dir, f"{track}_processed.pkl")
            if os.path.exists(cache_file):
                try: processed[track] = joblib.load(cache_file)
                except: continue
        return processed

    def _save_track_data(self, track: str, data: Dict):
        try:
            joblib.dump(data, os.path.join(self.training_state_dir, f"{track}_processed.pkl"))
        except Exception: pass

    def _save_models(self, models: Dict[str, Any]) -> Dict[str, str]:
        saved = {}
        for name, result in models.items():
            try:
                filepath = os.path.join(self.models_output_dir, f"{name}_model.pkl")
                if isinstance(result, dict) and "model" in result and hasattr(result["model"], "save_model"):
                    result["model"].save_model(filepath)
                else:
                    joblib.dump(result, filepath)
                saved[name] = filepath
            except Exception as e:
                self.logger.error(f"❌ Failed to save {name} model: {e}")
        return saved

    def _load_existing_models(self) -> Dict:
        models = {}
        for file in os.listdir(self.models_output_dir):
            if file.endswith(".pkl"):
                name = file.replace("_model.pkl", "")
                try: models[name] = joblib.load(os.path.join(self.models_output_dir, file))
                except: continue
        return models

    def _finalize_training(self) -> Dict:
        models = self._load_existing_models()
        self._update_training_state({"completed": True})
        return models

    def _upload_models_to_firebase(self, models: Dict[str, Any]):
        try:
            if hasattr(self.storage, "upload_models_to_firebase"):
                success = self.storage.upload_models_to_firebase()
                if success: self.logger.info("🚀 Models uploaded to Firebase Storage")
        except Exception as e:
            self.logger.error(f"❌ Firebase upload failed: {e}")

    # -------------------------------
    # LOGGING & VALIDATION
    # -------------------------------
    def _log_data_quality(self, track: str, data: Dict[str, pd.DataFrame]):
        report = {}
        for k, df in data.items():
            if df.empty:
                report[k] = "EMPTY"
            else:
                report[k] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "null_pct": (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
                }
        self.logger.info(f"📊 {track} data quality: {report}")

    def validate_training_results(self, models: Dict) -> Dict[str, Any]:
        results = {}
        for name, result in models.items():
            if isinstance(result, dict) and "error" in result:
                results[name] = {"status": "FAILED", "error": result["error"]}
            elif isinstance(result, dict) and "accuracy" in result:
                acc = result["accuracy"]
                status = "GOOD" if acc > 0.8 else "FAIR" if acc > 0.6 else "POOR"
                results[name] = {"status": status, "accuracy": acc, "training_samples": result.get("training_samples", 0)}
            elif isinstance(result, dict) and "test_score" in result:
                score = result["test_score"]
                status = "GOOD" if score > 0.7 else "FAIR" if score > 0.5 else "POOR"
                results[name] = {"status": status, "test_score": score, "training_samples": result.get("training_samples", 0)}
            else:
                results[name] = {"status": "UNKNOWN", "result": result}
        return results



























# import os
# import logging
# import joblib
# import pandas as pd
# import numpy as np
# from typing import Dict, Any, List, Optional

# from models.tire_trainer import TireModelTrainer
# from models.fuel_trainer import FuelModelTrainer
# from models.pit_strategy_trainer import PitStrategyTrainer
# from models.weather_trainer import WeatherModelTrainer
# from data.preprocessor import DataPreprocessor


# class TrainingOrchestrator:
#     def __init__(self, storage):
#         self.storage = storage
#         self.logger = logging.getLogger(__name__)
#         self.models_output_dir = "outputs/models"
#         self.training_state_dir = "outputs/training_state"
        
#         # Create directories with error handling
#         try:
#             os.makedirs(self.models_output_dir, exist_ok=True)
#             os.makedirs(self.training_state_dir, exist_ok=True)
#         except Exception as e:
#             self.logger.error(f"❌ Failed to create output directories: {e}")
#             raise

#     # -------------------------------
#     # MAIN TRAINING PIPELINE
#     # -------------------------------
#     def train_all_models(self) -> dict:
#         """Main training pipeline with comprehensive error handling"""
#         try:
#             self.logger.info("🚀 Starting robust model training pipeline...")

#             # Load training state
#             state = self._load_training_state()
#             if state.get("completed", False):
#                 self.logger.info("✅ Training already completed. Loading existing models...")
#                 return self._load_existing_models()

#             # Get available tracks
#             available_tracks = self._get_available_tracks_with_fallback()
#             if not available_tracks:
#                 self.logger.error("❌ No tracks available for training")
#                 return self._generate_fallback_models()

#             self.logger.info(f"📁 Found {len(available_tracks)} tracks: {available_tracks}")

#             # Process tracks incrementally
#             processed_data = self._process_all_tracks(available_tracks, state)
#             if not processed_data:
#                 self.logger.error("❌ No valid data processed from any track")
#                 return self._generate_fallback_models()

#             # Train models with validated data
#             models = self._train_models_with_validation(processed_data, state.get("models", {}))
            
#             # Finalize training
#             if models:
#                 self._finalize_training_success(available_tracks, models)
#                 self.logger.info(f"🎉 Training completed successfully! Generated {len(models)} models")
#             else:
#                 self.logger.warning("⚠️ No models were trained successfully")
#                 models = self._generate_fallback_models()

#             return models

#         except Exception as e:
#             self.logger.error(f"❌ Training pipeline failed: {e}")
#             return self._generate_fallback_models()

#     def _get_available_tracks_with_fallback(self) -> List[str]:
#         """Get available tracks with fallback to default list"""
#         try:
#             tracks = self.storage.list_available_tracks()
#             if tracks:
#                 return tracks
#             else:
#                 self.logger.warning("📝 No tracks from storage, using default tracks")
#                 return [
#                     'sonoma', 'indianapolis', 'road-america', 'circuit-of-the-americas',
#                     'sebring', 'virginia-international-raceway', 'barber-motorsports-park'
#                 ]
#         except Exception as e:
#             self.logger.error(f"❌ Failed to get tracks: {e}")
#             return ['sonoma', 'indianapolis']  # Minimal fallback

#     def _process_all_tracks(self, available_tracks: List[str], state: Dict) -> Dict[str, Dict]:
#         """Process all tracks with incremental progress tracking"""
#         processed_data = self._load_processed_data_state()
#         processed_tracks = state.get("processed_tracks", [])
#         remaining_tracks = [t for t in available_tracks if t not in processed_tracks]

#         if not remaining_tracks:
#             self.logger.info("📊 All tracks already processed")
#             return processed_data

#         self.logger.info(f"📥 Processing {len(remaining_tracks)} new tracks: {remaining_tracks}")

#         successful_processing = 0
#         for idx, track in enumerate(remaining_tracks, 1):
#             try:
#                 self.logger.info(f"🔄 [{idx}/{len(remaining_tracks)}] Processing: {track}")
                
#                 track_data = self._process_single_track_robustly(track)
#                 if self._validate_track_data(track_data):
#                     processed_data[track] = track_data
#                     processed_tracks.append(track)
#                     successful_processing += 1
                    
#                     # Update state after each successful track
#                     self._update_training_state({
#                         "processed_tracks": processed_tracks,
#                         "processed_data_keys": list(processed_data.keys())
#                     })
                    
#                     self.logger.info(f"✅ Successfully processed {track}")
#                 else:
#                     self.logger.warning(f"⚠️ Insufficient data in {track}, skipping")

#             except Exception as e:
#                 self.logger.error(f"❌ Failed to process {track}: {e}")
#                 continue

#         self.logger.info(f"📊 Processed {successful_processing}/{len(remaining_tracks)} new tracks successfully")
#         return processed_data

#     def _process_single_track_robustly(self, track: str) -> Dict[str, pd.DataFrame]:
#         """Process single track with comprehensive error handling and caching"""
#         cache_file = os.path.join(self.training_state_dir, f"{track}_processed.pkl")
        
#         # Try to load from cache first
#         if os.path.exists(cache_file):
#             try:
#                 self.logger.info(f"📂 Loading cached data for {track}")
#                 cached_data = joblib.load(cache_file)
#                 if self._validate_track_data(cached_data):
#                     return cached_data
#                 else:
#                     self.logger.warning(f"⚠️ Cached data for {track} is invalid, reprocessing")
#             except Exception as e:
#                 self.logger.warning(f"⚠️ Failed to load cache for {track}: {e}")

#         # Process from raw data
#         try:
#             self.logger.info(f"📥 Loading raw data for {track}")
#             track_raw = self.storage.load_track_data(track)
            
#             if not track_raw:
#                 self.logger.warning(f"⚠️ No raw data returned for {track}")
#                 return self._generate_empty_track_data()

#             # Initialize preprocessor
#             preprocessor = DataPreprocessor(debug=False)
            
#             # Process each data type with individual error handling
#             processed = {}
#             data_types = ['lap_data', 'race_data', 'weather_data', 'telemetry_data']
            
#             for data_type in data_types:
#                 try:
#                     raw_df = track_raw.get(data_type, pd.DataFrame())
#                     if data_type == 'lap_data':
#                         processed[data_type] = preprocessor.preprocess_lap_data(raw_df, track)
#                     elif data_type == 'race_data':
#                         processed[data_type] = preprocessor.preprocess_race_data(raw_df, track)
#                     elif data_type == 'weather_data':
#                         processed[data_type] = preprocessor.preprocess_weather_data(raw_df, track)
#                     elif data_type == 'telemetry_data':
#                         processed[data_type] = preprocessor.preprocess_telemetry_data(raw_df, track)
                    
#                     self.logger.debug(f"  ✅ Processed {data_type}: {len(processed[data_type])} rows")
                    
#                 except Exception as e:
#                     self.logger.warning(f"  ⚠️ Failed to process {data_type} for {track}: {e}")
#                     processed[data_type] = pd.DataFrame()

#             # Log data quality
#             self._log_data_quality(track, processed)

#             # Cache processed data
#             try:
#                 joblib.dump(processed, cache_file)
#                 self.logger.debug(f"💾 Cached processed data for {track}")
#             except Exception as e:
#                 self.logger.warning(f"⚠️ Failed to cache {track}: {e}")

#             return processed

#         except Exception as e:
#             self.logger.error(f"❌ Critical error processing {track}: {e}")
#             return self._generate_empty_track_data()

#     def _validate_track_data(self, track_data: Dict[str, pd.DataFrame]) -> bool:
#         """Validate that track data has sufficient quality for training"""
#         if not track_data or not isinstance(track_data, dict):
#             return False
        
#         # Check if we have at least one substantial data type
#         min_rows_required = 3
#         valid_data_types = 0
        
#         for data_type, df in track_data.items():
#             if isinstance(df, pd.DataFrame) and len(df) >= min_rows_required:
#                 valid_data_types += 1
        
#         # Require at least lap data and one other data type
#         has_lap_data = (isinstance(track_data.get('lap_data'), pd.DataFrame) and 
#                        len(track_data['lap_data']) >= min_rows_required)
        
#         return has_lap_data and valid_data_types >= 2

#     # -------------------------------
#     # MODEL TRAINING WITH VALIDATION
#     # -------------------------------
#     def _train_models_with_validation(self, processed_data: Dict, existing_models: Dict = None) -> Dict[str, Any]:
#         """Train models with comprehensive data validation"""
#         self.logger.info("🏃 Starting model training with validation...")
#         models = existing_models or {}

#         # Prepare and validate data for each model type
#         training_datasets = self._prepare_all_training_datasets(processed_data)
        
#         # Train each model type with validation
#         model_trainers = {
#             'tire_degradation': (TireModelTrainer, self._train_tire_model),
#             'fuel_consumption': (FuelModelTrainer, self._train_fuel_model),
#             'pit_strategy': (PitStrategyTrainer, self._train_pit_strategy_model),
#             'weather_impact': (WeatherModelTrainer, self._train_weather_model)
#         }

#         for model_name, (trainer_class, train_function) in model_trainers.items():
#             try:
#                 if model_name in models and 'error' not in models.get(model_name, {}):
#                     self.logger.info(f"📝 Model {model_name} already exists, skipping")
#                     continue
                    
#                 self.logger.info(f"🔧 Training {model_name} model...")
#                 result = train_function(trainer_class, training_datasets, processed_data)
#                 models[model_name] = result
                
#                 if 'error' in result:
#                     self.logger.error(f"❌ {model_name} training failed: {result['error']}")
#                 else:
#                     self.logger.info(f"✅ {model_name} model trained successfully")
                    
#             except Exception as e:
#                 error_msg = f"Critical error training {model_name}: {str(e)}"
#                 self.logger.error(f"❌ {error_msg}")
#                 models[model_name] = {'error': error_msg}

#         # Save successful models
#         successful_models = {k: v for k, v in models.items() if 'error' not in v}
#         if successful_models:
#             self._save_models(successful_models)
            
#         return models

#     def _prepare_all_training_datasets(self, processed_data: Dict) -> Dict[str, Dict]:
#         """Prepare and validate training datasets for all model types"""
#         datasets = {}
        
#         # Tire model data
#         tire_data = self._prepare_tire_training_data(processed_data)
#         if self._validate_tire_data(tire_data):
#             datasets['tire'] = tire_data
#         else:
#             self.logger.warning("⚠️ Insufficient data for tire model")
        
#         # Fuel model data
#         fuel_data = self._prepare_fuel_training_data(processed_data)
#         if self._validate_fuel_data(fuel_data):
#             datasets['fuel'] = fuel_data
#         else:
#             self.logger.warning("⚠️ Insufficient data for fuel model")
        
#         # Weather model data
#         weather_data = self._prepare_weather_training_data(processed_data)
#         if self._validate_weather_data(weather_data):
#             datasets['weather'] = weather_data
#         else:
#             self.logger.warning("⚠️ Insufficient data for weather model")
            
#         return datasets

#     def _train_tire_model(self, trainer_class, datasets: Dict, processed_data: Dict) -> Dict:
#         """Train tire degradation model with validation"""
#         if 'tire' not in datasets:
#             return {'error': 'Insufficient data for tire model training'}
        
#         try:
#             trainer = trainer_class()
#             tire_data = datasets['tire']
#             result = trainer.train(
#                 tire_data["lap_data"], 
#                 tire_data["telemetry_data"], 
#                 tire_data["weather_data"]
#             )
#             return result
#         except Exception as e:
#             return {'error': f'Tire model training failed: {str(e)}'}

#     def _train_fuel_model(self, trainer_class, datasets: Dict, processed_data: Dict) -> Dict:
#         """Train fuel consumption model with validation"""
#         if 'fuel' not in datasets:
#             return {'error': 'Insufficient data for fuel model training'}
        
#         try:
#             trainer = trainer_class()
#             fuel_data = datasets['fuel']
#             result = trainer.train(fuel_data["lap_data"], fuel_data["telemetry_data"])
#             return result
#         except Exception as e:
#             return {'error': f'Fuel model training failed: {str(e)}'}

#     def _train_pit_strategy_model(self, trainer_class, datasets: Dict, processed_data: Dict) -> Dict:
#         """Train pit strategy model with validation"""
#         # Pit strategy requires multiple tracks with sufficient data
#         valid_tracks = {}
#         for track, data in processed_data.items():
#             if (self._validate_track_data({track: data}) and 
#                 len(data.get('lap_data', pd.DataFrame())) >= 5 and
#                 len(data.get('race_data', pd.DataFrame())) >= 2):
#                 valid_tracks[track] = data
        
#         if len(valid_tracks) < 2:
#             return {'error': f'Pit strategy requires 2+ tracks with sufficient data (found {len(valid_tracks)})'}
        
#         try:
#             trainer = trainer_class()
#             result = trainer.train(valid_tracks)
#             return result
#         except Exception as e:
#             return {'error': f'Pit strategy training failed: {str(e)}'}

#     def _train_weather_model(self, trainer_class, datasets: Dict, processed_data: Dict) -> Dict:
#         """Train weather impact model with validation"""
#         if 'weather' not in datasets:
#             return {'error': 'Insufficient data for weather model training'}
        
#         try:
#             trainer = trainer_class()
#             result = trainer.train(datasets['weather'])
#             return result
#         except Exception as e:
#             return {'error': f'Weather model training failed: {str(e)}'}

#     # -------------------------------
#     # DATA PREPARATION WITH VALIDATION
#     # -------------------------------
#     def _prepare_tire_training_data(self, processed_data: Dict) -> Dict[str, pd.DataFrame]:
#         """Prepare tire training data with schema validation"""
#         laps, telemetry, weather = [], [], []
        
#         for track, data in processed_data.items():
#             lap_data = data.get("lap_data", pd.DataFrame())
#             telemetry_data = data.get("telemetry_data", pd.DataFrame())
#             weather_data = data.get("weather_data", pd.DataFrame())
            
#             # Validate minimum data requirements
#             if (len(lap_data) >= 5 and len(telemetry_data) >= 10 and 
#                 not weather_data.empty):
                
#                 # Add track identifier for debugging
#                 lap_data = lap_data.copy().assign(TRACK=track)
#                 telemetry_data = telemetry_data.copy().assign(TRACK=track) if not telemetry_data.empty else telemetry_data
#                 weather_data = weather_data.copy().assign(TRACK=track) if not weather_data.empty else weather_data
                
#                 laps.append(lap_data)
#                 telemetry.append(telemetry_data)
#                 weather.append(weather_data)
        
#         if not laps:
#             return {}
            
#         try:
#             return {
#                 "lap_data": pd.concat(laps, ignore_index=True) if laps else pd.DataFrame(),
#                 "telemetry_data": pd.concat(telemetry, ignore_index=True) if telemetry else pd.DataFrame(),
#                 "weather_data": pd.concat(weather, ignore_index=True) if weather else pd.DataFrame()
#             }
#         except Exception as e:
#             self.logger.error(f"❌ Failed to concatenate tire data: {e}")
#             return {}

#     def _prepare_fuel_training_data(self, processed_data: Dict) -> Dict[str, pd.DataFrame]:
#         """Prepare fuel training data with validation"""
#         laps, telemetry = [], []
        
#         for track, data in processed_data.items():
#             lap_data = data.get("lap_data", pd.DataFrame())
#             telemetry_data = data.get("telemetry_data", pd.DataFrame())
            
#             if len(lap_data) >= 3 and len(telemetry_data) >= 5:
#                 lap_data = lap_data.copy().assign(TRACK=track)
#                 telemetry_data = telemetry_data.copy().assign(TRACK=track) if not telemetry_data.empty else telemetry_data
                
#                 laps.append(lap_data)
#                 telemetry.append(telemetry_data)
        
#         if not laps:
#             return {}
            
#         try:
#             return {
#                 "lap_data": pd.concat(laps, ignore_index=True) if laps else pd.DataFrame(),
#                 "telemetry_data": pd.concat(telemetry, ignore_index=True) if telemetry else pd.DataFrame()
#             }
#         except Exception as e:
#             self.logger.error(f"❌ Failed to concatenate fuel data: {e}")
#             return {}

#     def _prepare_weather_training_data(self, processed_data: Dict) -> Dict:
#         """Prepare weather training data with validation"""
#         valid_data = {}
        
#         for track, data in processed_data.items():
#             lap_data = data.get("lap_data", pd.DataFrame())
#             weather_data = data.get("weather_data", pd.DataFrame())
            
#             if len(lap_data) >= 5 and len(weather_data) >= 3:
#                 valid_data[track] = data
        
#         return valid_data if len(valid_data) >= 2 else {}

#     def _validate_tire_data(self, tire_data: Dict) -> bool:
#         """Validate tire training data requirements"""
#         return (tire_data and 
#                 len(tire_data.get("lap_data", pd.DataFrame())) >= 10 and
#                 len(tire_data.get("telemetry_data", pd.DataFrame())) >= 50 and
#                 not tire_data.get("weather_data", pd.DataFrame()).empty)

#     def _validate_fuel_data(self, fuel_data: Dict) -> bool:
#         """Validate fuel training data requirements"""
#         return (fuel_data and 
#                 len(fuel_data.get("lap_data", pd.DataFrame())) >= 8 and
#                 len(fuel_data.get("telemetry_data", pd.DataFrame())) >= 20)

#     def _validate_weather_data(self, weather_data: Dict) -> bool:
#         """Validate weather training data requirements"""
#         return weather_data and len(weather_data) >= 2

#     # -------------------------------
#     # STATE MANAGEMENT & PERSISTENCE
#     # -------------------------------
#     def _load_training_state(self) -> Dict:
#         """Load training state with error handling"""
#         state_file = os.path.join(self.training_state_dir, "training_state.pkl")
#         if os.path.exists(state_file):
#             try:
#                 state = joblib.load(state_file)
#                 if isinstance(state, dict):
#                     return state
#             except Exception as e:
#                 self.logger.warning(f"⚠️ Failed to load training state: {e}")
#         return {"processed_tracks": [], "models": {}, "completed": False}

#     def _update_training_state(self, updates: Dict):
#         """Update training state with error handling"""
#         try:
#             state = self._load_training_state()
#             state.update(updates)
#             state_file = os.path.join(self.training_state_dir, "training_state.pkl")
#             joblib.dump(state, state_file)
#         except Exception as e:
#             self.logger.error(f"❌ Failed to update training state: {e}")

#     def _load_processed_data_state(self) -> Dict:
#         """Load all processed track data from cache"""
#         processed = {}
#         state = self._load_training_state()
        
#         for track in state.get("processed_data_keys", []):
#             cache_file = os.path.join(self.training_state_dir, f"{track}_processed.pkl")
#             if os.path.exists(cache_file):
#                 try:
#                     track_data = joblib.load(cache_file)
#                     if self._validate_track_data(track_data):
#                         processed[track] = track_data
#                 except Exception as e:
#                     self.logger.warning(f"⚠️ Failed to load cached data for {track}: {e}")
#                     continue
        
#         return processed

#     def _save_models(self, models: Dict[str, Any]):
#         """Save models to disk with comprehensive error handling"""
#         saved_count = 0
#         for name, result in models.items():
#             try:
#                 if 'error' in result:
#                     continue
                    
#                 filepath = os.path.join(self.models_output_dir, f"{name}_model.pkl")
                
#                 # Handle different model result structures
#                 if isinstance(result, dict) and "model" in result and hasattr(result["model"], "save_model"):
#                     result["model"].save_model(filepath)
#                 else:
#                     joblib.dump(result, filepath)
                
#                 saved_count += 1
#                 self.logger.debug(f"💾 Saved {name} model")
                
#             except Exception as e:
#                 self.logger.error(f"❌ Failed to save {name} model: {e}")
        
#         self.logger.info(f"💾 Saved {saved_count} models to disk")

#     def _load_existing_models(self) -> Dict:
#         """Load existing models with validation"""
#         models = {}
#         if not os.path.exists(self.models_output_dir):
#             return models
            
#         for file in os.listdir(self.models_output_dir):
#             if file.endswith("_model.pkl"):
#                 name = file.replace("_model.pkl", "")
#                 try:
#                     model_path = os.path.join(self.models_output_dir, file)
#                     models[name] = joblib.load(model_path)
#                     self.logger.debug(f"📂 Loaded existing model: {name}")
#                 except Exception as e:
#                     self.logger.warning(f"⚠️ Failed to load model {name}: {e}")
#                     continue
        
#         return models

#     def _finalize_training_success(self, tracks: List[str], models: Dict):
#         """Finalize successful training"""
#         self._update_training_state({
#             "processed_tracks": tracks,
#             "processed_data_keys": tracks,
#             "models": list(models.keys()),
#             "completed": True
#         })
        
#         # Upload to Firebase if available
#         self._upload_models_to_firebase(models)

#     def _upload_models_to_firebase(self, models: Dict[str, Any]):
#         """Upload models to Firebase with error handling"""
#         try:
#             if hasattr(self.storage, "upload_models_to_firebase"):
#                 success = self.storage.upload_models_to_firebase()
#                 if success:
#                     self.logger.info("🚀 Models uploaded to Firebase Storage")
#                 else:
#                     self.logger.warning("⚠️ Firebase upload failed")
#         except Exception as e:
#             self.logger.error(f"❌ Firebase upload error: {e}")

#     # -------------------------------
#     # FALLBACKS & ERROR RECOVERY
#     # -------------------------------
#     def _generate_fallback_models(self) -> Dict[str, Any]:
#         """Generate fallback models when training fails"""
#         self.logger.warning("🔄 Generating fallback models...")
        
#         fallback_models = {}
#         fallback_message = {"error": "Training failed, using fallback model"}
        
#         # Create fallback entries for all expected models
#         expected_models = ['tire_degradation', 'fuel_consumption', 'pit_strategy', 'weather_impact']
#         for model_name in expected_models:
#             fallback_models[model_name] = fallback_message
            
#         return fallback_models

#     def _generate_empty_track_data(self) -> Dict[str, pd.DataFrame]:
#         """Generate empty track data structure"""
#         return {
#             'lap_data': pd.DataFrame(),
#             'race_data': pd.DataFrame(), 
#             'weather_data': pd.DataFrame(),
#             'telemetry_data': pd.DataFrame()
#         }

#     # -------------------------------
#     # LOGGING & VALIDATION
#     # -------------------------------
#     def _log_data_quality(self, track: str, data: Dict[str, pd.DataFrame]):
#         """Log data quality metrics"""
#         report = {}
#         for data_type, df in data.items():
#             if df.empty:
#                 report[data_type] = "EMPTY"
#             else:
#                 report[data_type] = {
#                     "rows": len(df),
#                     "columns": len(df.columns),
#                     "null_pct": round((df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100), 2) if len(df) > 0 else 0
#                 }
        
#         self.logger.info(f"📊 {track} data quality: {report}")

#     def validate_training_results(self, models: Dict) -> Dict[str, Any]:
#         """Validate and report training results"""
#         results = {}
#         for name, result in models.items():
#             if isinstance(result, dict) and "error" in result:
#                 results[name] = {"status": "FAILED", "error": result["error"]}
#             elif isinstance(result, dict) and "accuracy" in result:
#                 acc = result["accuracy"]
#                 status = "GOOD" if acc > 0.8 else "FAIR" if acc > 0.6 else "POOR"
#                 results[name] = {"status": status, "accuracy": acc, "training_samples": result.get("training_samples", 0)}
#             elif isinstance(result, dict) and "test_score" in result:
#                 score = result["test_score"]
#                 status = "GOOD" if score > 0.7 else "FAIR" if score > 0.5 else "POOR"
#                 results[name] = {"status": status, "test_score": score, "training_samples": result.get("training_samples", 0)}
#             elif isinstance(result, dict) and "model" in result:
#                 results[name] = {"status": "TRAINED", "details": "Model trained successfully"}
#             else:
#                 results[name] = {"status": "UNKNOWN", "result": str(result)[:100]}
        
#         return results























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
# import joblib
# from typing import Dict, Any


# class TrainingOrchestrator:
#     def __init__(self, storage):
#         self.storage = storage
#         self.logger = logging.getLogger(__name__)
#         self.models_output_dir = "outputs/models"
#         self.training_state_dir = "outputs/training_state"
#         os.makedirs(self.models_output_dir, exist_ok=True)
#         os.makedirs(self.training_state_dir, exist_ok=True)

#     # -------------------------------
#     # MAIN TRAINING PIPELINE
#     # -------------------------------
#     def train_all_models(self) -> dict:
#         """Orchestrate training of all models with resume capability"""
#         self.logger.info("🚀 Starting robust model training pipeline...")

#         training_state = self._load_training_state()
#         if training_state.get('completed', False):
#             self.logger.info("✅ Training already completed. Loading models...")
#             return self._load_existing_models()

#         available_tracks = self.storage.list_available_tracks()
#         if not available_tracks:
#             self.logger.error("❌ No tracks available in storage")
#             return {}

#         processed_tracks = training_state.get('processed_tracks', [])
#         remaining_tracks = [t for t in available_tracks if t not in processed_tracks]

#         if not remaining_tracks:
#             self.logger.info("📊 All tracks processed. Finalizing models...")
#             return self._finalize_training()

#         self.logger.info(f"📥 Processing {len(remaining_tracks)} tracks: {remaining_tracks}")
#         all_processed_data = self._load_processed_data_state()

#         for idx, track in enumerate(remaining_tracks, 1):
#             try:
#                 self.logger.info(f"🔄 Processing track {idx}/{len(available_tracks)}: {track}")
#                 track_data = self._process_single_track(track)
#                 if track_data:
#                     all_processed_data[track] = track_data
#                     processed_tracks.append(track)
#                     self._update_training_state({
#                         'processed_tracks': processed_tracks,
#                         'processed_data_keys': list(all_processed_data.keys())
#                     })
#                     self._save_track_data(track, track_data)
#                     self.logger.info(f"✅ Track processed: {track}")
#                 else:
#                     self.logger.warning(f"⚠️ Skipped {track}: insufficient data")
#             except Exception as e:
#                 self.logger.error(f"❌ Failed to process {track}: {e}")
#                 continue

#         if all_processed_data:
#             models = self._train_models_incrementally(all_processed_data, training_state.get('models', {}))
#             self._upload_models_to_firebase(models)
#             self._update_training_state({
#                 'processed_tracks': processed_tracks,
#                 'processed_data_keys': list(all_processed_data.keys()),
#                 'models': list(models.keys()),
#                 'completed': len(processed_tracks) == len(available_tracks)
#             })
#             return models
#         else:
#             self.logger.error("❌ No valid data processed for training")
#             return {}

#     # -------------------------------
#     # SINGLE TRACK PROCESSING
#     # -------------------------------
#     def _process_single_track(self, track: str) -> Dict[str, pd.DataFrame]:
#         cache_file = f'{self.training_state_dir}/{track}_processed.pkl'
#         if os.path.exists(cache_file):
#             self.logger.info(f"📂 Loading cached processed data for {track}")
#             return joblib.load(cache_file)

#         track_raw_data = self.storage.load_track_data(track)
#         preprocessor = DataPreprocessor()
#         feature_engineer = FeatureEngineer()

#         processed_track_data = {
#             'lap_data': preprocessor.preprocess_lap_data(track_raw_data.get('lap_data', pd.DataFrame())),
#             'race_data': preprocessor.preprocess_race_data(track_raw_data.get('race_data', pd.DataFrame())),
#             'weather_data': preprocessor.preprocess_weather_data(track_raw_data.get('weather_data', pd.DataFrame())),
#             'telemetry_data': preprocessor.preprocess_telemetry_data(track_raw_data.get('telemetry_data', pd.DataFrame()))
#         }

#         enhanced_data = feature_engineer.create_composite_features({track: processed_track_data})
#         final_data = enhanced_data.get(track, processed_track_data)
#         self._log_data_quality(track, final_data)

#         joblib.dump(final_data, cache_file)
#         self.logger.info(f"💾 Cached processed data for {track}")
#         return final_data

#     # -------------------------------
#     # INCREMENTAL MODEL TRAINING
#     # -------------------------------
#     def _train_models_incrementally(self, processed_data: Dict, existing_models: Dict = None) -> Dict[str, Any]:
#         self.logger.info("🏃 Training models incrementally...")
#         models = existing_models or {}

#         # Tire Model
#         tire_data = self._prepare_tire_training_data(processed_data)
#         if tire_data:
#             try:
#                 tire_trainer = TireModelTrainer()
#                 models['tire_degradation'] = tire_trainer.train(
#                     tire_data['lap_data'],
#                     tire_data['telemetry_data'],
#                     tire_data['weather_data']
#                 )
#             except Exception as e:
#                 self.logger.error(f"❌ Tire model training failed: {e}")

#         # Fuel Model
#         fuel_data = self._prepare_fuel_training_data(processed_data)
#         if fuel_data:
#             try:
#                 fuel_trainer = FuelModelTrainer()
#                 models['fuel_consumption'] = fuel_trainer.train(
#                     fuel_data['lap_data'],
#                     fuel_data['telemetry_data']
#                 )
#             except Exception as e:
#                 self.logger.error(f"❌ Fuel model training failed: {e}")

#         # Pit Strategy Model
#         if len(processed_data) >= 2:
#             try:
#                 pit_trainer = PitStrategyTrainer()
#                 pit_result = pit_trainer.train(processed_data)
#                 if 'error' not in pit_result:
#                     models['pit_strategy'] = pit_result
#             except Exception as e:
#                 self.logger.error(f"❌ Pit strategy model training failed: {e}")

#         # Weather Model
#         weather_data = self._prepare_weather_training_data(processed_data)
#         if weather_data:
#             try:
#                 weather_trainer = WeatherModelTrainer()
#                 models['weather_impact'] = weather_trainer.train(weather_data)
#             except Exception as e:
#                 self.logger.error(f"❌ Weather model training failed: {e}")

#         self._save_models(models)
#         self.logger.info(f"✅ Incremental training completed: {len(models)} models processed")
#         return models

#     # -------------------------------
#     # DATA PREPARATION (LOWER THRESHOLDS)
#     # -------------------------------
#     def _prepare_tire_training_data(self, processed_data: Dict) -> Dict[str, pd.DataFrame]:
#         laps, telemetry, weather = [], [], []
#         for track, data in processed_data.items():
#             lap_data, telem_data, weather_data = data.get('lap_data', pd.DataFrame()), data.get('telemetry_data', pd.DataFrame()), data.get('weather_data', pd.DataFrame())
#             if len(lap_data) >= 3 and len(telem_data) >= 10:
#                 laps.append(lap_data.assign(TRACK=track))
#                 telemetry.append(telem_data)
#                 weather.append(weather_data)
#         if not laps: return {}
#         return {'lap_data': pd.concat(laps, ignore_index=True),
#                 'telemetry_data': pd.concat(telemetry, ignore_index=True),
#                 'weather_data': pd.concat(weather, ignore_index=True)}

#     def _prepare_fuel_training_data(self, processed_data: Dict) -> Dict[str, pd.DataFrame]:
#         laps, telemetry = [], []
#         for track, data in processed_data.items():
#             lap_data, telem_data = data.get('lap_data', pd.DataFrame()), data.get('telemetry_data', pd.DataFrame())
#             if len(lap_data) >= 2 and not telem_data.empty:
#                 laps.append(lap_data.assign(TRACK=track))
#                 telemetry.append(telem_data)
#         if not laps: return {}
#         return {'lap_data': pd.concat(laps, ignore_index=True),
#                 'telemetry_data': pd.concat(telemetry, ignore_index=True)}

#     def _prepare_weather_training_data(self, processed_data: Dict) -> Dict:
#         valid = {}
#         for track, data in processed_data.items():
#             lap_data, weather_data = data.get('lap_data', pd.DataFrame()), data.get('weather_data', pd.DataFrame())
#             if len(lap_data) >= 3 and len(weather_data) >= 2:
#                 valid[track] = data
#         return valid if len(valid) >= 2 else {}

#     # -------------------------------
#     # MODEL & STATE MANAGEMENT
#     # -------------------------------
#     def _load_training_state(self) -> Dict:
#         state_file = f'{self.training_state_dir}/training_state.pkl'
#         if os.path.exists(state_file):
#             try: return joblib.load(state_file)
#             except: pass
#         return {'processed_tracks': [], 'models': {}}

#     def _update_training_state(self, updates: Dict):
#         state_file = f'{self.training_state_dir}/training_state.pkl'
#         state = self._load_training_state()
#         state.update(updates)
#         joblib.dump(state, state_file)

#     def _load_processed_data_state(self) -> Dict:
#         processed = {}
#         for track in self._load_training_state().get('processed_data_keys', []):
#             cache_file = f'{self.training_state_dir}/{track}_processed.pkl'
#             if os.path.exists(cache_file):
#                 try: processed[track] = joblib.load(cache_file)
#                 except: continue
#         return processed

#     def _save_track_data(self, track: str, data: Dict): joblib.dump(data, f'{self.training_state_dir}/{track}_processed.pkl')

#     def _save_models(self, models: Dict[str, Any]) -> Dict[str, str]:
#         saved = {}
#         for name, result in models.items():
#             try:
#                 filepath = os.path.join(self.models_output_dir, f"{name}_model.pkl")
#                 if isinstance(result, dict) and 'model' in result and hasattr(result['model'], 'save_model'):
#                     result['model'].save_model(filepath)
#                 else: joblib.dump(result, filepath)
#                 saved[name] = filepath
#             except Exception as e: self.logger.error(f"❌ Failed to save {name} model: {e}")
#         return saved

#     def _load_existing_models(self) -> Dict: 
#         models = {}
#         for file in os.listdir(self.models_output_dir):
#             if file.endswith('.pkl'):
#                 name = file.replace('_model.pkl','')
#                 try: models[name] = joblib.load(os.path.join(self.models_output_dir, file))
#                 except: continue
#         return models

#     def _finalize_training(self) -> Dict: 
#         models = self._load_existing_models()
#         self._update_training_state({'completed': True})
#         return models

#     def _upload_models_to_firebase(self, models: Dict[str, Any]):
#         try:
#             if hasattr(self.storage, 'upload_models_to_firebase'):
#                 success = self.storage.upload_models_to_firebase()
#                 if success: self.logger.info("🚀 Models uploaded to Firebase Storage")
#         except Exception as e: self.logger.error(f"❌ Firebase upload failed: {e}")

#     # -------------------------------
#     # LOGGING & VALIDATION
#     # -------------------------------
#     def _log_data_quality(self, track: str, data: Dict[str, pd.DataFrame]):
#         report = {}
#         for k, df in data.items():
#             if df.empty: report[k] = "EMPTY"
#             else: report[k] = {'rows': len(df), 'columns': len(df.columns), 'null_pct': (df.isnull().sum().sum()/(len(df)*len(df.columns)))*100}
#         self.logger.info(f"📊 {track} data quality: {report}")

#     def validate_training_results(self, models: Dict) -> Dict[str, Any]:
#         results = {}
#         for name, result in models.items():
#             if isinstance(result, dict) and 'error' in result: results[name] = {'status':'FAILED','error':result['error']}
#             elif isinstance(result, dict) and 'accuracy' in result:
#                 acc = result['accuracy']; status = 'GOOD' if acc>0.8 else 'FAIR' if acc>0.6 else 'POOR'
#                 results[name] = {'status': status,'accuracy':acc,'training_samples':result.get('training_samples',0)}
#             elif isinstance(result, dict) and 'test_score' in result:
#                 score = result['test_score']; status = 'GOOD' if score>0.7 else 'FAIR' if score>0.5 else 'POOR'
#                 results[name] = {'status': status,'test_score':score,'training_samples':result.get('training_samples',0)}
#             else: results[name] = {'status':'UNKNOWN','result':result}
#         return results























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
# import joblib
# from typing import Dict, Any, List


# class TrainingOrchestrator:
#     def __init__(self, storage):
#         self.storage = storage
#         self.logger = logging.getLogger(__name__)
#         self.models_output_dir = "outputs/models"
#         self.training_state_dir = "outputs/training_state"
#         os.makedirs(self.models_output_dir, exist_ok=True)
#         os.makedirs(self.training_state_dir, exist_ok=True)

#     def train_all_models(self) -> dict:
#         """Orchestrate training of all models with resume capability and optimized data loading"""
#         self.logger.info("🚀 Starting optimized model training pipeline with resume capability...")

#         # Check for existing training state
#         training_state = self._load_training_state()
#         if training_state.get('completed', False):
#             self.logger.info("✅ Training already completed. Loading existing models...")
#             return self._load_existing_models()

#         # Load available tracks dynamically using optimized loading
#         available_tracks = self.storage.list_available_tracks()
#         if not available_tracks:
#             self.logger.error("❌ No tracks found in storage")
#             return {}

#         # Filter out already processed tracks
#         processed_tracks = training_state.get('processed_tracks', [])
#         remaining_tracks = [t for t in available_tracks if t not in processed_tracks]
        
#         if not remaining_tracks:
#             self.logger.info("📊 All tracks already processed. Finalizing models...")
#             return self._finalize_training()

#         self.logger.info(f"📥 Processing {len(remaining_tracks)} remaining tracks: {remaining_tracks}")

#         # Process tracks individually with state persistence
#         all_processed_data = self._load_processed_data_state()
        
#         for track in remaining_tracks:
#             try:
#                 self.logger.info(f"🔄 Processing track {len(processed_tracks) + 1}/{len(available_tracks)}: {track}")
                
#                 # Load and process single track
#                 track_data = self._process_single_track(track)
#                 if track_data:
#                     all_processed_data[track] = track_data
#                     processed_tracks.append(track)
                    
#                     # Update training state after each track
#                     self._update_training_state({
#                         'processed_tracks': processed_tracks,
#                         'processed_data_keys': list(all_processed_data.keys())
#                     })
                    
#                     # Save processed data for this track
#                     self._save_track_data(track, track_data)
                    
#                     self.logger.info(f"✅ Successfully processed {track}")
#                 else:
#                     self.logger.warning(f"⚠️ Skipping {track} due to insufficient data")

#             except Exception as e:
#                 self.logger.error(f"❌ Failed to process {track}: {e}")
#                 continue

#         # Train models incrementally if we have new data
#         if all_processed_data:
#             models = self._train_models_incrementally(all_processed_data, training_state.get('models', {}))
            
#             # Upload successful models to Firebase
#             self._upload_models_to_firebase(models)
            
#             self._update_training_state({
#                 'processed_tracks': processed_tracks,
#                 'processed_data_keys': list(all_processed_data.keys()),
#                 'models': list(models.keys()),
#                 'completed': len(processed_tracks) == len(available_tracks)
#             })
#             return models
#         else:
#             self.logger.error("❌ No valid data processed for training")
#             return {}

#     def _process_single_track(self, track: str) -> Dict[str, pd.DataFrame]:
#         """Process a single track's data with caching - uses optimized data loading"""
#         # Check for cached processed data
#         cache_file = f'{self.training_state_dir}/{track}_processed.pkl'
#         if os.path.exists(cache_file):
#             self.logger.info(f"📂 Loading cached processed data for {track}")
#             return joblib.load(cache_file)

#         # Load raw data for this track only - uses optimized loading that checks cache first
#         track_raw_data = self.storage.load_track_data(track)
        
#         # Preprocess and engineer features
#         preprocessor = DataPreprocessor()
#         feature_engineer = FeatureEngineer()

#         processed_track_data = {
#             'lap_data': preprocessor.preprocess_lap_data(track_raw_data.get('lap_data', pd.DataFrame())),
#             'race_data': preprocessor.preprocess_race_data(track_raw_data.get('race_data', pd.DataFrame())),
#             'weather_data': preprocessor.preprocess_weather_data(track_raw_data.get('weather_data', pd.DataFrame())),
#             'telemetry_data': preprocessor.preprocess_telemetry_data(track_raw_data.get('telemetry_data', pd.DataFrame()))
#         }

#         # Engineer features safely
#         enhanced_data = feature_engineer.create_composite_features({track: processed_track_data})
#         final_data = enhanced_data.get(track, processed_track_data)

#         self._log_data_quality(track, final_data)

#         # Cache the processed data
#         joblib.dump(final_data, cache_file)
#         self.logger.info(f"💾 Cached processed data for {track}")

#         return final_data

#     def _train_models_incrementally(self, processed_data: Dict, existing_models: Dict = None) -> Dict[str, Any]:
#         """Train models incrementally with optimized data thresholds"""
#         self.logger.info("🏃 Training models incrementally...")
#         models = existing_models or {}

#         # Tire Model - incremental training with LOWERED THRESHOLDS
#         tire_data = self._prepare_tire_training_data(processed_data)
#         if tire_data:
#             try:
#                 tire_trainer = TireModelTrainer()
#                 if 'tire_degradation' in models:
#                     self.logger.info("🔄 Updating existing tire model with new data")
#                     models['tire_degradation'] = tire_trainer.update_model(
#                         models['tire_degradation'], 
#                         tire_data['lap_data'],
#                         tire_data['telemetry_data'],
#                         tire_data['weather_data']
#                     )
#                 else:
#                     self.logger.info("🆕 Training new tire model")
#                     models['tire_degradation'] = tire_trainer.train(
#                         tire_data['lap_data'],
#                         tire_data['telemetry_data'],
#                         tire_data['weather_data']
#                     )
#             except Exception as e:
#                 self.logger.error(f"❌ Tire model training failed: {e}")
#         else:
#             self.logger.warning("⚠️ Insufficient data for tire model training")

#         # Fuel Model - incremental training with LOWERED THRESHOLDS
#         fuel_data = self._prepare_fuel_training_data(processed_data)
#         if fuel_data:
#             try:
#                 fuel_trainer = FuelModelTrainer()
#                 if 'fuel_consumption' in models:
#                     self.logger.info("🔄 Updating existing fuel model with new data")
#                     models['fuel_consumption'] = fuel_trainer.update_model(
#                         models['fuel_consumption'],
#                         fuel_data['lap_data'],
#                         fuel_data['telemetry_data']
#                     )
#                 else:
#                     self.logger.info("🆕 Training new fuel model")
#                     models['fuel_consumption'] = fuel_trainer.train(
#                         fuel_data['lap_data'],
#                         fuel_data['telemetry_data']
#                     )
#             except Exception as e:
#                 self.logger.error(f"❌ Fuel model training failed: {e}")
#         else:
#             self.logger.warning("⚠️ Insufficient data for fuel model training")

#         # Pit Strategy Model - requires multiple tracks
#         if len(processed_data) >= 2:
#             try:
#                 pit_trainer = PitStrategyTrainer()
#                 pit_result = pit_trainer.train(processed_data)
#                 if 'error' not in pit_result:
#                     models['pit_strategy'] = pit_result
#                     self.logger.info("✅ Pit strategy model trained/updated")
#                 else:
#                     self.logger.warning(f"⚠️ Pit strategy training skipped: {pit_result['error']}")
#             except Exception as e:
#                 self.logger.error(f"❌ Pit strategy model training failed: {e}")
#         else:
#             self.logger.warning("⚠️ Insufficient tracks for pit strategy model")

#         # Weather Model - incremental training
#         weather_data = self._prepare_weather_training_data(processed_data)
#         if weather_data:
#             try:
#                 weather_trainer = WeatherModelTrainer()
#                 if 'weather_impact' in models:
#                     self.logger.info("🔄 Updating existing weather model with new data")
#                     models['weather_impact'] = weather_trainer.update_model(
#                         models['weather_impact'],
#                         weather_data
#                     )
#                 else:
#                     self.logger.info("🆕 Training new weather model")
#                     models['weather_impact'] = weather_trainer.train(weather_data)
#             except Exception as e:
#                 self.logger.error(f"❌ Weather model training failed: {e}")
#         else:
#             self.logger.warning("⚠️ Insufficient data for weather model training")

#         # Save models after incremental training
#         self.logger.info("💾 Saving trained models...")
#         self._save_models(models)

#         self.logger.info(f"✅ Incremental training completed: {len(models)} models processed")
#         return models

#     # -------------------------------
#     # State Management Methods
#     # -------------------------------
#     def _load_training_state(self) -> Dict:
#         """Load training state to resume from interruption"""
#         state_file = f'{self.training_state_dir}/training_state.pkl'
#         if os.path.exists(state_file):
#             try:
#                 return joblib.load(state_file)
#             except:
#                 pass
#         return {'processed_tracks': [], 'models': {}}

#     def _update_training_state(self, updates: Dict):
#         """Update training state with new information"""
#         state_file = f'{self.training_state_dir}/training_state.pkl'
#         current_state = self._load_training_state()
#         current_state.update(updates)
#         joblib.dump(current_state, state_file)

#     def _load_processed_data_state(self) -> Dict:
#         """Load all previously processed track data"""
#         processed_data = {}
#         state = self._load_training_state()
#         for track in state.get('processed_data_keys', []):
#             cache_file = f'{self.training_state_dir}/{track}_processed.pkl'
#             if os.path.exists(cache_file):
#                 try:
#                     processed_data[track] = joblib.load(cache_file)
#                 except:
#                     continue
#         return processed_data

#     def _save_track_data(self, track: str, data: Dict):
#         """Save processed track data"""
#         cache_file = f'{self.training_state_dir}/{track}_processed.pkl'
#         joblib.dump(data, cache_file)

#     def _load_existing_models(self) -> Dict:
#         """Load existing trained models"""
#         models = {}
#         for model_file in os.listdir(self.models_output_dir):
#             if model_file.endswith('.pkl'):
#                 model_name = model_file.replace('_model.pkl', '')
#                 try:
#                     models[model_name] = joblib.load(f'{self.models_output_dir}/{model_file}')
#                     self.logger.info(f"📂 Loaded existing model: {model_name}")
#                 except Exception as e:
#                     self.logger.error(f"❌ Failed to load model {model_name}: {e}")
#         return models

#     def _finalize_training(self) -> Dict:
#         """Finalize training by loading all models and marking completion"""
#         models = self._load_existing_models()
#         self._update_training_state({'completed': True})
#         return models

#     def _upload_models_to_firebase(self, models: Dict[str, Any]):
#         """Upload successfully trained models to Firebase Storage"""
#         try:
#             if hasattr(self.storage, 'upload_models_to_firebase'):
#                 success = self.storage.upload_models_to_firebase()
#                 if success:
#                     self.logger.info("🚀 Models uploaded to Firebase Storage")
#                 else:
#                     self.logger.warning("⚠️ Failed to upload models to Firebase")
#             else:
#                 self.logger.warning("⚠️ Firebase storage doesn't support model upload")
#         except Exception as e:
#             self.logger.error(f"❌ Error uploading models to Firebase: {e}")

#     # -------------------------------
#     # Data preparation helpers - OPTIMIZED WITH LOWER THRESHOLDS
#     # -------------------------------
#     def _prepare_tire_training_data(self, processed_data: Dict) -> Dict[str, pd.DataFrame]:
#         """Prepare tire training data with LOWERED thresholds"""
#         lap_list, telemetry_list, weather_list = [], [], []
#         for track, data in processed_data.items():
#             lap_data = data.get('lap_data', pd.DataFrame())
#             telemetry = data.get('telemetry_data', pd.DataFrame())
#             weather = data.get('weather_data', pd.DataFrame())
            
#             # LOWERED THRESHOLDS: From 20 laps → 3 laps, from 100 telemetry → 10
#             if not lap_data.empty and len(lap_data) >= 3 and not telemetry.empty and len(telemetry) >= 10:
#                 lap_copy = lap_data.copy()
#                 lap_copy['TRACK'] = track
#                 lap_list.append(lap_copy)
#                 telemetry_list.append(telemetry)
#                 weather_list.append(weather)
        
#         if not lap_list:
#             return {}
        
#         return {
#             'lap_data': pd.concat(lap_list, ignore_index=True),
#             'telemetry_data': pd.concat(telemetry_list, ignore_index=True),
#             'weather_data': pd.concat(weather_list, ignore_index=True)
#         }

#     def _prepare_fuel_training_data(self, processed_data: Dict) -> Dict[str, pd.DataFrame]:
#         """Prepare fuel training data with LOWERED thresholds"""
#         lap_list, telemetry_list = [], []
#         for track, data in processed_data.items():
#             lap_data = data.get('lap_data', pd.DataFrame())
#             telemetry = data.get('telemetry_data', pd.DataFrame())
            
#             # LOWERED THRESHOLDS: From 15 laps → 2 laps, removed throttle requirement
#             if not lap_data.empty and len(lap_data) >= 2 and not telemetry.empty:
#                 lap_copy = lap_data.copy()
#                 lap_copy['TRACK'] = track
#                 lap_list.append(lap_copy)
#                 telemetry_list.append(telemetry)
        
#         if not lap_list:
#             return {}
        
#         return {
#             'lap_data': pd.concat(lap_list, ignore_index=True),
#             'telemetry_data': pd.concat(telemetry_list, ignore_index=True)
#         }

#     def _prepare_weather_training_data(self, processed_data: Dict) -> Dict:
#         """Prepare weather training data with LOWERED thresholds"""
#         valid_tracks = {}
#         for track, data in processed_data.items():
#             lap_data = data.get('lap_data', pd.DataFrame())
#             weather_data = data.get('weather_data', pd.DataFrame())
            
#             # LOWERED THRESHOLDS: From 10 laps → 3 laps, from 5 weather → 2
#             if not lap_data.empty and len(lap_data) >= 3 and not weather_data.empty and len(weather_data) >= 2:
#                 valid_tracks[track] = data
        
#         return valid_tracks if len(valid_tracks) >= 2 else {}

#     # -------------------------------
#     # Model saving and logging (unchanged - backward compatible)
#     # -------------------------------
#     def _save_models(self, models: Dict[str, Any]) -> Dict[str, str]:
#         saved = {}
#         for name, result in models.items():
#             try:
#                 if isinstance(result, dict) and 'model' in result and hasattr(result['model'], 'save_model'):
#                     filepath = os.path.join(self.models_output_dir, f"{name}_model.pkl")
#                     result['model'].save_model(filepath)
#                     saved[name] = filepath
#                     self.logger.info(f"💾 Saved {name} model to {filepath}")
#                 else:
#                     # Save the entire result for models that don't have save_model method
#                     filepath = os.path.join(self.models_output_dir, f"{name}_model.pkl")
#                     joblib.dump(result, filepath)
#                     saved[name] = filepath
#                     self.logger.info(f"💾 Saved {name} model result to {filepath}")
#             except Exception as e:
#                 self.logger.error(f"❌ Failed to save {name} model: {e}")
#         return saved

#     def _log_data_quality(self, track: str, data: Dict[str, pd.DataFrame]):
#         report = {}
#         for dt, df in data.items():
#             if df.empty:
#                 report[dt] = "EMPTY"
#             else:
#                 report[dt] = {
#                     'rows': len(df),
#                     'columns': len(df.columns),
#                     'null_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
#                 }
#         self.logger.info(f"📊 {track} data quality: {report}")

#     # -------------------------------
#     # Validation helper (unchanged - backward compatible)
#     # -------------------------------
#     def validate_training_results(self, models: Dict) -> Dict[str, Any]:
#         results = {}
#         for name, result in models.items():
#             if isinstance(result, dict) and 'error' in result:
#                 results[name] = {'status': 'FAILED', 'error': result['error']}
#             elif isinstance(result, dict) and 'accuracy' in result:
#                 acc = result['accuracy']
#                 status = 'GOOD' if acc > 0.8 else 'FAIR' if acc > 0.6 else 'POOR'
#                 results[name] = {'status': status, 'accuracy': acc, 'training_samples': result.get('training_samples', 0)}
#             elif isinstance(result, dict) and 'test_score' in result:
#                 score = result['test_score']
#                 status = 'GOOD' if score > 0.7 else 'FAIR' if score > 0.5 else 'POOR'
#                 results[name] = {'status': status, 'test_score': score, 'training_samples': result.get('training_samples', 0)}
#             else:
#                 results[name] = {'status': 'UNKNOWN', 'result': result}
#         return results

























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
# import joblib
# from typing import Dict, Any, List


# class TrainingOrchestrator:
#     def __init__(self, storage):
#         self.storage = storage
#         self.logger = logging.getLogger(__name__)
#         self.models_output_dir = "outputs/models"
#         self.training_state_dir = "outputs/training_state"
#         os.makedirs(self.models_output_dir, exist_ok=True)
#         os.makedirs(self.training_state_dir, exist_ok=True)

#     def train_all_models(self) -> dict:
#         """Orchestrate training of all models with resume capability and individual track processing"""
#         self.logger.info("🚀 Starting optimized model training pipeline with resume capability...")

#         # Check for existing training state
#         training_state = self._load_training_state()
#         if training_state.get('completed', False):
#             self.logger.info("✅ Training already completed. Loading existing models...")
#             return self._load_existing_models()

#         # Load available tracks dynamically
#         available_tracks = self.storage.list_available_tracks()
#         if not available_tracks:
#             self.logger.error("❌ No tracks found in storage")
#             return {}

#         # Filter out already processed tracks
#         processed_tracks = training_state.get('processed_tracks', [])
#         remaining_tracks = [t for t in available_tracks if t not in processed_tracks]
        
#         if not remaining_tracks:
#             self.logger.info("📊 All tracks already processed. Finalizing models...")
#             return self._finalize_training()

#         self.logger.info(f"📥 Processing {len(remaining_tracks)} remaining tracks: {remaining_tracks}")

#         # Process tracks individually with state persistence
#         all_processed_data = self._load_processed_data_state()
        
#         for track in remaining_tracks:
#             try:
#                 self.logger.info(f"🔄 Processing track {len(processed_tracks) + 1}/{len(available_tracks)}: {track}")
                
#                 # Load and process single track
#                 track_data = self._process_single_track(track)
#                 if track_data:
#                     all_processed_data[track] = track_data
#                     processed_tracks.append(track)
                    
#                     # Update training state after each track
#                     self._update_training_state({
#                         'processed_tracks': processed_tracks,
#                         'processed_data_keys': list(all_processed_data.keys())
#                     })
                    
#                     # Save processed data for this track
#                     self._save_track_data(track, track_data)
                    
#                     self.logger.info(f"✅ Successfully processed {track}")
#                 else:
#                     self.logger.warning(f"⚠️ Skipping {track} due to insufficient data")

#             except Exception as e:
#                 self.logger.error(f"❌ Failed to process {track}: {e}")
#                 continue

#         # Train models incrementally if we have new data
#         if all_processed_data:
#             models = self._train_models_incrementally(all_processed_data, training_state.get('models', {}))
#             self._update_training_state({
#                 'processed_tracks': processed_tracks,
#                 'processed_data_keys': list(all_processed_data.keys()),
#                 'models': list(models.keys()),
#                 'completed': len(processed_tracks) == len(available_tracks)
#             })
#             return models
#         else:
#             self.logger.error("❌ No valid data processed for training")
#             return {}

#     def _process_single_track(self, track: str) -> Dict[str, pd.DataFrame]:
#         """Process a single track's data with caching"""
#         # Check for cached processed data
#         cache_file = f'{self.training_state_dir}/{track}_processed.pkl'
#         if os.path.exists(cache_file):
#             self.logger.info(f"📂 Loading cached processed data for {track}")
#             return joblib.load(cache_file)

#         # Load raw data for this track only
#         track_raw_data = self.storage.load_track_data(track)
        
#         # Preprocess and engineer features
#         preprocessor = DataPreprocessor()
#         feature_engineer = FeatureEngineer()

#         processed_track_data = {
#             'lap_data': preprocessor.preprocess_lap_data(track_raw_data.get('lap_data', pd.DataFrame())),
#             'race_data': preprocessor.preprocess_race_data(track_raw_data.get('race_data', pd.DataFrame())),
#             'weather_data': preprocessor.preprocess_weather_data(track_raw_data.get('weather_data', pd.DataFrame())),
#             'telemetry_data': preprocessor.preprocess_telemetry_data(track_raw_data.get('telemetry_data', pd.DataFrame()))
#         }

#         # Engineer features safely
#         enhanced_data = feature_engineer.create_composite_features({track: processed_track_data})
#         final_data = enhanced_data.get(track, processed_track_data)

#         self._log_data_quality(track, final_data)

#         # Cache the processed data
#         joblib.dump(final_data, cache_file)
#         self.logger.info(f"💾 Cached processed data for {track}")

#         return final_data

#     def _train_models_incrementally(self, processed_data: Dict, existing_models: Dict = None) -> Dict[str, Any]:
#         """Train models incrementally, updating existing models with new data"""
#         self.logger.info("🏃 Training models incrementally...")
#         models = existing_models or {}

#         # Tire Model - incremental training
#         tire_data = self._prepare_tire_training_data(processed_data)
#         if tire_data:
#             try:
#                 tire_trainer = TireModelTrainer()
#                 if 'tire_degradation' in models:
#                     self.logger.info("🔄 Updating existing tire model with new data")
#                     models['tire_degradation'] = tire_trainer.update_model(
#                         models['tire_degradation'], 
#                         tire_data['lap_data'],
#                         tire_data['telemetry_data'],
#                         tire_data['weather_data']
#                     )
#                 else:
#                     self.logger.info("🆕 Training new tire model")
#                     models['tire_degradation'] = tire_trainer.train(
#                         tire_data['lap_data'],
#                         tire_data['telemetry_data'],
#                         tire_data['weather_data']
#                     )
#             except Exception as e:
#                 self.logger.error(f"❌ Tire model training failed: {e}")
#         else:
#             self.logger.warning("⚠️ Insufficient data for tire model training")

#         # Fuel Model - incremental training
#         fuel_data = self._prepare_fuel_training_data(processed_data)
#         if fuel_data:
#             try:
#                 fuel_trainer = FuelModelTrainer()
#                 if 'fuel_consumption' in models:
#                     self.logger.info("🔄 Updating existing fuel model with new data")
#                     models['fuel_consumption'] = fuel_trainer.update_model(
#                         models['fuel_consumption'],
#                         fuel_data['lap_data'],
#                         fuel_data['telemetry_data']
#                     )
#                 else:
#                     self.logger.info("🆕 Training new fuel model")
#                     models['fuel_consumption'] = fuel_trainer.train(
#                         fuel_data['lap_data'],
#                         fuel_data['telemetry_data']
#                     )
#             except Exception as e:
#                 self.logger.error(f"❌ Fuel model training failed: {e}")
#         else:
#             self.logger.warning("⚠️ Insufficient data for fuel model training")

#         # Pit Strategy Model - requires multiple tracks
#         if len(processed_data) >= 2:
#             try:
#                 pit_trainer = PitStrategyTrainer()
#                 pit_result = pit_trainer.train(processed_data)
#                 if 'error' not in pit_result:
#                     models['pit_strategy'] = pit_result
#                     self.logger.info("✅ Pit strategy model trained/updated")
#                 else:
#                     self.logger.warning(f"⚠️ Pit strategy training skipped: {pit_result['error']}")
#             except Exception as e:
#                 self.logger.error(f"❌ Pit strategy model training failed: {e}")
#         else:
#             self.logger.warning("⚠️ Insufficient tracks for pit strategy model")

#         # Weather Model - incremental training
#         weather_data = self._prepare_weather_training_data(processed_data)
#         if weather_data:
#             try:
#                 weather_trainer = WeatherModelTrainer()
#                 if 'weather_impact' in models:
#                     self.logger.info("🔄 Updating existing weather model with new data")
#                     models['weather_impact'] = weather_trainer.update_model(
#                         models['weather_impact'],
#                         weather_data
#                     )
#                 else:
#                     self.logger.info("🆕 Training new weather model")
#                     models['weather_impact'] = weather_trainer.train(weather_data)
#             except Exception as e:
#                 self.logger.error(f"❌ Weather model training failed: {e}")
#         else:
#             self.logger.warning("⚠️ Insufficient data for weather model training")

#         # Save models after incremental training
#         self.logger.info("💾 Saving trained models...")
#         self._save_models(models)

#         self.logger.info(f"✅ Incremental training completed: {len(models)} models processed")
#         return models

#     # -------------------------------
#     # State Management Methods
#     # -------------------------------
#     def _load_training_state(self) -> Dict:
#         """Load training state to resume from interruption"""
#         state_file = f'{self.training_state_dir}/training_state.pkl'
#         if os.path.exists(state_file):
#             try:
#                 return joblib.load(state_file)
#             except:
#                 pass
#         return {'processed_tracks': [], 'models': {}}

#     def _update_training_state(self, updates: Dict):
#         """Update training state with new information"""
#         state_file = f'{self.training_state_dir}/training_state.pkl'
#         current_state = self._load_training_state()
#         current_state.update(updates)
#         joblib.dump(current_state, state_file)

#     def _load_processed_data_state(self) -> Dict:
#         """Load all previously processed track data"""
#         processed_data = {}
#         state = self._load_training_state()
#         for track in state.get('processed_data_keys', []):
#             cache_file = f'{self.training_state_dir}/{track}_processed.pkl'
#             if os.path.exists(cache_file):
#                 try:
#                     processed_data[track] = joblib.load(cache_file)
#                 except:
#                     continue
#         return processed_data

#     def _save_track_data(self, track: str, data: Dict):
#         """Save processed track data"""
#         cache_file = f'{self.training_state_dir}/{track}_processed.pkl'
#         joblib.dump(data, cache_file)

#     def _load_existing_models(self) -> Dict:
#         """Load existing trained models"""
#         models = {}
#         for model_file in os.listdir(self.models_output_dir):
#             if model_file.endswith('.pkl'):
#                 model_name = model_file.replace('_model.pkl', '')
#                 try:
#                     models[model_name] = joblib.load(f'{self.models_output_dir}/{model_file}')
#                     self.logger.info(f"📂 Loaded existing model: {model_name}")
#                 except Exception as e:
#                     self.logger.error(f"❌ Failed to load model {model_name}: {e}")
#         return models

#     def _finalize_training(self) -> Dict:
#         """Finalize training by loading all models and marking completion"""
#         models = self._load_existing_models()
#         self._update_training_state({'completed': True})
#         return models

#     # -------------------------------
#     # Data preparation helpers (unchanged but kept for completeness)
#     # -------------------------------
#     def _prepare_tire_training_data(self, processed_data: Dict) -> Dict[str, pd.DataFrame]:
#         lap_list, telemetry_list, weather_list = [], [], []
#         for track, data in processed_data.items():
#             lap_data = data.get('lap_data', pd.DataFrame())
#             telemetry = data.get('telemetry_data', pd.DataFrame())
#             weather = data.get('weather_data', pd.DataFrame())
#             if not lap_data.empty and len(lap_data) >= 20 and not telemetry.empty and len(telemetry) >= 100:
#                 lap_copy = lap_data.copy()
#                 lap_copy['TRACK'] = track
#                 lap_list.append(lap_copy)
#                 telemetry_list.append(telemetry)
#                 weather_list.append(weather)
#         if not lap_list:
#             return {}
#         return {
#             'lap_data': pd.concat(lap_list, ignore_index=True),
#             'telemetry_data': pd.concat(telemetry_list, ignore_index=True),
#             'weather_data': pd.concat(weather_list, ignore_index=True)
#         }

#     def _prepare_fuel_training_data(self, processed_data: Dict) -> Dict[str, pd.DataFrame]:
#         lap_list, telemetry_list = [], []
#         for track, data in processed_data.items():
#             lap_data = data.get('lap_data', pd.DataFrame())
#             telemetry = data.get('telemetry_data', pd.DataFrame())
#             if not lap_data.empty and len(lap_data) >= 15 and not telemetry.empty and 'THROTTLE_POSITION' in telemetry.columns:
#                 lap_copy = lap_data.copy()
#                 lap_copy['TRACK'] = track
#                 lap_list.append(lap_copy)
#                 telemetry_list.append(telemetry)
#         if not lap_list:
#             return {}
#         return {
#             'lap_data': pd.concat(lap_list, ignore_index=True),
#             'telemetry_data': pd.concat(telemetry_list, ignore_index=True)
#         }

#     def _prepare_weather_training_data(self, processed_data: Dict) -> Dict:
#         valid_tracks = {}
#         for track, data in processed_data.items():
#             lap_data = data.get('lap_data', pd.DataFrame())
#             weather_data = data.get('weather_data', pd.DataFrame())
#             if not lap_data.empty and len(lap_data) >= 10 and not weather_data.empty and len(weather_data) >= 5:
#                 valid_tracks[track] = data
#         return valid_tracks if len(valid_tracks) >= 2 else {}

#     # -------------------------------
#     # Model saving and logging (unchanged)
#     # -------------------------------
#     def _save_models(self, models: Dict[str, Any]) -> Dict[str, str]:
#         saved = {}
#         for name, result in models.items():
#             try:
#                 if isinstance(result, dict) and 'model' in result and hasattr(result['model'], 'save_model'):
#                     filepath = os.path.join(self.models_output_dir, f"{name}_model.pkl")
#                     result['model'].save_model(filepath)
#                     saved[name] = filepath
#                     self.logger.info(f"💾 Saved {name} model to {filepath}")
#                 else:
#                     # Save the entire result for models that don't have save_model method
#                     filepath = os.path.join(self.models_output_dir, f"{name}_model.pkl")
#                     joblib.dump(result, filepath)
#                     saved[name] = filepath
#                     self.logger.info(f"💾 Saved {name} model result to {filepath}")
#             except Exception as e:
#                 self.logger.error(f"❌ Failed to save {name} model: {e}")
#         return saved

#     def _log_data_quality(self, track: str, data: Dict[str, pd.DataFrame]):
#         report = {}
#         for dt, df in data.items():
#             if df.empty:
#                 report[dt] = "EMPTY"
#             else:
#                 report[dt] = {
#                     'rows': len(df),
#                     'columns': len(df.columns),
#                     'null_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
#                 }
#         self.logger.info(f"📊 {track} data quality: {report}")

#     # -------------------------------
#     # Validation helper (unchanged)
#     # -------------------------------
#     def validate_training_results(self, models: Dict) -> Dict[str, Any]:
#         results = {}
#         for name, result in models.items():
#             if isinstance(result, dict) and 'error' in result:
#                 results[name] = {'status': 'FAILED', 'error': result['error']}
#             elif isinstance(result, dict) and 'accuracy' in result:
#                 acc = result['accuracy']
#                 status = 'GOOD' if acc > 0.8 else 'FAIR' if acc > 0.6 else 'POOR'
#                 results[name] = {'status': status, 'accuracy': acc, 'training_samples': result.get('training_samples', 0)}
#             elif isinstance(result, dict) and 'test_score' in result:
#                 score = result['test_score']
#                 status = 'GOOD' if score > 0.7 else 'FAIR' if score > 0.5 else 'POOR'
#                 results[name] = {'status': status, 'test_score': score, 'training_samples': result.get('training_samples', 0)}
#             else:
#                 results[name] = {'status': 'UNKNOWN', 'result': result}
#         return results




















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

#         # Load data from Firebase
#         all_data = self.storage.load_all_tracks(available_tracks)

#         # Preprocess and engineer features
#         preprocessor = DataPreprocessor()
#         feature_engineer = FeatureEngineer()

#         processed_data = {}
#         for track, data in all_data.items():
#             self.logger.info(f"🔄 Processing {track}...")

#             processed_track_data = {
#                 'lap_data': preprocessor.preprocess_lap_data(data.get('lap_data', pd.DataFrame())),
#                 'race_data': preprocessor.preprocess_race_data(data.get('race_data', pd.DataFrame())),
#                 'weather_data': preprocessor.preprocess_weather_data(data.get('weather_data', pd.DataFrame())),
#                 'telemetry_data': preprocessor.preprocess_telemetry_data(data.get('telemetry_data', pd.DataFrame()))
#             }

#             # Engineer features safely
#             enhanced_data = feature_engineer.create_composite_features({track: processed_track_data})
#             processed_data[track] = enhanced_data.get(track, processed_track_data)

#             self._log_data_quality(track, processed_data[track])

#         # Train models
#         self.logger.info("🏃 Training models with integrated data...")
#         models = {}

#         # Tire Model
#         tire_trainer = TireModelTrainer()
#         tire_data = self._prepare_tire_training_data(processed_data)
#         if tire_data:
#             try:
#                 models['tire_degradation'] = tire_trainer.train(
#                     tire_data['lap_data'],
#                     tire_data['telemetry_data'],
#                     tire_data['weather_data']
#                 )
#             except Exception as e:
#                 self.logger.error(f"❌ Tire model training failed: {e}")
#         else:
#             self.logger.warning("⚠️ Insufficient data for tire model training")

#         # Fuel Model
#         fuel_trainer = FuelModelTrainer()
#         fuel_data = self._prepare_fuel_training_data(processed_data)
#         if fuel_data:
#             try:
#                 models['fuel_consumption'] = fuel_trainer.train(
#                     fuel_data['lap_data'],
#                     fuel_data['telemetry_data']
#                 )
#             except Exception as e:
#                 self.logger.error(f"❌ Fuel model training failed: {e}")
#         else:
#             self.logger.warning("⚠️ Insufficient data for fuel model training")

#         # Pit Strategy Model
#         pit_trainer = PitStrategyTrainer()
#         if len(processed_data) >= 2:
#             try:
#                 pit_result = pit_trainer.train(processed_data)
#                 if 'error' not in pit_result:
#                     models['pit_strategy'] = pit_result
#                 else:
#                     self.logger.warning(f"⚠️ Pit strategy training skipped: {pit_result['error']}")
#             except Exception as e:
#                 self.logger.error(f"❌ Pit strategy model training failed: {e}")
#         else:
#             self.logger.warning("⚠️ Insufficient tracks for pit strategy model")

#         # Weather Model
#         weather_trainer = WeatherModelTrainer()
#         weather_data = self._prepare_weather_training_data(processed_data)
#         if weather_data:
#             try:
#                 models['weather_impact'] = weather_trainer.train(weather_data)
#             except Exception as e:
#                 self.logger.error(f"❌ Weather model training failed: {e}")
#         else:
#             self.logger.warning("⚠️ Insufficient data for weather model training")

#         # Save models
#         self.logger.info("💾 Saving trained models...")
#         self._save_models(models)

#         self.logger.info(f"✅ Training completed: {len(models)} models processed")
#         return models

#     # -------------------------------
#     # Data preparation helpers
#     # -------------------------------
#     def _prepare_tire_training_data(self, processed_data: Dict) -> Dict[str, pd.DataFrame]:
#         lap_list, telemetry_list, weather_list = [], [], []
#         for track, data in processed_data.items():
#             lap_data = data.get('lap_data', pd.DataFrame())
#             telemetry = data.get('telemetry_data', pd.DataFrame())
#             weather = data.get('weather_data', pd.DataFrame())
#             if not lap_data.empty and len(lap_data) >= 20 and not telemetry.empty and len(telemetry) >= 100:
#                 lap_copy = lap_data.copy()
#                 lap_copy['TRACK'] = track
#                 lap_list.append(lap_copy)
#                 telemetry_list.append(telemetry)
#                 weather_list.append(weather)
#         if not lap_list:
#             return {}
#         return {
#             'lap_data': pd.concat(lap_list, ignore_index=True),
#             'telemetry_data': pd.concat(telemetry_list, ignore_index=True),
#             'weather_data': pd.concat(weather_list, ignore_index=True)
#         }

#     def _prepare_fuel_training_data(self, processed_data: Dict) -> Dict[str, pd.DataFrame]:
#         lap_list, telemetry_list = [], []
#         for track, data in processed_data.items():
#             lap_data = data.get('lap_data', pd.DataFrame())
#             telemetry = data.get('telemetry_data', pd.DataFrame())
#             if not lap_data.empty and len(lap_data) >= 15 and not telemetry.empty and 'THROTTLE_POSITION' in telemetry.columns:
#                 lap_copy = lap_data.copy()
#                 lap_copy['TRACK'] = track
#                 lap_list.append(lap_copy)
#                 telemetry_list.append(telemetry)
#         if not lap_list:
#             return {}
#         return {
#             'lap_data': pd.concat(lap_list, ignore_index=True),
#             'telemetry_data': pd.concat(telemetry_list, ignore_index=True)
#         }

#     def _prepare_weather_training_data(self, processed_data: Dict) -> Dict:
#         valid_tracks = {}
#         for track, data in processed_data.items():
#             lap_data = data.get('lap_data', pd.DataFrame())
#             weather_data = data.get('weather_data', pd.DataFrame())
#             if not lap_data.empty and len(lap_data) >= 10 and not weather_data.empty and len(weather_data) >= 5:
#                 valid_tracks[track] = data
#         return valid_tracks if len(valid_tracks) >= 2 else {}

#     # -------------------------------
#     # Model saving and logging
#     # -------------------------------
#     def _save_models(self, models: Dict[str, Any]) -> Dict[str, str]:
#         saved = {}
#         for name, result in models.items():
#             try:
#                 if isinstance(result, dict) and 'model' in result and hasattr(result['model'], 'save_model'):
#                     filepath = os.path.join(self.models_output_dir, f"{name}_model.pkl")
#                     result['model'].save_model(filepath)
#                     saved[name] = filepath
#                     self.logger.info(f"💾 Saved {name} model to {filepath}")
#             except Exception as e:
#                 self.logger.error(f"❌ Failed to save {name} model: {e}")
#         return saved

#     def _log_data_quality(self, track: str, data: Dict[str, pd.DataFrame]):
#         report = {}
#         for dt, df in data.items():
#             if df.empty:
#                 report[dt] = "EMPTY"
#             else:
#                 report[dt] = {
#                     'rows': len(df),
#                     'columns': len(df.columns),
#                     'null_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
#                 }
#         self.logger.info(f"📊 {track} data quality: {report}")

#     # -------------------------------
#     # Validation helper
#     # -------------------------------
#     def validate_training_results(self, models: Dict) -> Dict[str, Any]:
#         results = {}
#         for name, result in models.items():
#             if isinstance(result, dict) and 'error' in result:
#                 results[name] = {'status': 'FAILED', 'error': result['error']}
#             elif isinstance(result, dict) and 'accuracy' in result:
#                 acc = result['accuracy']
#                 status = 'GOOD' if acc > 0.8 else 'FAIR' if acc > 0.6 else 'POOR'
#                 results[name] = {'status': status, 'accuracy': acc, 'training_samples': result.get('training_samples', 0)}
#             elif isinstance(result, dict) and 'test_score' in result:
#                 score = result['test_score']
#                 status = 'GOOD' if score > 0.7 else 'FAIR' if score > 0.5 else 'POOR'
#                 results[name] = {'status': status, 'test_score': score, 'training_samples': result.get('training_samples', 0)}
#             else:
#                 results[name] = {'status': 'UNKNOWN', 'result': result}
#         return results



























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