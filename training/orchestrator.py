import os
import logging
import joblib
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

from models.tire_trainer import TireModelTrainer
from models.fuel_trainer import FuelModelTrainer
from models.pit_strategy_trainer import PitStrategyTrainer
from models.weather_trainer import WeatherModelTrainer
from data.preprocessor import DataPreprocessor
from data.feature_engineer import FeatureEngineer
from data.firebase_loader import FirebaseDataLoader


class TrainingOrchestrator:
    def __init__(self, storage: FirebaseDataLoader):
        self.storage = storage
        self.logger = logging.getLogger(__name__)
        self.models_output_dir = "outputs/models"
        self.training_state_dir = "outputs/training_state"
        os.makedirs(self.models_output_dir, exist_ok=True)
        os.makedirs(self.training_state_dir, exist_ok=True)

    # -------------------------------
    # MAIN TRAINING PIPELINE - Updated for FirebaseDataLoader consistency
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
                self.logger.info(f"🔄 Processing track {idx}/{len(remaining_tracks)}: {track}")
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
            models = self._train_models_with_processed_data(all_processed_data, state.get("models", {}))
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
    # SINGLE TRACK PROCESSING - Updated for FirebaseDataLoader consistency
    # -------------------------------
    def _process_single_track(self, track: str) -> Dict[str, pd.DataFrame]:
        cache_file = os.path.join(self.training_state_dir, f"{track}_processed.pkl")
        if os.path.exists(cache_file):
            self.logger.info(f"📂 Loading cached processed data for {track}")
            try:
                return joblib.load(cache_file)
            except Exception:
                self.logger.warning(f"⚠️ Cached file corrupted. Reprocessing {track}")

        # Load track data using FirebaseDataLoader (returns dict with pit_data, race_data, etc.)
        track_raw = self.storage.load_track_data(track)
        
        # Preprocess using updated DataPreprocessor
        preprocessor = DataPreprocessor()
        processed = preprocessor.preprocess_track_data(track_raw)
        
        # Engineer features using updated FeatureEngineer
        feature_engineer = FeatureEngineer()
        enhanced_data = feature_engineer.create_composite_features({track: processed})
        
        track_processed = enhanced_data.get(track, processed)
        self._log_data_quality(track, track_processed)

        try:
            joblib.dump(track_processed, cache_file)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to cache processed data for {track}: {e}")
            
        return track_processed

    # -------------------------------
    # MODEL TRAINING - Updated validation logic to accept tracks with any pit data
    # -------------------------------
    def _train_models_with_processed_data(self, processed_data: Dict[str, Dict[str, pd.DataFrame]], existing_models: Dict = None) -> Dict[str, Any]:
        self.logger.info("🏃 Training models with processed data...")
        models = existing_models or {}

        # UPDATED: Accept tracks with ANY pit data, not just specific columns
        valid_tracks = {
            track_name: data_dict 
            for track_name, data_dict in processed_data.items() 
            if not data_dict.get('pit_data', pd.DataFrame()).empty
        }

        if not valid_tracks:
            self.logger.error("❌ No valid tracks with pit data for training")
            return models

        self.logger.info(f"📊 Training with {len(valid_tracks)} valid tracks: {list(valid_tracks.keys())}")

        # Tire Model Training - Updated for new method signature
        try:
            tire_trainer = TireModelTrainer()
            tire_result = tire_trainer.train(valid_tracks)
            if tire_result.get('status') == 'success':
                models["tire_degradation"] = tire_result
                self.logger.info(f"✅ Tire model trained: {tire_result.get('test_score', 'N/A')} score")
            else:
                models["tire_degradation"] = {"error": tire_result.get('error', 'Unknown error')}
                self.logger.error(f"❌ Tire model training failed: {tire_result.get('error')}")
        except Exception as e:
            self.logger.error(f"❌ Tire model training exception: {e}")
            models["tire_degradation"] = {"error": str(e)}

        # Fuel Model Training - Updated for new method signature
        try:
            fuel_trainer = FuelModelTrainer()
            fuel_result = fuel_trainer.train(valid_tracks)
            if fuel_result.get('status') == 'success':
                models["fuel_consumption"] = fuel_result
                self.logger.info(f"✅ Fuel model trained: {fuel_result.get('test_score', 'N/A')} score")
            else:
                models["fuel_consumption"] = {"error": fuel_result.get('error', 'Unknown error')}
                self.logger.error(f"❌ Fuel model training failed: {fuel_result.get('error')}")
        except Exception as e:
            self.logger.error(f"❌ Fuel model training exception: {e}")
            models["fuel_consumption"] = {"error": str(e)}

        # Pit Strategy Model Training - Updated for new method signature
        if len(valid_tracks) >= 2:
            try:
                pit_trainer = PitStrategyTrainer()
                pit_result = pit_trainer.train(valid_tracks)
                if pit_result.get('status') == 'success':
                    models["pit_strategy"] = pit_result
                    self.logger.info(f"✅ Pit strategy model trained: {pit_result.get('accuracy', 'N/A')} accuracy")
                else:
                    models["pit_strategy"] = {"error": pit_result.get('error', 'Unknown error')}
                    self.logger.error(f"❌ Pit strategy model training failed: {pit_result.get('error')}")
            except Exception as e:
                self.logger.error(f"❌ Pit strategy model training exception: {e}")
                models["pit_strategy"] = {"error": str(e)}
        else:
            self.logger.warning("⚠️ Insufficient tracks for pit strategy model (need >= 2)")

        # Weather Model Training - Updated for new method signature
        if len(valid_tracks) >= 2:
            try:
                weather_trainer = WeatherModelTrainer()
                weather_result = weather_trainer.train(valid_tracks)
                if weather_result.get('status') == 'success':
                    models["weather_impact"] = weather_result
                    self.logger.info(f"✅ Weather model trained: {weather_result.get('test_score', 'N/A')} score")
                else:
                    models["weather_impact"] = {"error": weather_result.get('error', 'Unknown error')}
                    self.logger.error(f"❌ Weather model training failed: {weather_result.get('error')}")
            except Exception as e:
                self.logger.error(f"❌ Weather model training exception: {e}")
                models["weather_impact"] = {"error": str(e)}
        else:
            self.logger.warning("⚠️ Insufficient tracks for weather model (need >= 2)")

        self._save_models(models)
        self.logger.info(f"✅ Training completed: {len([m for m in models.values() if m.get('status') == 'success'])} successful models")
        return models

    # -------------------------------
    # DATA VALIDATION - UPDATED: More flexible validation that accepts any pit data structure
    # -------------------------------
    def _validate_track_data(self, track_data: Dict[str, pd.DataFrame]) -> bool:
        """Validate that track data has minimum required data for training"""
        if not track_data:
            return False
            
        pit_data = track_data.get('pit_data', pd.DataFrame())
        
        # UPDATED: Only check if pit_data exists and has some data, not specific columns
        if pit_data.empty:
            self.logger.warning(f"⚠️ No pit data available")
            return False
            
        # UPDATED: Check for minimum data volume only
        if len(pit_data) < 3:  # Reduced threshold from 5 to 3
            self.logger.warning(f"⚠️ Insufficient pit data: {len(pit_data)} rows")
            return False
            
        # UPDATED: Log available columns for debugging
        self.logger.info(f"📋 Pit data columns available: {list(pit_data.columns)}")
            
        return True

    # -------------------------------
    # MODEL & STATE MANAGEMENT
    # -------------------------------
    def _load_training_state(self) -> Dict:
        state_file = os.path.join(self.training_state_dir, "training_state.pkl")
        if os.path.exists(state_file):
            try: 
                return joblib.load(state_file)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load training state: {e}")
        return {"processed_tracks": [], "models": {}}

    def _update_training_state(self, updates: Dict):
        state_file = os.path.join(self.training_state_dir, "training_state.pkl")
        state = self._load_training_state()
        state.update(updates)
        try: 
            joblib.dump(state, state_file)
        except Exception as e:
            self.logger.error(f"❌ Failed to save training state: {e}")

    def _load_processed_data_state(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        processed = {}
        state = self._load_training_state()
        for track in state.get("processed_data_keys", []):
            cache_file = os.path.join(self.training_state_dir, f"{track}_processed.pkl")
            if os.path.exists(cache_file):
                try: 
                    processed[track] = joblib.load(cache_file)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load cached data for {track}: {e}")
                    continue
        return processed

    def _save_track_data(self, track: str, data: Dict[str, pd.DataFrame]):
        try:
            cache_file = os.path.join(self.training_state_dir, f"{track}_processed.pkl")
            joblib.dump(data, cache_file)
        except Exception as e:
            self.logger.error(f"❌ Failed to save track data for {track}: {e}")

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
                self.logger.info(f"💾 Saved {name} model to {filepath}")
            except Exception as e:
                self.logger.error(f"❌ Failed to save {name} model: {e}")
        return saved

    def _load_existing_models(self) -> Dict:
        models = {}
        if not os.path.exists(self.models_output_dir):
            return models
            
        for file in os.listdir(self.models_output_dir):
            if file.endswith("_model.pkl"):
                name = file.replace("_model.pkl", "")
                try: 
                    models[name] = joblib.load(os.path.join(self.models_output_dir, file))
                    self.logger.info(f"📂 Loaded existing model: {name}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to load model {file}: {e}")
                    continue
        return models

    def _finalize_training(self) -> Dict:
        models = self._load_existing_models()
        self._update_training_state({"completed": True})
        self.logger.info("🎉 Training pipeline completed successfully!")
        return models

    def _upload_models_to_firebase(self, models: Dict[str, Any]):
        try:
            if hasattr(self.storage, "upload_models_to_firebase"):
                success = self.storage.upload_models_to_firebase()
                if success: 
                    self.logger.info("🚀 Models uploaded to Firebase Storage")
                else:
                    self.logger.error("❌ Failed to upload models to Firebase")
        except Exception as e:
            self.logger.error(f"❌ Firebase upload failed: {e}")

    # -------------------------------
    # LOGGING & VALIDATION - Updated for FirebaseDataLoader structure
    # -------------------------------
    def _log_data_quality(self, track: str, data: Dict[str, pd.DataFrame]):
        """Log data quality report for a track"""
        report = {}
        for data_type, df in data.items():
            if df.empty:
                report[data_type] = "EMPTY"
            else:
                null_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0
                report[data_type] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "null_pct": f"{null_pct:.1f}%"
                }
        
        self.logger.info(f"📊 {track} data quality: {report}")

    def validate_training_results(self, models: Dict) -> Dict[str, Any]:
        """Validate and report on training results"""
        results = {}
        
        for name, result in models.items():
            if isinstance(result, dict):
                if "error" in result:
                    results[name] = {"status": "FAILED", "error": result["error"]}
                elif result.get('status') == 'success':
                    # Model-specific validation
                    if name == "pit_strategy" and "accuracy" in result:
                        acc = result["accuracy"]
                        status = "GOOD" if acc > 0.8 else "FAIR" if acc > 0.6 else "POOR"
                        results[name] = {
                            "status": status, 
                            "accuracy": acc, 
                            "training_samples": result.get("training_samples", 0),
                            "tracks_used": result.get("tracks_used", 0)
                        }
                    elif "test_score" in result:
                        score = result["test_score"]
                        status = "GOOD" if score > 0.7 else "FAIR" if score > 0.5 else "POOR"
                        results[name] = {
                            "status": status, 
                            "test_score": score, 
                            "training_samples": result.get("training_samples", 0),
                            "tracks_used": result.get("tracks_used", 0)
                        }
                    else:
                        results[name] = {"status": "SUCCESS", "details": "Model trained successfully"}
                else:
                    results[name] = {"status": "UNKNOWN", "result": result}
            else:
                results[name] = {"status": "UNKNOWN", "result": type(result).__name__}
                
        # Print summary
        self._print_validation_summary(results)
        return results

    def _print_validation_summary(self, validation_results: Dict[str, Any]):
        """Print formatted validation summary"""
        print("\n" + "="*70)
        print("TRAINING RESULTS VALIDATION SUMMARY")
        print("="*70)
        
        successful_models = 0
        total_models = len(validation_results)
        
        for model_name, result in validation_results.items():
            status = result.get('status', 'UNKNOWN')
            status_icon = "✅" if status in ['SUCCESS', 'GOOD', 'FAIR'] else "❌" if status == 'FAILED' else "⚠️"
            
            if status in ['GOOD', 'FAIR', 'SUCCESS']:
                successful_models += 1
                
            details = []
            if 'accuracy' in result:
                details.append(f"Accuracy: {result['accuracy']:.3f}")
            if 'test_score' in result:
                details.append(f"R² Score: {result['test_score']:.3f}")
            if 'training_samples' in result:
                details.append(f"Samples: {result['training_samples']}")
            if 'tracks_used' in result:
                details.append(f"Tracks: {result['tracks_used']}")
                
            detail_str = " | ".join(details) if details else result.get('error', 'No details')
            
            print(f"{status_icon} {model_name:.<20} {status:.<10} {detail_str}")
        
        success_rate = (successful_models / total_models * 100) if total_models > 0 else 0
        print(f"\n📊 Overall: {successful_models}/{total_models} models successful ({success_rate:.1f}%)")

    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress status"""
        state = self._load_training_state()
        available_tracks = self.storage.list_available_tracks()
        processed_tracks = state.get("processed_tracks", [])
        
        return {
            "total_tracks": len(available_tracks),
            "processed_tracks": len(processed_tracks),
            "remaining_tracks": len(available_tracks) - len(processed_tracks),
            "completion_percentage": (len(processed_tracks) / len(available_tracks) * 100) if available_tracks else 0,
            "trained_models": state.get("models", []),
            "completed": state.get("completed", False),
            "last_updated": datetime.now().isoformat()
        }
