# import os
# import logging
# import joblib
# import pandas as pd
# from typing import Dict, Any, List, Optional
# from datetime import datetime

# from models.tire_trainer import TireModelTrainer
# from models.fuel_trainer import FuelModelTrainer
# from models.pit_strategy_trainer import PitStrategyTrainer
# from models.weather_trainer import WeatherModelTrainer
# from data.preprocessor import DataPreprocessor
# from data.feature_engineer import FeatureEngineer
# from data.firebase_loader import FirebaseDataLoader


# class TrainingOrchestrator:
#     def __init__(self, storage: FirebaseDataLoader):
#         self.storage = storage
#         self.logger = logging.getLogger(__name__)
#         self.models_output_dir = "outputs/models"
#         self.training_state_dir = "outputs/training_state"
#         os.makedirs(self.models_output_dir, exist_ok=True)
#         os.makedirs(self.training_state_dir, exist_ok=True)

#     # -------------------------------
#     # MAIN TRAINING PIPELINE - Enhanced with comprehensive error handling
#     # -------------------------------
#     def train_all_models(self) -> dict:
#         """Enhanced training pipeline with robust error recovery and validation"""
#         self.logger.info("🚀 Starting robust model training pipeline...")

#         try:
#             state = self._load_training_state()
#             if state.get("completed", False):
#                 self.logger.info("✅ Training already completed. Loading models...")
#                 return self._load_existing_models()

#             available_tracks = self.storage.list_available_tracks()
#             if not available_tracks:
#                 self.logger.error("❌ No tracks available in storage")
#                 return self._create_fallback_models("No tracks available")

#             processed_tracks = state.get("processed_tracks", [])
#             remaining_tracks = [t for t in available_tracks if t not in processed_tracks]

#             if not remaining_tracks:
#                 self.logger.info("📊 All tracks processed. Finalizing models...")
#                 return self._finalize_training()

#             self.logger.info(f"📥 Processing {len(remaining_tracks)} tracks: {remaining_tracks}")
#             all_processed_data = self._load_processed_data_state()

#             # Enhanced track processing with comprehensive error handling
#             successful_tracks = self._process_tracks_batch(remaining_tracks, all_processed_data, processed_tracks)

#             if successful_tracks:
#                 models = self._train_models_with_processed_data(all_processed_data, state.get("models", {}))
#                 self._upload_models_to_firebase(models)
#                 self._update_training_state({
#                     "processed_tracks": processed_tracks,
#                     "processed_data_keys": list(all_processed_data.keys()),
#                     "models": list(models.keys()),
#                     "completed": len(processed_tracks) == len(available_tracks)
#                 })
#                 return models
#             else:
#                 self.logger.error("❌ No valid data processed for training")
#                 return self._create_fallback_models("No valid tracks processed")

#         except Exception as e:
#             self.logger.error(f"❌ Training pipeline failed: {e}")
#             return self._create_fallback_models(f"Pipeline error: {str(e)}")

#     def _process_tracks_batch(self, tracks: List[str], all_processed_data: Dict, processed_tracks: List[str]) -> int:
#         """Process multiple tracks with enhanced error handling and recovery"""
#         successful_tracks = 0
        
#         for idx, track in enumerate(tracks, 1):
#             try:
#                 self.logger.info(f"🔄 Processing track {idx}/{len(tracks)}: {track}")
#                 track_data = self._process_single_track(track)
                
#                 if self._validate_track_data_quality(track_data):
#                     all_processed_data[track] = track_data
#                     processed_tracks.append(track)
#                     successful_tracks += 1
                    
#                     self._update_training_state({
#                         "processed_tracks": processed_tracks,
#                         "processed_data_keys": list(all_processed_data.keys())
#                     })
#                     self._save_track_data(track, track_data)
#                     self.logger.info(f"✅ Track processed successfully: {track}")
#                 else:
#                     self.logger.warning(f"⚠️ Skipped {track}: insufficient data quality")
                    
#             except Exception as e:
#                 self.logger.error(f"❌ Failed to process {track}: {e}")
#                 continue

#         return successful_tracks

#     # -------------------------------
#     # SINGLE TRACK PROCESSING - Enhanced with comprehensive data validation
#     # -------------------------------
#     def _process_single_track(self, track: str) -> Dict[str, pd.DataFrame]:
#         """Enhanced single track processing with comprehensive caching and validation"""
#         cache_file = os.path.join(self.training_state_dir, f"{track}_processed.pkl")
        
#         # Try to load from cache first
#         if os.path.exists(cache_file):
#             self.logger.info(f"📂 Loading cached processed data for {track}")
#             try:
#                 cached_data = joblib.load(cache_file)
#                 if self._validate_cached_data(cached_data):
#                     return cached_data
#                 else:
#                     self.logger.warning(f"⚠️ Cached data invalid. Reprocessing {track}")
#             except Exception as e:
#                 self.logger.warning(f"⚠️ Cached file corrupted. Reprocessing {track}: {e}")

#         # Load and process track data
#         try:
#             # Load track data using FirebaseDataLoader
#             track_raw = self.storage.load_track_data(track)
#             self.logger.info(f"📥 Loaded raw data for {track}: {len(track_raw)} data types")
            
#             # Validate raw data before preprocessing
#             if not self._validate_raw_track_data(track_raw):
#                 self.logger.warning(f"⚠️ Raw data validation failed for {track}")
#                 return {}

#             # Preprocess using DataPreprocessor
#             preprocessor = DataPreprocessor()
#             processed = preprocessor.preprocess_track_data(track_raw)
#             self.logger.info(f"🔧 Preprocessed {track}: {len(processed)} data types")
            
#             # Validate processed data
#             if not self._validate_processed_data(processed):
#                 self.logger.warning(f"⚠️ Processed data validation failed for {track}")
#                 return {}

#             # Engineer features using FeatureEngineer
#             feature_engineer = FeatureEngineer()
#             enhanced_data = feature_engineer.create_composite_features({track: processed})
            
#             track_processed = enhanced_data.get(track, processed)
            
#             # Final validation before caching
#             if self._validate_enhanced_data(track_processed):
#                 try:
#                     joblib.dump(track_processed, cache_file)
#                     self.logger.info(f"💾 Cached processed data for {track}")
#                 except Exception as e:
#                     self.logger.warning(f"⚠️ Failed to cache processed data for {track}: {e}")
                
#                 self._log_data_quality(track, track_processed)
#                 return track_processed
#             else:
#                 self.logger.warning(f"⚠️ Enhanced data validation failed for {track}")
#                 return {}
                
#         except Exception as e:
#             self.logger.error(f"❌ Track processing failed for {track}: {e}")
#             return {}

#     def _validate_raw_track_data(self, track_data: Dict[str, pd.DataFrame]) -> bool:
#         """Validate raw track data before preprocessing"""
#         if not track_data:
#             return False
            
#         # Check for minimum required data types
#         required_types = ['pit_data', 'race_data']
#         available_types = [k for k, v in track_data.items() if not v.empty]
        
#         if not any(t in available_types for t in required_types):
#             self.logger.debug(f"⚠️ Missing required data types. Available: {available_types}")
#             return False
            
#         # Check pit data specifically
#         pit_data = track_data.get('pit_data', pd.DataFrame())
#         if pit_data.empty:
#             self.logger.debug("⚠️ No pit data available")
#             return False
            
#         return True

#     def _validate_processed_data(self, processed_data: Dict[str, pd.DataFrame]) -> bool:
#         """Validate processed data before feature engineering"""
#         if not processed_data:
#             return False
            
#         pit_data = processed_data.get('pit_data', pd.DataFrame())
#         if pit_data.empty:
#             return False
            
#         # Check for minimum required processed columns
#         if 'LAP_NUMBER' not in pit_data.columns or 'NUMBER' not in pit_data.columns:
#             self.logger.debug("⚠️ Missing required processed columns")
#             return False
            
#         return True

#     def _validate_enhanced_data(self, enhanced_data: Dict[str, pd.DataFrame]) -> bool:
#         """Validate enhanced data after feature engineering"""
#         if not enhanced_data:
#             return False
            
#         pit_data = enhanced_data.get('pit_data', pd.DataFrame())
#         if pit_data.empty:
#             return False
            
#         # Check for engineered features
#         engineered_features = [col for col in pit_data.columns if any(keyword in col for keyword in 
#                                                                      ['DEGRADATION', 'CONSISTENCY', 'EFFICIENCY'])]
#         if not engineered_features:
#             self.logger.debug("⚠️ No engineered features found")
            
#         return len(pit_data) >= 3  # Minimum data requirement

#     def _validate_cached_data(self, cached_data: Dict[str, pd.DataFrame]) -> bool:
#         """Validate cached data before using it"""
#         if not cached_data:
#             return False
            
#         if not isinstance(cached_data, dict):
#             return False
            
#         pit_data = cached_data.get('pit_data', pd.DataFrame())
#         if pit_data.empty:
#             return False
            
#         # Check data freshness (optional - could check timestamps if available)
#         return True

#     def _validate_track_data_quality(self, track_data: Dict[str, pd.DataFrame]) -> bool:
#         """Comprehensive track data quality validation"""
#         if not track_data:
#             return False
            
#         pit_data = track_data.get('pit_data', pd.DataFrame())
        
#         # Basic existence check
#         if pit_data.empty:
#             self.logger.warning(f"⚠️ No pit data available")
#             return False
            
#         # Data volume check
#         if len(pit_data) < 3:
#             self.logger.warning(f"⚠️ Insufficient pit data: {len(pit_data)} rows")
#             return False
            
#         # Data quality metrics
#         null_percentage = (pit_data.isnull().sum().sum() / (len(pit_data) * len(pit_data.columns))) * 100
#         if null_percentage > 50:  # Allow up to 50% nulls
#             self.logger.warning(f"⚠️ High null percentage: {null_percentage:.1f}%")
#             return False
            
#         # Check for critical columns
#         critical_columns = ['NUMBER', 'LAP_NUMBER']
#         missing_critical = [col for col in critical_columns if col not in pit_data.columns]
#         if missing_critical:
#             self.logger.warning(f"⚠️ Missing critical columns: {missing_critical}")
#             return False
            
#         self.logger.info(f"📋 Track data quality: {len(pit_data)} rows, {len(pit_data.columns)} cols, {null_percentage:.1f}% nulls")
#         return True

#     # -------------------------------
#     # MODEL TRAINING - Enhanced with comprehensive error handling and fallbacks
#     # -------------------------------
#     def _train_models_with_processed_data(self, processed_data: Dict[str, Dict[str, pd.DataFrame]], existing_models: Dict = None) -> Dict[str, Any]:
#         """Enhanced model training with comprehensive error handling and fallbacks"""
#         self.logger.info("🏃 Training models with processed data...")
#         models = existing_models or {}

#         # Enhanced track validation with detailed logging
#         valid_tracks = {}
#         for track_name, data_dict in processed_data.items():
#             if self._validate_track_data_quality(data_dict):
#                 valid_tracks[track_name] = data_dict
#                 self.logger.info(f"✅ Valid track: {track_name} ({len(data_dict.get('pit_data', pd.DataFrame()))} pit records)")
#             else:
#                 self.logger.warning(f"⚠️ Invalid track skipped: {track_name}")

#         if not valid_tracks:
#             self.logger.error("❌ No valid tracks with quality data for training")
#             return self._create_fallback_models("No valid tracks available")

#         self.logger.info(f"📊 Training with {len(valid_tracks)} valid tracks: {list(valid_tracks.keys())}")

#         # Train all models with enhanced error handling
#         models.update(self._train_tire_model(valid_tracks))
#         models.update(self._train_fuel_model(valid_tracks))
#         models.update(self._train_pit_strategy_model(valid_tracks))
#         models.update(self._train_weather_model(valid_tracks))

#         # Save models and validate results
#         self._save_models(models)
#         self._validate_training_results(models)
        
#         successful_models = len([m for m in models.values() if isinstance(m, dict) and m.get('status') == 'success'])
#         self.logger.info(f"✅ Training completed: {successful_models}/{len(models)} successful models")
        
#         return models

#     def _train_tire_model(self, valid_tracks: Dict) -> Dict[str, Any]:
#         """Train tire model with enhanced error handling"""
#         try:
#             tire_trainer = TireModelTrainer()
#             tire_result = tire_trainer.train(valid_tracks)
            
#             if tire_result.get('status') == 'success':
#                 self.logger.info(f"✅ Tire model trained: {tire_result.get('test_score', 'N/A')} score")
#                 return {"tire_degradation": tire_result}
#             else:
#                 error_msg = tire_result.get('error', 'Unknown error')
#                 self.logger.error(f"❌ Tire model training failed: {error_msg}")
#                 return {"tire_degradation": {"error": error_msg, "status": "failed"}}
                
#         except Exception as e:
#             self.logger.error(f"❌ Tire model training exception: {e}")
#             return {"tire_degradation": {"error": str(e), "status": "error"}}

#     def _train_fuel_model(self, valid_tracks: Dict) -> Dict[str, Any]:
#         """Train fuel model with enhanced error handling"""
#         try:
#             fuel_trainer = FuelModelTrainer()
#             fuel_result = fuel_trainer.train(valid_tracks)
            
#             if fuel_result.get('status') == 'success':
#                 self.logger.info(f"✅ Fuel model trained: {fuel_result.get('test_score', 'N/A')} score")
#                 return {"fuel_consumption": fuel_result}
#             else:
#                 error_msg = fuel_result.get('error', 'Unknown error')
#                 self.logger.error(f"❌ Fuel model training failed: {error_msg}")
#                 return {"fuel_consumption": {"error": error_msg, "status": "failed"}}
                
#         except Exception as e:
#             self.logger.error(f"❌ Fuel model training exception: {e}")
#             return {"fuel_consumption": {"error": str(e), "status": "error"}}

#     def _train_pit_strategy_model(self, valid_tracks: Dict) -> Dict[str, Any]:
#         """Train pit strategy model with enhanced error handling"""
#         if len(valid_tracks) < 2:
#             self.logger.warning("⚠️ Insufficient tracks for pit strategy model (need >= 2)")
#             return {"pit_strategy": {"error": "Insufficient tracks", "status": "skipped"}}
            
#         try:
#             pit_trainer = PitStrategyTrainer()
#             pit_result = pit_trainer.train(valid_tracks)
            
#             if pit_result.get('status') == 'success':
#                 self.logger.info(f"✅ Pit strategy model trained: {pit_result.get('accuracy', 'N/A')} accuracy")
#                 return {"pit_strategy": pit_result}
#             else:
#                 error_msg = pit_result.get('error', 'Unknown error')
#                 self.logger.error(f"❌ Pit strategy model training failed: {error_msg}")
#                 return {"pit_strategy": {"error": error_msg, "status": "failed"}}
                
#         except Exception as e:
#             self.logger.error(f"❌ Pit strategy model training exception: {e}")
#             return {"pit_strategy": {"error": str(e), "status": "error"}}

#     def _train_weather_model(self, valid_tracks: Dict) -> Dict[str, Any]:
#         """Train weather model with enhanced error handling"""
#         if len(valid_tracks) < 2:
#             self.logger.warning("⚠️ Insufficient tracks for weather model (need >= 2)")
#             return {"weather_impact": {"error": "Insufficient tracks", "status": "skipped"}}
            
#         try:
#             weather_trainer = WeatherModelTrainer()
#             weather_result = weather_trainer.train(valid_tracks)
            
#             if weather_result.get('status') == 'success':
#                 self.logger.info(f"✅ Weather model trained: {weather_result.get('test_score', 'N/A')} score")
#                 return {"weather_impact": weather_result}
#             else:
#                 error_msg = weather_result.get('error', 'Unknown error')
#                 self.logger.error(f"❌ Weather model training failed: {error_msg}")
#                 return {"weather_impact": {"error": error_msg, "status": "failed"}}
                
#         except Exception as e:
#             self.logger.error(f"❌ Weather model training exception: {e}")
#             return {"weather_impact": {"error": str(e), "status": "error"}}

#     def _create_fallback_models(self, reason: str) -> Dict[str, Any]:
#         """Create fallback models when training fails"""
#         self.logger.warning(f"⚠️ Creating fallback models due to: {reason}")
        
#         return {
#             "tire_degradation": {"status": "fallback", "fallback_reason": reason},
#             "fuel_consumption": {"status": "fallback", "fallback_reason": reason},
#             "pit_strategy": {"status": "fallback", "fallback_reason": reason},
#             "weather_impact": {"status": "fallback", "fallback_reason": reason}
#         }

#     # -------------------------------
#     # MODEL & STATE MANAGEMENT - Enhanced with better error recovery
#     # -------------------------------
#     def _load_training_state(self) -> Dict:
#         """Load training state with enhanced error recovery"""
#         state_file = os.path.join(self.training_state_dir, "training_state.pkl")
#         if os.path.exists(state_file):
#             try: 
#                 state = joblib.load(state_file)
#                 if isinstance(state, dict):
#                     return state
#                 else:
#                     self.logger.warning("⚠️ Training state file corrupted, creating new state")
#             except Exception as e:
#                 self.logger.warning(f"⚠️ Failed to load training state: {e}")
        
#         # Return default state
#         return {
#             "processed_tracks": [],
#             "models": {},
#             "processed_data_keys": [],
#             "completed": False
#         }

#     def _update_training_state(self, updates: Dict):
#         """Update training state with enhanced error handling"""
#         state_file = os.path.join(self.training_state_dir, "training_state.pkl")
#         state = self._load_training_state()
#         state.update(updates)
        
#         try: 
#             joblib.dump(state, state_file)
#             self.logger.debug("💾 Training state updated successfully")
#         except Exception as e:
#             self.logger.error(f"❌ Failed to save training state: {e}")

#     def _load_processed_data_state(self) -> Dict[str, Dict[str, pd.DataFrame]]:
#         """Load processed data state with validation"""
#         processed = {}
#         state = self._load_training_state()
        
#         for track in state.get("processed_data_keys", []):
#             cache_file = os.path.join(self.training_state_dir, f"{track}_processed.pkl")
#             if os.path.exists(cache_file):
#                 try: 
#                     track_data = joblib.load(cache_file)
#                     if self._validate_cached_data(track_data):
#                         processed[track] = track_data
#                     else:
#                         self.logger.warning(f"⚠️ Cached data invalid for {track}, skipping")
#                 except Exception as e:
#                     self.logger.warning(f"⚠️ Failed to load cached data for {track}: {e}")
#                     continue
                    
#         self.logger.info(f"📂 Loaded {len(processed)} tracks from cache")
#         return processed

#     def _save_track_data(self, track: str, data: Dict[str, pd.DataFrame]):
#         """Save track data with enhanced error handling"""
#         try:
#             cache_file = os.path.join(self.training_state_dir, f"{track}_processed.pkl")
#             joblib.dump(data, cache_file)
#             self.logger.debug(f"💾 Saved track data for {track}")
#         except Exception as e:
#             self.logger.error(f"❌ Failed to save track data for {track}: {e}")

#     def _save_models(self, models: Dict[str, Any]) -> Dict[str, str]:
#         """Save models with enhanced error handling"""
#         saved = {}
#         for name, result in models.items():
#             try:
#                 filepath = os.path.join(self.models_output_dir, f"{name}_model.pkl")
                
#                 # Handle different model types
#                 if isinstance(result, dict) and "model" in result and hasattr(result["model"], "save_model"):
#                     result["model"].save_model(filepath)
#                 elif isinstance(result, dict):
#                     joblib.dump(result, filepath)
#                 else:
#                     joblib.dump({"model": result}, filepath)
                    
#                 saved[name] = filepath
#                 self.logger.info(f"💾 Saved {name} model to {filepath}")
                
#             except Exception as e:
#                 self.logger.error(f"❌ Failed to save {name} model: {e}")
                
#         return saved

#     def _load_existing_models(self) -> Dict:
#         """Load existing models with enhanced validation"""
#         models = {}
#         if not os.path.exists(self.models_output_dir):
#             return models
            
#         for file in os.listdir(self.models_output_dir):
#             if file.endswith("_model.pkl"):
#                 name = file.replace("_model.pkl", "")
#                 try: 
#                     model_data = joblib.load(os.path.join(self.models_output_dir, file))
#                     if isinstance(model_data, dict) and model_data.get('status') in ['success', 'fallback']:
#                         models[name] = model_data
#                         self.logger.info(f"📂 Loaded existing model: {name}")
#                     else:
#                         self.logger.warning(f"⚠️ Invalid model file: {file}")
#                 except Exception as e:
#                     self.logger.error(f"❌ Failed to load model {file}: {e}")
#                     continue
                    
#         return models

#     def _finalize_training(self) -> Dict:
#         """Finalize training with comprehensive cleanup"""
#         models = self._load_existing_models()
#         self._update_training_state({"completed": True})
        
#         # Log final statistics
#         successful_models = len([m for m in models.values() if isinstance(m, dict) and m.get('status') == 'success'])
#         self.logger.info(f"🎉 Training pipeline completed! {successful_models}/{len(models)} models successful")
        
#         return models

#     def _upload_models_to_firebase(self, models: Dict[str, Any]):
#         """Upload models to Firebase with enhanced error handling"""
#         try:
#             if hasattr(self.storage, "upload_models_to_firebase"):
#                 success = self.storage.upload_models_to_firebase()
#                 if success: 
#                     self.logger.info("🚀 Models uploaded to Firebase Storage")
#                 else:
#                     self.logger.error("❌ Failed to upload models to Firebase")
#             else:
#                 self.logger.warning("⚠️ Firebase upload not available in storage")
#         except Exception as e:
#             self.logger.error(f"❌ Firebase upload failed: {e}")

#     # -------------------------------
#     # LOGGING & VALIDATION - Enhanced with comprehensive reporting
#     # -------------------------------
#     def _log_data_quality(self, track: str, data: Dict[str, pd.DataFrame]):
#         """Enhanced data quality reporting"""
#         report = {}
#         for data_type, df in data.items():
#             if df.empty:
#                 report[data_type] = "EMPTY"
#             else:
#                 null_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0
#                 engineered_features = len([col for col in df.columns if any(keyword in col for keyword in 
#                                                                           ['DEGRADATION', 'CONSISTENCY', 'EFFICIENCY', 'IMPACT'])])
#                 report[data_type] = {
#                     "rows": len(df),
#                     "columns": len(df.columns),
#                     "null_pct": f"{null_pct:.1f}%",
#                     "engineered_features": engineered_features
#                 }
        
#         self.logger.info(f"📊 {track} data quality: {report}")

#     def _validate_training_results(self, models: Dict[str, Any]):
#         """Validate training results and log comprehensive report"""
#         results = self.validate_training_results(models)
        
#         # Log detailed results
#         for model_name, result in results.items():
#             status = result.get('status', 'UNKNOWN')
#             if status in ['GOOD', 'FAIR', 'SUCCESS']:
#                 self.logger.info(f"✅ {model_name}: {status}")
#             elif status == 'FAILED':
#                 self.logger.error(f"❌ {model_name}: FAILED - {result.get('error', 'Unknown error')}")
#             else:
#                 self.logger.warning(f"⚠️ {model_name}: {status}")

#     def validate_training_results(self, models: Dict) -> Dict[str, Any]:
#         """Comprehensive training results validation"""
#         results = {}
        
#         for name, result in models.items():
#             if isinstance(result, dict):
#                 if "error" in result:
#                     results[name] = {"status": "FAILED", "error": result["error"]}
#                 elif result.get('status') == 'success':
#                     # Model-specific validation with enhanced metrics
#                     validation_data = {
#                         "status": "SUCCESS",
#                         "training_samples": result.get("training_samples", 0),
#                         "tracks_used": result.get("tracks_used", 0),
#                         "processed_tracks": result.get("processed_tracks", [])
#                     }
                    
#                     # Add performance metrics
#                     if name == "pit_strategy" and "accuracy" in result:
#                         acc = result["accuracy"]
#                         validation_data["accuracy"] = acc
#                         validation_data["performance"] = "GOOD" if acc > 0.8 else "FAIR" if acc > 0.6 else "POOR"
#                     elif "test_score" in result:
#                         score = result["test_score"]
#                         validation_data["test_score"] = score
#                         validation_data["performance"] = "GOOD" if score > 0.7 else "FAIR" if score > 0.5 else "POOR"
#                     elif "train_score" in result:
#                         score = result["train_score"]
#                         validation_data["train_score"] = score
#                         validation_data["performance"] = "GOOD" if score > 0.7 else "FAIR" if score > 0.5 else "POOR"
                        
#                     results[name] = validation_data
#                 elif result.get('status') == 'fallback':
#                     results[name] = {"status": "FALLBACK", "reason": result.get('fallback_reason', 'Unknown')}
#                 else:
#                     results[name] = {"status": "UNKNOWN", "result": result}
#             else:
#                 results[name] = {"status": "UNKNOWN", "result": type(result).__name__}
                
#         # Print comprehensive summary
#         self._print_validation_summary(results)
#         return results

#     def _print_validation_summary(self, validation_results: Dict[str, Any]):
#         """Enhanced validation summary with detailed reporting"""
#         print("\n" + "="*70)
#         print("TRAINING RESULTS VALIDATION SUMMARY")
#         print("="*70)
        
#         successful_models = 0
#         total_models = len(validation_results)
        
#         for model_name, result in validation_results.items():
#             status = result.get('status', 'UNKNOWN')
            
#             # Determine status icon
#             if status in ['SUCCESS', 'GOOD', 'FAIR']:
#                 status_icon = "✅"
#                 successful_models += 1
#             elif status == 'FAILED':
#                 status_icon = "❌"
#             elif status == 'FALLBACK':
#                 status_icon = "🔄"
#             else:
#                 status_icon = "⚠️"
                
#             # Build details string
#             details = []
#             if 'accuracy' in result:
#                 details.append(f"Accuracy: {result['accuracy']:.3f}")
#             if 'test_score' in result:
#                 details.append(f"R² Score: {result['test_score']:.3f}")
#             if 'train_score' in result:
#                 details.append(f"Train Score: {result['train_score']:.3f}")
#             if 'performance' in result:
#                 details.append(f"Performance: {result['performance']}")
#             if 'training_samples' in result:
#                 details.append(f"Samples: {result['training_samples']}")
#             if 'tracks_used' in result:
#                 details.append(f"Tracks: {result['tracks_used']}")
#             if 'error' in result:
#                 details.append(f"Error: {result['error']}")
#             if 'reason' in result:
#                 details.append(f"Reason: {result['reason']}")
                
#             detail_str = " | ".join(details) if details else 'No details available'
            
#             print(f"{status_icon} {model_name:.<20} {status:.<10} {detail_str}")
        
#         success_rate = (successful_models / total_models * 100) if total_models > 0 else 0
#         print(f"\n📊 Overall: {successful_models}/{total_models} models successful ({success_rate:.1f}%)")

#     def get_training_progress(self) -> Dict[str, Any]:
#         """Get comprehensive training progress status"""
#         state = self._load_training_state()
#         available_tracks = self.storage.list_available_tracks()
#         processed_tracks = state.get("processed_tracks", [])
        
#         # Calculate detailed progress
#         total_tracks = len(available_tracks)
#         processed_count = len(processed_tracks)
#         completion_pct = (processed_count / total_tracks * 100) if total_tracks else 0
        
#         # Model status
#         models_status = {}
#         for model_name in ['tire_degradation', 'fuel_consumption', 'pit_strategy', 'weather_impact']:
#             model_data = state.get('models', {}).get(model_name, {})
#             if isinstance(model_data, dict):
#                 models_status[model_name] = model_data.get('status', 'unknown')
#             else:
#                 models_status[model_name] = 'unknown'
        
#         return {
#             "total_tracks": total_tracks,
#             "processed_tracks": processed_count,
#             "remaining_tracks": total_tracks - processed_count,
#             "completion_percentage": completion_pct,
#             "models_status": models_status,
#             "completed": state.get("completed", False),
#             "last_updated": datetime.now().isoformat(),
#             "available_tracks": available_tracks,
#             "processed_track_names": processed_tracks
#         }
























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























# import os
# import logging
# import joblib
# import pandas as pd
# from typing import Dict, Any, List, Optional
# from datetime import datetime

# from models.tire_trainer import TireModelTrainer
# from models.fuel_trainer import FuelModelTrainer
# from models.pit_strategy_trainer import PitStrategyTrainer
# from models.weather_trainer import WeatherModelTrainer
# from data.preprocessor import DataPreprocessor
# from data.feature_engineer import FeatureEngineer
# from data.firebase_loader import FirebaseDataLoader


# class TrainingOrchestrator:
#     def __init__(self, storage: FirebaseDataLoader):
#         self.storage = storage
#         self.logger = logging.getLogger(__name__)
#         self.models_output_dir = "outputs/models"
#         self.training_state_dir = "outputs/training_state"
#         os.makedirs(self.models_output_dir, exist_ok=True)
#         os.makedirs(self.training_state_dir, exist_ok=True)

#     # -------------------------------
#     # MAIN TRAINING PIPELINE - Updated for FirebaseDataLoader consistency
#     # -------------------------------
#     def train_all_models(self) -> dict:
#         self.logger.info("🚀 Starting robust model training pipeline...")

#         state = self._load_training_state()
#         if state.get("completed", False):
#             self.logger.info("✅ Training already completed. Loading models...")
#             return self._load_existing_models()

#         available_tracks = self.storage.list_available_tracks()
#         if not available_tracks:
#             self.logger.error("❌ No tracks available in storage")
#             return {}

#         processed_tracks = state.get("processed_tracks", [])
#         remaining_tracks = [t for t in available_tracks if t not in processed_tracks]

#         if not remaining_tracks:
#             self.logger.info("📊 All tracks processed. Finalizing models...")
#             return self._finalize_training()

#         self.logger.info(f"📥 Processing {len(remaining_tracks)} tracks: {remaining_tracks}")
#         all_processed_data = self._load_processed_data_state()

#         for idx, track in enumerate(remaining_tracks, 1):
#             try:
#                 self.logger.info(f"🔄 Processing track {idx}/{len(remaining_tracks)}: {track}")
#                 track_data = self._process_single_track(track)
#                 if track_data:
#                     all_processed_data[track] = track_data
#                     processed_tracks.append(track)
#                     self._update_training_state({
#                         "processed_tracks": processed_tracks,
#                         "processed_data_keys": list(all_processed_data.keys())
#                     })
#                     self._save_track_data(track, track_data)
#                     self.logger.info(f"✅ Track processed: {track}")
#                 else:
#                     self.logger.warning(f"⚠️ Skipped {track}: insufficient data")
#             except Exception as e:
#                 self.logger.error(f"❌ Failed to process {track}: {e}")
#                 continue

#         if all_processed_data:
#             models = self._train_models_with_processed_data(all_processed_data, state.get("models", {}))
#             self._upload_models_to_firebase(models)
#             self._update_training_state({
#                 "processed_tracks": processed_tracks,
#                 "processed_data_keys": list(all_processed_data.keys()),
#                 "models": list(models.keys()),
#                 "completed": len(processed_tracks) == len(available_tracks)
#             })
#             return models
#         else:
#             self.logger.error("❌ No valid data processed for training")
#             return {}

#     # -------------------------------
#     # SINGLE TRACK PROCESSING - Updated for FirebaseDataLoader consistency
#     # -------------------------------
#     def _process_single_track(self, track: str) -> Dict[str, pd.DataFrame]:
#         cache_file = os.path.join(self.training_state_dir, f"{track}_processed.pkl")
#         if os.path.exists(cache_file):
#             self.logger.info(f"📂 Loading cached processed data for {track}")
#             try:
#                 return joblib.load(cache_file)
#             except Exception:
#                 self.logger.warning(f"⚠️ Cached file corrupted. Reprocessing {track}")

#         # Load track data using FirebaseDataLoader (returns dict with pit_data, race_data, etc.)
#         track_raw = self.storage.load_track_data(track)
        
#         # Preprocess using updated DataPreprocessor
#         preprocessor = DataPreprocessor()
#         processed = preprocessor.preprocess_track_data(track_raw)
        
#         # Engineer features using updated FeatureEngineer
#         feature_engineer = FeatureEngineer()
#         enhanced_data = feature_engineer.create_composite_features({track: processed})
        
#         track_processed = enhanced_data.get(track, processed)
#         self._log_data_quality(track, track_processed)

#         try:
#             joblib.dump(track_processed, cache_file)
#         except Exception as e:
#             self.logger.warning(f"⚠️ Failed to cache processed data for {track}: {e}")
            
#         return track_processed

#     # -------------------------------
#     # MODEL TRAINING - Updated for consistent method signatures
#     # -------------------------------
#     def _train_models_with_processed_data(self, processed_data: Dict[str, Dict[str, pd.DataFrame]], existing_models: Dict = None) -> Dict[str, Any]:
#         self.logger.info("🏃 Training models with processed data...")
#         models = existing_models or {}

#         # Filter out tracks with insufficient data
#         valid_tracks = {
#             track_name: data_dict 
#             for track_name, data_dict in processed_data.items() 
#             if not data_dict.get('pit_data', pd.DataFrame()).empty
#         }

#         if not valid_tracks:
#             self.logger.error("❌ No valid tracks with pit data for training")
#             return models

#         self.logger.info(f"📊 Training with {len(valid_tracks)} valid tracks: {list(valid_tracks.keys())}")

#         # Tire Model Training - Updated for new method signature
#         try:
#             tire_trainer = TireModelTrainer()
#             tire_result = tire_trainer.train(valid_tracks)
#             if tire_result.get('status') == 'success':
#                 models["tire_degradation"] = tire_result
#                 self.logger.info(f"✅ Tire model trained: {tire_result.get('test_score', 'N/A')} score")
#             else:
#                 models["tire_degradation"] = {"error": tire_result.get('error', 'Unknown error')}
#                 self.logger.error(f"❌ Tire model training failed: {tire_result.get('error')}")
#         except Exception as e:
#             self.logger.error(f"❌ Tire model training exception: {e}")
#             models["tire_degradation"] = {"error": str(e)}

#         # Fuel Model Training - Updated for new method signature
#         try:
#             fuel_trainer = FuelModelTrainer()
#             fuel_result = fuel_trainer.train(valid_tracks)
#             if fuel_result.get('status') == 'success':
#                 models["fuel_consumption"] = fuel_result
#                 self.logger.info(f"✅ Fuel model trained: {fuel_result.get('test_score', 'N/A')} score")
#             else:
#                 models["fuel_consumption"] = {"error": fuel_result.get('error', 'Unknown error')}
#                 self.logger.error(f"❌ Fuel model training failed: {fuel_result.get('error')}")
#         except Exception as e:
#             self.logger.error(f"❌ Fuel model training exception: {e}")
#             models["fuel_consumption"] = {"error": str(e)}

#         # Pit Strategy Model Training - Updated for new method signature
#         if len(valid_tracks) >= 2:
#             try:
#                 pit_trainer = PitStrategyTrainer()
#                 pit_result = pit_trainer.train(valid_tracks)
#                 if pit_result.get('status') == 'success':
#                     models["pit_strategy"] = pit_result
#                     self.logger.info(f"✅ Pit strategy model trained: {pit_result.get('accuracy', 'N/A')} accuracy")
#                 else:
#                     models["pit_strategy"] = {"error": pit_result.get('error', 'Unknown error')}
#                     self.logger.error(f"❌ Pit strategy model training failed: {pit_result.get('error')}")
#             except Exception as e:
#                 self.logger.error(f"❌ Pit strategy model training exception: {e}")
#                 models["pit_strategy"] = {"error": str(e)}
#         else:
#             self.logger.warning("⚠️ Insufficient tracks for pit strategy model (need >= 2)")

#         # Weather Model Training - Updated for new method signature
#         if len(valid_tracks) >= 2:
#             try:
#                 weather_trainer = WeatherModelTrainer()
#                 weather_result = weather_trainer.train(valid_tracks)
#                 if weather_result.get('status') == 'success':
#                     models["weather_impact"] = weather_result
#                     self.logger.info(f"✅ Weather model trained: {weather_result.get('test_score', 'N/A')} score")
#                 else:
#                     models["weather_impact"] = {"error": weather_result.get('error', 'Unknown error')}
#                     self.logger.error(f"❌ Weather model training failed: {weather_result.get('error')}")
#             except Exception as e:
#                 self.logger.error(f"❌ Weather model training exception: {e}")
#                 models["weather_impact"] = {"error": str(e)}
#         else:
#             self.logger.warning("⚠️ Insufficient tracks for weather model (need >= 2)")

#         self._save_models(models)
#         self.logger.info(f"✅ Training completed: {len([m for m in models.values() if m.get('status') == 'success'])} successful models")
#         return models

#     # -------------------------------
#     # DATA VALIDATION - Updated for FirebaseDataLoader structure
#     # -------------------------------
#     def _validate_track_data(self, track_data: Dict[str, pd.DataFrame]) -> bool:
#         """Validate that track data has minimum required data for training"""
#         if not track_data:
#             return False
            
#         pit_data = track_data.get('pit_data', pd.DataFrame())
#         if pit_data.empty:
#             return False
            
#         # Check for required columns in pit_data
#         required_pit_cols = ['NUMBER', 'LAP_NUMBER', 'LAP_TIME']
#         missing_pit_cols = [col for col in required_pit_cols if col not in pit_data.columns]
#         if missing_pit_cols:
#             self.logger.warning(f"⚠️ Missing required pit columns: {missing_pit_cols}")
#             return False
            
#         # Check minimum data volume
#         if len(pit_data) < 5:
#             self.logger.warning(f"⚠️ Insufficient pit data: {len(pit_data)} rows")
#             return False
            
#         return True

#     # -------------------------------
#     # MODEL & STATE MANAGEMENT
#     # -------------------------------
#     def _load_training_state(self) -> Dict:
#         state_file = os.path.join(self.training_state_dir, "training_state.pkl")
#         if os.path.exists(state_file):
#             try: 
#                 return joblib.load(state_file)
#             except Exception as e:
#                 self.logger.warning(f"⚠️ Failed to load training state: {e}")
#         return {"processed_tracks": [], "models": {}}

#     def _update_training_state(self, updates: Dict):
#         state_file = os.path.join(self.training_state_dir, "training_state.pkl")
#         state = self._load_training_state()
#         state.update(updates)
#         try: 
#             joblib.dump(state, state_file)
#         except Exception as e:
#             self.logger.error(f"❌ Failed to save training state: {e}")

#     def _load_processed_data_state(self) -> Dict[str, Dict[str, pd.DataFrame]]:
#         processed = {}
#         state = self._load_training_state()
#         for track in state.get("processed_data_keys", []):
#             cache_file = os.path.join(self.training_state_dir, f"{track}_processed.pkl")
#             if os.path.exists(cache_file):
#                 try: 
#                     processed[track] = joblib.load(cache_file)
#                 except Exception as e:
#                     self.logger.warning(f"⚠️ Failed to load cached data for {track}: {e}")
#                     continue
#         return processed

#     def _save_track_data(self, track: str, data: Dict[str, pd.DataFrame]):
#         try:
#             cache_file = os.path.join(self.training_state_dir, f"{track}_processed.pkl")
#             joblib.dump(data, cache_file)
#         except Exception as e:
#             self.logger.error(f"❌ Failed to save track data for {track}: {e}")

#     def _save_models(self, models: Dict[str, Any]) -> Dict[str, str]:
#         saved = {}
#         for name, result in models.items():
#             try:
#                 filepath = os.path.join(self.models_output_dir, f"{name}_model.pkl")
#                 if isinstance(result, dict) and "model" in result and hasattr(result["model"], "save_model"):
#                     result["model"].save_model(filepath)
#                 else:
#                     joblib.dump(result, filepath)
#                 saved[name] = filepath
#                 self.logger.info(f"💾 Saved {name} model to {filepath}")
#             except Exception as e:
#                 self.logger.error(f"❌ Failed to save {name} model: {e}")
#         return saved

#     def _load_existing_models(self) -> Dict:
#         models = {}
#         if not os.path.exists(self.models_output_dir):
#             return models
            
#         for file in os.listdir(self.models_output_dir):
#             if file.endswith("_model.pkl"):
#                 name = file.replace("_model.pkl", "")
#                 try: 
#                     models[name] = joblib.load(os.path.join(self.models_output_dir, file))
#                     self.logger.info(f"📂 Loaded existing model: {name}")
#                 except Exception as e:
#                     self.logger.error(f"❌ Failed to load model {file}: {e}")
#                     continue
#         return models

#     def _finalize_training(self) -> Dict:
#         models = self._load_existing_models()
#         self._update_training_state({"completed": True})
#         self.logger.info("🎉 Training pipeline completed successfully!")
#         return models

#     def _upload_models_to_firebase(self, models: Dict[str, Any]):
#         try:
#             if hasattr(self.storage, "upload_models_to_firebase"):
#                 success = self.storage.upload_models_to_firebase()
#                 if success: 
#                     self.logger.info("🚀 Models uploaded to Firebase Storage")
#                 else:
#                     self.logger.error("❌ Failed to upload models to Firebase")
#         except Exception as e:
#             self.logger.error(f"❌ Firebase upload failed: {e}")

#     # -------------------------------
#     # LOGGING & VALIDATION - Updated for FirebaseDataLoader structure
#     # -------------------------------
#     def _log_data_quality(self, track: str, data: Dict[str, pd.DataFrame]):
#         """Log data quality report for a track"""
#         report = {}
#         for data_type, df in data.items():
#             if df.empty:
#                 report[data_type] = "EMPTY"
#             else:
#                 null_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0
#                 report[data_type] = {
#                     "rows": len(df),
#                     "columns": len(df.columns),
#                     "null_pct": f"{null_pct:.1f}%"
#                 }
        
#         self.logger.info(f"📊 {track} data quality: {report}")

#     def validate_training_results(self, models: Dict) -> Dict[str, Any]:
#         """Validate and report on training results"""
#         results = {}
        
#         for name, result in models.items():
#             if isinstance(result, dict):
#                 if "error" in result:
#                     results[name] = {"status": "FAILED", "error": result["error"]}
#                 elif result.get('status') == 'success':
#                     # Model-specific validation
#                     if name == "pit_strategy" and "accuracy" in result:
#                         acc = result["accuracy"]
#                         status = "GOOD" if acc > 0.8 else "FAIR" if acc > 0.6 else "POOR"
#                         results[name] = {
#                             "status": status, 
#                             "accuracy": acc, 
#                             "training_samples": result.get("training_samples", 0),
#                             "tracks_used": result.get("tracks_used", 0)
#                         }
#                     elif "test_score" in result:
#                         score = result["test_score"]
#                         status = "GOOD" if score > 0.7 else "FAIR" if score > 0.5 else "POOR"
#                         results[name] = {
#                             "status": status, 
#                             "test_score": score, 
#                             "training_samples": result.get("training_samples", 0),
#                             "tracks_used": result.get("tracks_used", 0)
#                         }
#                     else:
#                         results[name] = {"status": "SUCCESS", "details": "Model trained successfully"}
#                 else:
#                     results[name] = {"status": "UNKNOWN", "result": result}
#             else:
#                 results[name] = {"status": "UNKNOWN", "result": type(result).__name__}
                
#         # Print summary
#         self._print_validation_summary(results)
#         return results

#     def _print_validation_summary(self, validation_results: Dict[str, Any]):
#         """Print formatted validation summary"""
#         print("\n" + "="*70)
#         print("TRAINING RESULTS VALIDATION SUMMARY")
#         print("="*70)
        
#         successful_models = 0
#         total_models = len(validation_results)
        
#         for model_name, result in validation_results.items():
#             status = result.get('status', 'UNKNOWN')
#             status_icon = "✅" if status in ['SUCCESS', 'GOOD', 'FAIR'] else "❌" if status == 'FAILED' else "⚠️"
            
#             if status in ['GOOD', 'FAIR', 'SUCCESS']:
#                 successful_models += 1
                
#             details = []
#             if 'accuracy' in result:
#                 details.append(f"Accuracy: {result['accuracy']:.3f}")
#             if 'test_score' in result:
#                 details.append(f"R² Score: {result['test_score']:.3f}")
#             if 'training_samples' in result:
#                 details.append(f"Samples: {result['training_samples']}")
#             if 'tracks_used' in result:
#                 details.append(f"Tracks: {result['tracks_used']}")
                
#             detail_str = " | ".join(details) if details else result.get('error', 'No details')
            
#             print(f"{status_icon} {model_name:.<20} {status:.<10} {detail_str}")
        
#         success_rate = (successful_models / total_models * 100) if total_models > 0 else 0
#         print(f"\n📊 Overall: {successful_models}/{total_models} models successful ({success_rate:.1f}%)")

#     def get_training_progress(self) -> Dict[str, Any]:
#         """Get current training progress status"""
#         state = self._load_training_state()
#         available_tracks = self.storage.list_available_tracks()
#         processed_tracks = state.get("processed_tracks", [])
        
#         return {
#             "total_tracks": len(available_tracks),
#             "processed_tracks": len(processed_tracks),
#             "remaining_tracks": len(available_tracks) - len(processed_tracks),
#             "completion_percentage": (len(processed_tracks) / len(available_tracks) * 100) if available_tracks else 0,
#             "trained_models": state.get("models", []),
#             "completed": state.get("completed", False),
#             "last_updated": datetime.now().isoformat()
#         }