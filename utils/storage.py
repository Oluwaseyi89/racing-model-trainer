import firebase_admin
from firebase_admin import credentials, storage
from google.cloud import storage as gcp_storage
import pandas as pd
import zipfile
import io
import os
import json
import base64
from typing import Dict, Optional
import logging

class FirebaseStorage:
    """Firebase Storage utility for model training service"""
    
    def __init__(self, bucket_name: Optional[str] = None):
        self.bucket_name = bucket_name or os.getenv('FIREBASE_BUCKET')
        self._initialize_firebase()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        if not firebase_admin._apps:
            try:
                cred_json = base64.b64decode(os.getenv("FIREBASE_CREDENTIALS_BASE64"))
                cred_dict = json.loads(cred_json)
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': self.bucket_name
                })
                self.logger.info("✅ Firebase Storage initialized successfully")
            except Exception as e:
                self.logger.error(f"❌ Firebase initialization failed: {e}")
                raise
        
        self.bucket = storage.bucket()
    
    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload file to Firebase Storage"""
        try:
            blob = self.bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
            self.logger.info(f"✅ Uploaded {local_path} to {remote_path}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Upload failed: {e}")
            return False
    
    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download file from Firebase Storage"""
        try:
            blob = self.bucket.blob(remote_path)
            blob.download_to_filename(local_path)
            self.logger.info(f"✅ Downloaded {remote_path} to {local_path}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Download failed: {e}")
            return False
    
    def download_file_as_bytes(self, remote_path: str) -> Optional[bytes]:
        """Download file as bytes from Firebase Storage"""
        try:
            blob = self.bucket.blob(remote_path)
            return blob.download_as_bytes()
        except Exception as e:
            self.logger.error(f"❌ Download as bytes failed: {e}")
            return None
    
    def list_files(self, prefix: str = "") -> list:
        """List files in Firebase Storage with prefix"""
        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            self.logger.error(f"❌ List files failed: {e}")
            return []
    
    def upload_model(self, model_path: str, model_name: str) -> bool:
        """Upload trained model to Firebase Storage"""
        remote_path = f"models/{model_name}.pkl"
        return self.upload_file(model_path, remote_path)
    
    def download_model(self, model_name: str, local_path: str) -> bool:
        """Download trained model from Firebase Storage"""
        remote_path = f"models/{model_name}.pkl"
        return self.download_file(remote_path, local_path)

class CloudStorage:
    """Google Cloud Storage utility as alternative to Firebase"""
    
    def __init__(self, bucket_name: Optional[str] = None):
        self.bucket_name = bucket_name or os.getenv('GCS_BUCKET')
        self.client = gcp_storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)
        self.logger = logging.getLogger(__name__)
    
    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload file to Google Cloud Storage"""
        try:
            blob = self.bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
            self.logger.info(f"✅ Uploaded {local_path} to {remote_path}")
            return True
        except Exception as e:
            self.logger.error(f"❌ GCS Upload failed: {e}")
            return False
    
    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download file from Google Cloud Storage"""
        try:
            blob = self.bucket.blob(remote_path)
            blob.download_to_filename(local_path)
            self.logger.info(f"✅ Downloaded {remote_path} to {local_path}")
            return True
        except Exception as e:
            self.logger.error(f"❌ GCS Download failed: {e}")
            return False

class ModelStorageManager:
    """Manager for storing and retrieving trained models"""
    
    def __init__(self, storage_backend: str = "firebase"):
        self.storage = FirebaseStorage() if storage_backend == "firebase" else CloudStorage()
        self.logger = logging.getLogger(__name__)
    
    def save_training_artifacts(self, models: Dict, metrics: Dict, output_dir: str = "outputs") -> bool:
        """Save all training artifacts to cloud storage"""
        try:
            # Save models
            for model_name, model_result in models.items():
                if hasattr(model_result.get('model', None), 'save_model'):
                    local_path = f"{output_dir}/models/{model_name}.pkl"
                    model_result['model'].save_model(local_path)
                    self.storage.upload_model(local_path, model_name)
            
            # Save metrics
            metrics_path = f"{output_dir}/reports/training_metrics.json"
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            self.storage.upload_file(metrics_path, "reports/training_metrics.json")
            
            self.logger.info("✅ All training artifacts saved to cloud storage")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save training artifacts: {e}")
            return False
    
    def load_trained_models(self, model_names: list, local_dir: str = "ml_models") -> Dict:
        """Load trained models from cloud storage"""
        loaded_models = {}
        
        for model_name in model_names:
            try:
                local_path = f"{local_dir}/{model_name}.pkl"
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                if self.storage.download_model(model_name, local_path):
                    # Models would be loaded by the main application
                    loaded_models[model_name] = {
                        'local_path': local_path,
                        'loaded': True
                    }
                    self.logger.info(f"✅ Loaded model: {model_name}")
                else:
                    loaded_models[model_name] = {
                        'local_path': None,
                        'loaded': False,
                        'error': 'Download failed'
                    }
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to load model {model_name}: {e}")
                loaded_models[model_name] = {
                    'local_path': None,
                    'loaded': False,
                    'error': str(e)
                }
        
        return loaded_models
    
    def cleanup_local_files(self, directory: str = "outputs"):
        """Clean up local files after uploading to cloud"""
        try:
            import shutil
            if os.path.exists(directory):
                shutil.rmtree(directory)
                self.logger.info(f"✅ Cleaned up local directory: {directory}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")
            return False

# Utility functions
def get_storage_backend(backend_type: str = None):
    """Get appropriate storage backend based on configuration"""
    backend_type = backend_type or os.getenv('STORAGE_BACKEND', 'firebase')
    
    if backend_type == 'firebase':
        return FirebaseStorage()
    elif backend_type == 'gcs':
        return CloudStorage()
    else:
        raise ValueError(f"Unsupported storage backend: {backend_type}")

def save_model_to_cloud(model, model_name: str, storage_backend: str = None) -> bool:
    """Convenience function to save model to cloud storage"""
    storage = get_storage_backend(storage_backend)
    manager = ModelStorageManager()
    
    # Create temporary local file
    temp_path = f"/tmp/{model_name}.pkl"
    try:
        if hasattr(model, 'save_model'):
            model.save_model(temp_path)
        else:
            import joblib
            joblib.dump(model, temp_path)
        
        return storage.upload_file(temp_path, f"models/{model_name}.pkl")
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)