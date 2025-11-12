import os
import json
from typing import Dict, Optional
from utils.logger import get_logger

class KaggleConfig:
    """Configuration manager for Kaggle integration"""
    
    def __init__(self):
        self.logger = get_logger("kaggle_config")
        self.kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
        self._ensure_kaggle_setup()
    
    def _ensure_kaggle_setup(self) -> bool:
        """Ensure Kaggle API is properly configured"""
        try:
            # Check if kaggle.json exists
            if not os.path.exists(self.kaggle_json_path):
                self._create_kaggle_config()
            
            # Set appropriate permissions
            os.chmod(self.kaggle_json_path, 0o600)
            
            self.logger.info("✅ Kaggle configuration verified")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Kaggle setup failed: {e}")
            return False
    
    def _create_kaggle_config(self) -> None:
        """Create kaggle.json from environment variables"""
        kaggle_dir = os.path.dirname(self.kaggle_json_path)
        os.makedirs(kaggle_dir, exist_ok=True)
        
        config = {
            "username": os.getenv("KAGGLE_USERNAME", ""),
            "key": os.getenv("KAGGLE_KEY", "")
        }
        
        with open(self.kaggle_json_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info("📝 Created kaggle.json from environment variables")
    
    def get_kaggle_credentials(self) -> Dict[str, str]:
        """Get Kaggle credentials"""
        try:
            with open(self.kaggle_json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"❌ Failed to read Kaggle credentials: {e}")
            return {"username": "", "key": ""}
    
    def get_kaggle_dataset_config(self, dataset_name: str) -> Dict:
        """Get configuration for specific Kaggle dataset"""
        # Map your dataset names to Kaggle dataset URLs
        dataset_map = {
            "f1-telemetry": "rohanrao/formula-1-world-championship-1950-2020",
            "racing-telemetry": "dhmadeley/racing-telemetry",
            "motor-racing": "dhmadeley/motor-racing"
        }
        
        return {
            "dataset": dataset_map.get(dataset_name, ""),
            "download_path": f"/kaggle/working/{dataset_name}",
            "file_types": [".csv", ".zip", ".parquet"]
        }
    
    def get_training_notebook_config(self) -> Dict:
        """Get configuration for Kaggle training notebook"""
        return {
            "accelerator": "GPU",
            "internet": True,
            "docker_image": "gcr.io/kaggle-gpu-images/python:latest",
            "memory_gb": 16,
            "disk_gb": 50,
            "timeout_hours": 6
        }

class KaggleDatasetManager:
    """Manager for Kaggle dataset operations"""
    
    def __init__(self):
        self.config = KaggleConfig()
        self.logger = get_logger("kaggle_dataset")
    
    def download_dataset(self, dataset_slug: str, target_dir: str) -> bool:
        """Download dataset from Kaggle"""
        try:
            import kaggle
            
            self.logger.info(f"📥 Downloading dataset: {dataset_slug}")
            kaggle.api.dataset_download_files(
                dataset_slug,
                path=target_dir,
                unzip=True,
                quiet=False
            )
            
            self.logger.info(f"✅ Successfully downloaded {dataset_slug}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Dataset download failed: {e}")
            return False
    
    def upload_dataset(self, dataset_dir: str, dataset_title: str, 
                      description: str = "") -> bool:
        """Upload dataset to Kaggle (for sharing trained models)"""
        try:
            import kaggle
            
            metadata = {
                "title": dataset_title,
                "id": f"{self.config.get_kaggle_credentials()['username']}/{dataset_title.lower().replace(' ', '-')}",
                "licenses": [{"name": "CC0-1.0"}]
            }
            
            # Create dataset metadata file
            with open(f"{dataset_dir}/dataset-metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            kaggle.api.dataset_create_new(
                folder=dataset_dir,
                public=False,
                dir_mode="tar"
            )
            
            self.logger.info(f"✅ Successfully uploaded dataset: {dataset_title}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Dataset upload failed: {e}")
            return False

class KaggleNotebookManager:
    """Manager for Kaggle notebook operations"""
    
    def __init__(self):
        self.config = KaggleConfig()
        self.logger = get_logger("kaggle_notebook")
    
    def generate_training_notebook(self, output_path: str, model_type: str = "tire") -> bool:
        """Generate a Kaggle notebook for model training"""
        try:
            notebook_content = self._create_notebook_content(model_type)
            
            with open(output_path, 'w') as f:
                json.dump(notebook_content, f, indent=2)
            
            self.logger.info(f"✅ Generated Kaggle notebook: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Notebook generation failed: {e}")
            return False
    
    def _create_notebook_content(self, model_type: str) -> Dict:
        """Create Jupyter notebook content for Kaggle"""
        
        if model_type == "tire":
            cells = [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Install required packages\n",
                        "!pip install scikit-learn==1.3.2 xgboost==2.0.0 pandas==2.1.4 firebase-admin==6.4.0\n",
                        "!pip install kaggle --upgrade"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Import libraries\n",
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "from sklearn.ensemble import RandomForestRegressor\n",
                        "from sklearn.model_selection import train_test_split\n",
                        "import joblib\n",
                        "import firebase_admin\n",
                        "from firebase_admin import credentials, storage\n",
                        "import zipfile\n",
                        "import io\n",
                        "import os, json, base64"
                    ]
                },
                {
                    "cell_type": "code", 
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Copy model training code from our service\n",
                        "!git clone https://github.com/your-username/model-training-service.git\n",
                        "import sys\n",
                        "sys.path.append('/kaggle/working/model-training-service')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Initialize Firebase (using environment variables)\n",
                        "from data.firebase_loader import FirebaseDataLoader\n",
                        "loader = FirebaseDataLoader(os.getenv('FIREBASE_BUCKET'))"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Train model\n",
                        "from training.orchestrator import TrainingOrchestrator\n",
                        "orchestrator = TrainingOrchestrator(loader)\n",
                        "models = orchestrator.train_all_models()\n",
                        "print('✅ Training completed!')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Save models to Kaggle output\n",
                        "for name, result in models.items():\n",
                        "    result['model'].save_model(f'/kaggle/working/{name}.pkl')\n",
                        "    print(f'Saved: {name}.pkl')"
                    ]
                }
            ]
        
        return {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.12"
                },
                "accelerator": "GPU"
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }

# Utility functions
def setup_kaggle_environment() -> bool:
    """Set up Kaggle environment for training"""
    config = KaggleConfig()
    return config._ensure_kaggle_setup()

def download_training_data(dataset_slug: str) -> str:
    """Download training data from Kaggle and return local path"""
    manager = KaggleDatasetManager()
    target_dir = f"/kaggle/working/{dataset_slug.replace('/', '_')}"
    
    if manager.download_dataset(dataset_slug, target_dir):
        return target_dir
    else:
        raise Exception(f"Failed to download dataset: {dataset_slug}")

# Export main classes and functions
__all__ = [
    'KaggleConfig',
    'KaggleDatasetManager', 
    'KaggleNotebookManager',
    'setup_kaggle_environment',
    'download_training_data'
]