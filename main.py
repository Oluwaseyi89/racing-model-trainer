from training.orchestrator import TrainingOrchestrator
from utils.storage import FirebaseStorage, CloudStorage
from config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Starting Model Training Service")
    
    # Initialize storage
    storage = FirebaseStorage() if settings.USE_FIREBASE else CloudStorage()
    
    # Run training pipeline
    orchestrator = TrainingOrchestrator(storage)
    models = orchestrator.train_all_models()
    
    logger.info("✅ Model training completed successfully")
    return models

if __name__ == "__main__":
    main()