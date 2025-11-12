import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Firebase
    FIREBASE_CREDENTIALS: str = os.getenv("FIREBASE_CREDENTIALS_BASE64")
    FIREBASE_BUCKET: str = os.getenv("FIREBASE_BUCKET")
    
    # Kaggle
    KAGGLE_USERNAME: str = os.getenv("KAGGLE_USERNAME")
    KAGGLE_KEY: str = os.getenv("KAGGLE_KEY")
    
    # Training
    USE_GPU: bool = True
    MODEL_SAVE_PATH: str = "outputs/models/"
    
    class Config:
        env_file = ".env"

settings = Settings()