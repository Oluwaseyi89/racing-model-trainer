from setuptools import setup, find_packages

setup(
    name="racing-model-trainer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy", 
        "scikit-learn",
        "xgboost",
        "joblib",
        "firebase-admin",
        "kaggle",
        "google-cloud-storage", 
        "pydantic-settings",
        "pydantic",
        "chardet"
    ],
    python_requires=">=3.8",
)