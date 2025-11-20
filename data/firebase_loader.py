import firebase_admin
from firebase_admin import credentials, storage
import pandas as pd
import io
import os, json, base64
import joblib
from typing import List, Dict
import time
import numpy as np
from datetime import datetime

class FirebaseDataLoader:
    def __init__(self, bucket_name: str):
        if not firebase_admin._apps:
            cred_json = base64.b64decode(os.getenv("FIREBASE_CREDENTIALS_BASE64"))
            cred_dict = json.loads(cred_json)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred, {
                'storageBucket': bucket_name
            })
        self.bucket = storage.bucket()
        
        # Define EXACT schemas based on your data structures
        self.schemas = {
            # 'pit_data': {
            #     'required_cols': ['NUMBER', 'DRIVER_NUMBER', 'LAP_NUMBER', 'LAP_TIME'],
            #     'dtype_mapping': {
            #         'NUMBER': 'int64', 'DRIVER_NUMBER': 'int64', 'LAP_NUMBER': 'int64',
            #         'LAP_TIME': 'str', 'LAP_IMPROVEMENT': 'float64', 
            #         'CROSSING_FINISH_LINE_IN_PIT': 'str', 'S1': 'str', 'S1_IMPROVEMENT': 'float64',
            #         'S2': 'str', 'S2_IMPROVEMENT': 'float64', 'S3': 'str', 'S3_IMPROVEMENT': 'float64',
            #         'KPH': 'float64', 'ELAPSED': 'str', 'HOUR': 'str', 'S1_LARGE': 'str',
            #         'S2_LARGE': 'str', 'S3_LARGE': 'str', 'TOP_SPEED': 'float64', 'PIT_TIME': 'str',
            #         'CLASS': 'str', 'GROUP': 'str', 'MANUFACTURER': 'str', 'FLAG_AT_FL': 'str',
            #         'S1_SECONDS': 'float64', 'S2_SECONDS': 'float64', 'S3_SECONDS': 'float64',
            #         'IM1a_time': 'str', 'IM1a_elapsed': 'str', 'IM1_time': 'str', 'IM1_elapsed': 'str',
            #         'IM2a_time': 'str', 'IM2a_elapsed': 'str', 'IM2_time': 'str', 'IM2_elapsed': 'str',
            #         'IM3a_time': 'str', 'IM3a_elapsed': 'str', 'FL_time': 'str', 'FL_elapsed': 'str'
            #     },
            #     'default_values': {
            #         'DRIVER_NUMBER': 0,  # CRITICAL FIX: Added default for DRIVER_NUMBER
            #         'LAP_IMPROVEMENT': 0.0, 'S1_IMPROVEMENT': 0.0, 'S2_IMPROVEMENT': 0.0, 
            #         'S3_IMPROVEMENT': 0.0, 'KPH': 0.0, 'TOP_SPEED': 0.0, 'S1_SECONDS': 0.0,
            #         'S2_SECONDS': 0.0, 'S3_SECONDS': 0.0, 'CROSSING_FINISH_LINE_IN_PIT': '',
            #         'CLASS': '', 'GROUP': '', 'MANUFACTURER': '', 'FLAG_AT_FL': '',
            #         'IM1a_time': '', 'IM1a_elapsed': '', 'IM1_time': '', 'IM1_elapsed': '',
            #         'IM2a_time': '', 'IM2a_elapsed': '', 'IM2_time': '', 'IM2_elapsed': '',
            #         'IM3a_time': '', 'IM3a_elapsed': '', 'FL_time': '', 'FL_elapsed': '',
            #         'S1': '', 'S2': '', 'S3': '', 'ELAPSED': '', 'HOUR': '', 'S1_LARGE': '',
            #         'S2_LARGE': '', 'S3_LARGE': '', 'PIT_TIME': ''
            #     }
            # },
            'pit_data': {
                'required_cols': ['NUMBER', 'LAP_NUMBER', 'LAP_TIME'],  # REMOVED DRIVER_NUMBER from required
                'dtype_mapping': {
                    'NUMBER': 'int64', 'DRIVER_NUMBER': 'int64', 'LAP_NUMBER': 'int64',
                    'LAP_TIME': 'str', 'LAP_IMPROVEMENT': 'float64', 
                    'CROSSING_FINISH_LINE_IN_PIT': 'str', 'S1': 'str', 'S1_IMPROVEMENT': 'float64',
                    'S2': 'str', 'S2_IMPROVEMENT': 'float64', 'S3': 'str', 'S3_IMPROVEMENT': 'float64',
                    'KPH': 'float64', 'ELAPSED': 'str', 'HOUR': 'str', 'S1_LARGE': 'str',
                    'S2_LARGE': 'str', 'S3_LARGE': 'str', 'TOP_SPEED': 'float64', 'PIT_TIME': 'str',
                    'CLASS': 'str', 'GROUP': 'str', 'MANUFACTURER': 'str', 'FLAG_AT_FL': 'str',
                    'S1_SECONDS': 'float64', 'S2_SECONDS': 'float64', 'S3_SECONDS': 'float64',
                    'IM1a_time': 'str', 'IM1a_elapsed': 'str', 'IM1_time': 'str', 'IM1_elapsed': 'str',
                    'IM2a_time': 'str', 'IM2a_elapsed': 'str', 'IM2_time': 'str', 'IM2_elapsed': 'str',
                    'IM3a_time': 'str', 'IM3a_elapsed': 'str', 'FL_time': 'str', 'FL_elapsed': 'str'
                },
                'default_values': {
                    'DRIVER_NUMBER': 0,  # KEPT default for DRIVER_NUMBER
                    'LAP_IMPROVEMENT': 0.0, 'S1_IMPROVEMENT': 0.0, 'S2_IMPROVEMENT': 0.0, 
                    'S3_IMPROVEMENT': 0.0, 'KPH': 0.0, 'TOP_SPEED': 0.0, 'S1_SECONDS': 0.0,
                    'S2_SECONDS': 0.0, 'S3_SECONDS': 0.0, 'CROSSING_FINISH_LINE_IN_PIT': '',
                    'CLASS': '', 'GROUP': '', 'MANUFACTURER': '', 'FLAG_AT_FL': '',
                    'IM1a_time': '', 'IM1a_elapsed': '', 'IM1_time': '', 'IM1_elapsed': '',
                    'IM2a_time': '', 'IM2a_elapsed': '', 'IM2_time': '', 'IM2_elapsed': '',
                    'IM3a_time': '', 'IM3a_elapsed': '', 'FL_time': '', 'FL_elapsed': '',
                    'S1': '', 'S2': '', 'S3': '', 'ELAPSED': '', 'HOUR': '', 'S1_LARGE': '',
                    'S2_LARGE': '', 'S3_LARGE': '', 'PIT_TIME': ''
                }
            },
            'race_data': {
                'required_cols': ['POSITION', 'NUMBER', 'STATUS', 'LAPS', 'TOTAL_TIME'],
                'dtype_mapping': {
                    'POSITION': 'int64', 'NUMBER': 'int64', 'STATUS': 'str', 'LAPS': 'int64',
                    'TOTAL_TIME': 'str', 'GAP_FIRST': 'str', 'GAP_PREVIOUS': 'str',
                    'FL_LAPNUM': 'int64', 'FL_TIME': 'str', 'FL_KPH': 'float64',
                    'CLASS': 'str', 'GROUP': 'str', 'DIVISION': 'str', 'VEHICLE': 'str',
                    'TIRES': 'str', 'ECM Participant Id': 'str', 'ECM Team Id': 'str',
                    'ECM Category Id': 'str', 'ECM Car Id': 'str', 'ECM Brand Id': 'str',
                    'ECM Country Id': 'str', '*Extra 7': 'str', '*Extra 8': 'str',
                    '*Extra 9': 'str', 'Sort Key': 'str', 'DRIVER_*Extra 3': 'str',
                    'DRIVER_*Extra 4': 'str', 'DRIVER_*Extra 5': 'str'
                },
                'default_values': {
                    'FL_LAPNUM': 0, 'FL_KPH': 0.0, 'GAP_FIRST': '-', 'GAP_PREVIOUS': '-',
                    'STATUS': '', 'CLASS': '', 'GROUP': '', 'DIVISION': '', 'VEHICLE': '',
                    'TIRES': '', 'ECM Participant Id': '', 'ECM Team Id': '', 
                    'ECM Category Id': '', 'ECM Car Id': '', 'ECM Brand Id': '',
                    'ECM Country Id': '', '*Extra 7': '', '*Extra 8': '', '*Extra 9': '',
                    'Sort Key': '', 'DRIVER_*Extra 3': '', 'DRIVER_*Extra 4': '', 'DRIVER_*Extra 5': ''
                }
            },
            'telemetry_data': {
                'required_cols': ['timestamp', 'vehicle_id', 'lap', 'speed'],
                'dtype_mapping': {
                    'timestamp': 'datetime64[ns]', 'vehicle_id': 'str', 'lap': 'int64',
                    'outing': 'int64', 'meta_session': 'str', 'accx_can': 'float64',
                    'accy_can': 'float64', 'gear': 'int64', 'speed': 'float64'
                },
                'default_values': {
                    'accx_can': 0.0, 'accy_can': 0.0, 'gear': 0, 'outing': 0,
                    'meta_session': ''
                }
            },
            'weather_data': {
                'required_cols': ['TIME_UTC_SECONDS', 'AIR_TEMP', 'TRACK_TEMP'],
                'dtype_mapping': {
                    'TIME_UTC_SECONDS': 'int64', 'TIME_UTC_STR': 'str',
                    'AIR_TEMP': 'float64', 'TRACK_TEMP': 'float64',
                    'HUMIDITY': 'float64', 'PRESSURE': 'float64',
                    'WIND_SPEED': 'float64', 'WIND_DIRECTION': 'int64',
                    'RAIN': 'int64'
                },
                'default_values': {
                    'TRACK_TEMP': 0.0, 'HUMIDITY': 0.0, 'PRESSURE': 0.0,
                    'WIND_SPEED': 0.0, 'WIND_DIRECTION': 0, 'RAIN': 0,
                    'TIME_UTC_STR': ''
                }
            }
        }

    def _enforce_schema(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Enforce EXACT schema on DataFrame using your column names"""
        if df.empty:
            return df
            
        schema = self.schemas[data_type]
        df_clean = df.copy()
        
        # Ensure required columns exist
        for col in schema['required_cols']:
            if col not in df_clean.columns:
                raise ValueError(f"Missing required column '{col}' for {data_type}")
        
        # Add ALL missing columns with default values
        for col in schema['dtype_mapping'].keys():
            if col not in df_clean.columns:
                default_val = schema['default_values'].get(col, 
                    0 if schema['dtype_mapping'][col] in ['int64', 'float64'] else '')
                df_clean[col] = default_val
        
        # Convert data types - FIXED: Use direct assignment instead of inplace
        for col, dtype in schema['dtype_mapping'].items():
            if col in df_clean.columns:
                try:
                    if dtype == 'datetime64[ns]' and col == 'timestamp':
                        df_clean[col] = pd.to_datetime(df_clean[col])
                    elif dtype in ['int64', 'float64']:
                        # Use direct assignment to avoid chained assignment warning
                        numeric_series = pd.to_numeric(df_clean[col], errors='coerce')
                        default_val = schema['default_values'].get(col, 0)
                        df_clean[col] = numeric_series.fillna(default_val)
                    else:
                        # Use direct assignment for string columns
                        df_clean[col] = df_clean[col].astype(str).replace(['nan', 'None'], '', regex=True)
                except Exception as e:
                    print(f"⚠️ Failed to convert {col} to {dtype}: {e}")
                    default_val = schema['default_values'].get(col, 
                        0 if dtype in ['int64', 'float64'] else '')
                    df_clean[col] = default_val
        
        # Fill remaining NaN values - FIXED: Use direct assignment instead of inplace
        for col in df_clean.columns:
            if df_clean[col].isna().any():
                default_val = schema['default_values'].get(col, 
                    0 if schema['dtype_mapping'].get(col, '').endswith('64') else '')
                # Use direct assignment instead of inplace
                df_clean[col] = df_clean[col].fillna(default_val)
        
        return df_clean

    # def _load_csv_with_schema(self, file_content: bytes, data_type: str, filename: str) -> pd.DataFrame:
    #     """Load CSV file with EXACT schema enforcement"""
    #     try:
    #         # Determine separator based on data type
    #         sep = ';' if data_type in ['pit_data', 'race_data', 'weather_data'] else ','
            
    #         try:
    #             df = pd.read_csv(io.BytesIO(file_content), sep=sep, low_memory=False)
    #             if len(df.columns) > 1 and len(df) > 0:
    #                 print(f"✅ Loaded {filename} (sep: {sep})")
    #                 return self._enforce_schema(df, data_type)
    #         except Exception as e:
    #             print(f"⚠️ Failed with sep {sep}, trying alternatives: {e}")
    #             # Try other separators as fallback
    #             for alt_sep in [',', '\t', ';']:
    #                 if alt_sep != sep:
    #                     try:
    #                         df = pd.read_csv(io.BytesIO(file_content), sep=alt_sep, low_memory=False)
    #                         if len(df.columns) > 1 and len(df) > 0:
    #                             print(f"✅ Loaded {filename} (sep: {alt_sep})")
    #                             return self._enforce_schema(df, data_type)
    #                     except:
    #                         continue
            
    #         print(f"❌ Could not load {filename} with any separator")
    #         return pd.DataFrame()
            
    #     except Exception as e:
    #         print(f"⚠️ Failed to load {filename}: {e}")
    #         return pd.DataFrame()

    def _load_csv_with_schema(self, file_content: bytes, data_type: str, filename: str) -> pd.DataFrame:
        """Load CSV file with EXACT schema enforcement"""
        try:
            # Determine separator based on data type
            sep = ';' if data_type in ['pit_data', 'race_data', 'weather_data'] else ','
            
            try:
                # First try normal loading
                df = pd.read_csv(io.BytesIO(file_content), sep=sep, low_memory=False)
                
                # Check if we have malformed headers (only 1 column but should have multiple)
                if len(df.columns) == 1 and sep == ';':
                    print(f"⚠️ Detected malformed semicolon header in {filename}, attempting repair...")
                    
                    # Read the file as raw text to manually parse headers
                    file_text = file_content.decode('utf-8')
                    lines = file_text.split('\n')
                    
                    if lines:
                        # Split first line on semicolon and clean up headers
                        raw_headers = lines[0].split(';')
                        cleaned_headers = [header.strip() for header in raw_headers]
                        
                        # Read data without header
                        df = pd.read_csv(io.BytesIO(file_content), sep=sep, header=0, skiprows=1, low_memory=False)
                        
                        # Assign cleaned headers
                        if len(cleaned_headers) == len(df.columns):
                            df.columns = cleaned_headers
                            print(f"✅ Repaired header for {filename}")
                        else:
                            # If header count doesn't match, use the first row as data and create generic headers
                            df_with_header = pd.read_csv(io.BytesIO(file_content), sep=sep, header=None, low_memory=False)
                            df = df_with_header.iloc[1:]  # Skip first row (the malformed header)
                            df.columns = [f'col_{i}' for i in range(len(df.columns))]
                            print(f"⚠️ Created generic headers for {filename}")
                
                if len(df.columns) > 1 and len(df) > 0:
                    print(f"✅ Loaded {filename} (sep: {sep})")
                    return self._enforce_schema(df, data_type)
                    
            except Exception as e:
                print(f"⚠️ Failed with sep {sep}, trying alternatives: {e}")
                # Try other separators as fallback
                for alt_sep in [',', '\t', ';']:
                    if alt_sep != sep:
                        try:
                            df = pd.read_csv(io.BytesIO(file_content), sep=alt_sep, low_memory=False)
                            if len(df.columns) > 1 and len(df) > 0:
                                print(f"✅ Loaded {filename} (sep: {alt_sep})")
                                return self._enforce_schema(df, data_type)
                        except:
                            continue
            
            print(f"❌ Could not load {filename} with any separator")
            return pd.DataFrame()
            
        except Exception as e:
            print(f"⚠️ Failed to load {filename}: {e}")
            return pd.DataFrame()

    def _download_and_load_file(self, blob_path: str, data_type: str) -> pd.DataFrame:
        """Download and load individual CSV file from Firebase"""
        try:
            blob = self.bucket.blob(blob_path)
            if not blob.exists():
                print(f"📝 File not found: {blob_path}")
                return pd.DataFrame()
            
            file_content = blob.download_as_bytes()
            filename = os.path.basename(blob_path)
            
            print(f"📥 Downloading {filename}...")
            df = self._load_csv_with_schema(file_content, data_type, filename)
            
            if not df.empty:
                print(f"✅ Successfully loaded {filename} with {len(df)} rows, {len(df.columns)} columns")
                print(f"   Columns: {list(df.columns)}")
            else:
                print(f"❌ Failed to load valid data from {filename}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error downloading {blob_path}: {e}")
            return pd.DataFrame()

    def load_track_data(self, track_name: str) -> Dict[str, pd.DataFrame]:
        """Load all data types for a specific track from dataset_files/ folder"""
        try:
            # Check cache first
            if self._check_track_data_cached(track_name):
                return self._load_cached_data(track_name)
            
            print(f"🏁 Loading data for track: {track_name}")
            
            track_data = {}
            data_types = ['pit_data', 'race_data', 'telemetry_data', 'weather_data']
            
            for data_type in data_types:
                # Construct blob path
                blob_path = f"dataset_files/{track_name}_{data_type}.csv"
                df = self._download_and_load_file(blob_path, data_type)
                track_data[data_type] = df
            
            # Cache the loaded data
            self._cache_track_data(track_name, track_data)
            
            # Report loading statistics
            loaded_count = sum(1 for df in track_data.values() if not df.empty)
            print(f"📊 Loaded {loaded_count} data types for {track_name}")
            
            return track_data
            
        except Exception as e:
            print(f"❌ Failed to load {track_name}: {e}")
            return self._return_empty_data()

    def _check_track_data_cached(self, track_name: str) -> bool:
        """Check if track data is cached"""
        cache_file = f'/kaggle/working/{track_name}_cached.pkl'
        return os.path.exists(cache_file)

    def _load_cached_data(self, track_name: str) -> Dict[str, pd.DataFrame]:
        """Load data from cache"""
        cache_file = f'/kaggle/working/{track_name}_cached.pkl'
        print(f"📂 Loading cached data for {track_name}")
        with open(cache_file, 'rb') as f:
            return joblib.load(f)

    def _cache_track_data(self, track_name: str, data: Dict[str, pd.DataFrame]):
        """Cache track data"""
        try:
            cache_file = f'/kaggle/working/{track_name}_cached.pkl'
            with open(cache_file, 'wb') as f:
                joblib.dump(data, f)
            print(f"💾 Cached {track_name} data")
        except Exception as e:
            print(f"⚠️ Could not cache {track_name}: {e}")

    def load_all_tracks(self, tracks: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load multiple tracks efficiently"""
        all_data = {}
        
        for i, track in enumerate(tracks, 1):
            try:
                print(f"\n🏁 Processing track {i}/{len(tracks)}: {track}")
                all_data[track] = self.load_track_data(track)
                time.sleep(1)  # Avoid overwhelming Firebase
            except Exception as e:
                print(f"⚠️ Failed to load {track}: {e}")
                all_data[track] = self._return_empty_data()
        
        return all_data

    def _return_empty_data(self) -> Dict[str, pd.DataFrame]:
        """Return empty data structure with proper schemas"""
        return {
            'pit_data': pd.DataFrame(),
            'race_data': pd.DataFrame(),
            'telemetry_data': pd.DataFrame(),
            'weather_data': pd.DataFrame()
        }

    def list_available_tracks(self) -> List[str]:
        """List all available tracks in dataset_files folder"""
        tracks = set()
        try:
            blobs = self.bucket.list_blobs(prefix="dataset_files/")
            for blob in blobs:
                if blob.name.endswith('.csv'):
                    # Extract track name from filename (e.g., "dataset_files/road_america_pit_data.csv" -> "road_america")
                    filename = os.path.basename(blob.name)
                    # Remove file extension and split by underscore
                    base_name = filename.replace('.csv', '')
                    parts = base_name.split('_')
                    
                    # Handle special case for road_america
                    if len(parts) >= 2 and parts[0] == 'road' and parts[1] == 'america':
                        track_name = 'road_america'
                    else:
                        track_name = parts[0]  # Default to first part for simple names
                    
                    if track_name:
                        tracks.add(track_name)
            
            print(f"📦 Found {len(tracks)} tracks: {sorted(list(tracks))}")
            return sorted(list(tracks))
            
        except Exception as e:
            print(f"❌ Failed to list tracks: {e}")
            return []

    def load_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load all available tracks"""
        available_tracks = self.list_available_tracks()
        return self.load_all_tracks(available_tracks)

    def validate_data_quality(self, track_data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """Validate data quality with schema compliance"""
        validation = {}
        
        for data_type, df in track_data.items():
            if df.empty:
                validation[data_type] = {'status': 'missing', 'rows': 0, 'columns': 0}
            else:
                # Check schema compliance
                schema = self.schemas[data_type]
                missing_required = [col for col in schema['required_cols'] if col not in df.columns]
                
                validation[data_type] = {
                    'status': 'loaded',
                    'rows': len(df),
                    'columns': len(df.columns),
                    'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                    'schema_compliant': len(missing_required) == 0,
                    'missing_required_cols': missing_required,
                    'actual_columns': list(df.columns)
                }
        
        return validation
    
    def upload_models_to_firebase(self, models_dir: str = "outputs/models"):
        """Upload all trained models to Firebase Storage after training completes"""
        try:
            if not os.path.exists(models_dir):
                print(f"❌ Models directory not found: {models_dir}")
                return False
                
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            if not model_files:
                print("❌ No model files found to upload")
                return False
                
            for model_file in model_files:
                local_path = os.path.join(models_dir, model_file)
                blob_path = f"trained_models/{model_file}"
                
                blob = self.bucket.blob(blob_path)
                blob.upload_from_filename(local_path)
                print(f"✅ Uploaded {model_file} to Firebase Storage")
                
            print("🎉 All models successfully uploaded to Firebase!")
            return True
        except Exception as e:
            print(f"❌ Failed to upload models: {e}")
            return False
    
    def download_models_from_firebase(self, local_dir: str = "outputs/models"):
        """Download models from Firebase Storage"""
        try:
            os.makedirs(local_dir, exist_ok=True)
            blobs = self.bucket.list_blobs(prefix="trained_models/")
            
            downloaded_count = 0
            for blob in blobs:
                if blob.name.endswith('.pkl'):
                    local_path = os.path.join(local_dir, os.path.basename(blob.name))
                    blob.download_to_filename(local_path)
                    downloaded_count += 1
                    print(f"✅ Downloaded {os.path.basename(blob.name)} from Firebase")
            
            if downloaded_count > 0:
                print(f"🎉 Downloaded {downloaded_count} models from Firebase!")
            else:
                print("📝 No models found in Firebase Storage")
                
            return downloaded_count > 0
        except Exception as e:
            print(f"❌ Failed to download models: {e}")
            return False























# import firebase_admin
# from firebase_admin import credentials, storage
# import pandas as pd
# import io
# import os, json, base64
# import joblib
# from typing import List, Dict
# import time
# import numpy as np
# from datetime import datetime

# class FirebaseDataLoader:
#     def __init__(self, bucket_name: str):
#         if not firebase_admin._apps:
#             cred_json = base64.b64decode(os.getenv("FIREBASE_CREDENTIALS_BASE64"))
#             cred_dict = json.loads(cred_json)
#             cred = credentials.Certificate(cred_dict)
#             firebase_admin.initialize_app(cred, {
#                 'storageBucket': bucket_name
#             })
#         self.bucket = storage.bucket()
        
#         # Define EXACT schemas based on your data structures
#         self.schemas = {
#             'pit_data': {
#                 'required_cols': ['NUMBER', 'DRIVER_NUMBER', 'LAP_NUMBER', 'LAP_TIME'],
#                 'dtype_mapping': {
#                     'NUMBER': 'int64', 'DRIVER_NUMBER': 'int64', 'LAP_NUMBER': 'int64',
#                     'LAP_TIME': 'str', 'LAP_IMPROVEMENT': 'float64', 
#                     'CROSSING_FINISH_LINE_IN_PIT': 'str', 'S1': 'str', 'S1_IMPROVEMENT': 'float64',
#                     'S2': 'str', 'S2_IMPROVEMENT': 'float64', 'S3': 'str', 'S3_IMPROVEMENT': 'float64',
#                     'KPH': 'float64', 'ELAPSED': 'str', 'HOUR': 'str', 'S1_LARGE': 'str',
#                     'S2_LARGE': 'str', 'S3_LARGE': 'str', 'TOP_SPEED': 'float64', 'PIT_TIME': 'str',
#                     'CLASS': 'str', 'GROUP': 'str', 'MANUFACTURER': 'str', 'FLAG_AT_FL': 'str',
#                     'S1_SECONDS': 'float64', 'S2_SECONDS': 'float64', 'S3_SECONDS': 'float64',
#                     'IM1a_time': 'str', 'IM1a_elapsed': 'str', 'IM1_time': 'str', 'IM1_elapsed': 'str',
#                     'IM2a_time': 'str', 'IM2a_elapsed': 'str', 'IM2_time': 'str', 'IM2_elapsed': 'str',
#                     'IM3a_time': 'str', 'IM3a_elapsed': 'str', 'FL_time': 'str', 'FL_elapsed': 'str'
#                 },
#                 'default_values': {
#                     'LAP_IMPROVEMENT': 0.0, 'S1_IMPROVEMENT': 0.0, 'S2_IMPROVEMENT': 0.0, 
#                     'S3_IMPROVEMENT': 0.0, 'KPH': 0.0, 'TOP_SPEED': 0.0, 'S1_SECONDS': 0.0,
#                     'S2_SECONDS': 0.0, 'S3_SECONDS': 0.0, 'CROSSING_FINISH_LINE_IN_PIT': '',
#                     'CLASS': '', 'GROUP': '', 'MANUFACTURER': '', 'FLAG_AT_FL': '',
#                     'IM1a_time': '', 'IM1a_elapsed': '', 'IM1_time': '', 'IM1_elapsed': '',
#                     'IM2a_time': '', 'IM2a_elapsed': '', 'IM2_time': '', 'IM2_elapsed': '',
#                     'IM3a_time': '', 'IM3a_elapsed': '', 'FL_time': '', 'FL_elapsed': '',
#                     'S1': '', 'S2': '', 'S3': '', 'ELAPSED': '', 'HOUR': '', 'S1_LARGE': '',
#                     'S2_LARGE': '', 'S3_LARGE': '', 'PIT_TIME': ''
#                 }
#             },
#             'race_data': {
#                 'required_cols': ['POSITION', 'NUMBER', 'STATUS', 'LAPS', 'TOTAL_TIME'],
#                 'dtype_mapping': {
#                     'POSITION': 'int64', 'NUMBER': 'int64', 'STATUS': 'str', 'LAPS': 'int64',
#                     'TOTAL_TIME': 'str', 'GAP_FIRST': 'str', 'GAP_PREVIOUS': 'str',
#                     'FL_LAPNUM': 'int64', 'FL_TIME': 'str', 'FL_KPH': 'float64',
#                     'CLASS': 'str', 'GROUP': 'str', 'DIVISION': 'str', 'VEHICLE': 'str',
#                     'TIRES': 'str', 'ECM Participant Id': 'str', 'ECM Team Id': 'str',
#                     'ECM Category Id': 'str', 'ECM Car Id': 'str', 'ECM Brand Id': 'str',
#                     'ECM Country Id': 'str', '*Extra 7': 'str', '*Extra 8': 'str',
#                     '*Extra 9': 'str', 'Sort Key': 'str', 'DRIVER_*Extra 3': 'str',
#                     'DRIVER_*Extra 4': 'str', 'DRIVER_*Extra 5': 'str'
#                 },
#                 'default_values': {
#                     'FL_LAPNUM': 0, 'FL_KPH': 0.0, 'GAP_FIRST': '-', 'GAP_PREVIOUS': '-',
#                     'STATUS': '', 'CLASS': '', 'GROUP': '', 'DIVISION': '', 'VEHICLE': '',
#                     'TIRES': '', 'ECM Participant Id': '', 'ECM Team Id': '', 
#                     'ECM Category Id': '', 'ECM Car Id': '', 'ECM Brand Id': '',
#                     'ECM Country Id': '', '*Extra 7': '', '*Extra 8': '', '*Extra 9': '',
#                     'Sort Key': '', 'DRIVER_*Extra 3': '', 'DRIVER_*Extra 4': '', 'DRIVER_*Extra 5': ''
#                 }
#             },
#             'telemetry_data': {
#                 'required_cols': ['timestamp', 'vehicle_id', 'lap', 'speed'],
#                 'dtype_mapping': {
#                     'timestamp': 'datetime64[ns]', 'vehicle_id': 'str', 'lap': 'int64',
#                     'outing': 'int64', 'meta_session': 'str', 'accx_can': 'float64',
#                     'accy_can': 'float64', 'gear': 'int64', 'speed': 'float64'
#                 },
#                 'default_values': {
#                     'accx_can': 0.0, 'accy_can': 0.0, 'gear': 0, 'outing': 0,
#                     'meta_session': ''
#                 }
#             },
#             'weather_data': {
#                 'required_cols': ['TIME_UTC_SECONDS', 'AIR_TEMP', 'TRACK_TEMP'],
#                 'dtype_mapping': {
#                     'TIME_UTC_SECONDS': 'int64', 'TIME_UTC_STR': 'str',
#                     'AIR_TEMP': 'float64', 'TRACK_TEMP': 'float64',
#                     'HUMIDITY': 'float64', 'PRESSURE': 'float64',
#                     'WIND_SPEED': 'float64', 'WIND_DIRECTION': 'int64',
#                     'RAIN': 'int64'
#                 },
#                 'default_values': {
#                     'TRACK_TEMP': 0.0, 'HUMIDITY': 0.0, 'PRESSURE': 0.0,
#                     'WIND_SPEED': 0.0, 'WIND_DIRECTION': 0, 'RAIN': 0,
#                     'TIME_UTC_STR': ''
#                 }
#             }
#         }

#     def _enforce_schema(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
#         """Enforce EXACT schema on DataFrame using your column names"""
#         if df.empty:
#             return df
            
#         schema = self.schemas[data_type]
#         df_clean = df.copy()
        
#         # Ensure required columns exist
#         for col in schema['required_cols']:
#             if col not in df_clean.columns:
#                 raise ValueError(f"Missing required column '{col}' for {data_type}")
        
#         # Add ALL missing columns with default values
#         for col in schema['dtype_mapping'].keys():
#             if col not in df_clean.columns:
#                 default_val = schema['default_values'].get(col, 
#                     0 if schema['dtype_mapping'][col] in ['int64', 'float64'] else '')
#                 df_clean[col] = default_val
        
#         # Convert data types
#         for col, dtype in schema['dtype_mapping'].items():
#             if col in df_clean.columns:
#                 try:
#                     if dtype == 'datetime64[ns]' and col == 'timestamp':
#                         df_clean[col] = pd.to_datetime(df_clean[col])
#                     elif dtype in ['int64', 'float64']:
#                         df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(schema['default_values'].get(col, 0))
#                     else:
#                         df_clean[col] = df_clean[col].astype(str).replace('nan', '').replace('None', '')
#                 except Exception as e:
#                     print(f"⚠️ Failed to convert {col} to {dtype}: {e}")
#                     df_clean[col] = schema['default_values'].get(col, 
#                         0 if dtype in ['int64', 'float64'] else '')
        
#         # Fill remaining NaN values
#         for col in df_clean.columns:
#             if df_clean[col].isna().any():
#                 default_val = schema['default_values'].get(col, 
#                     0 if schema['dtype_mapping'].get(col, '').endswith('64') else '')
#                 df_clean[col].fillna(default_val, inplace=True)
        
#         return df_clean

#     def _load_csv_with_schema(self, file_content: bytes, data_type: str, filename: str) -> pd.DataFrame:
#         """Load CSV file with EXACT schema enforcement"""
#         try:
#             # Determine separator based on data type
#             sep = ';' if data_type in ['pit_data', 'race_data', 'weather_data'] else ','
            
#             try:
#                 df = pd.read_csv(io.BytesIO(file_content), sep=sep, low_memory=False)
#                 if len(df.columns) > 1 and len(df) > 0:
#                     print(f"✅ Loaded {filename} (sep: {sep})")
#                     return self._enforce_schema(df, data_type)
#             except Exception as e:
#                 print(f"⚠️ Failed with sep {sep}, trying alternatives: {e}")
#                 # Try other separators as fallback
#                 for alt_sep in [',', '\t', ';']:
#                     if alt_sep != sep:
#                         try:
#                             df = pd.read_csv(io.BytesIO(file_content), sep=alt_sep, low_memory=False)
#                             if len(df.columns) > 1 and len(df) > 0:
#                                 print(f"✅ Loaded {filename} (sep: {alt_sep})")
#                                 return self._enforce_schema(df, data_type)
#                         except:
#                             continue
            
#             print(f"❌ Could not load {filename} with any separator")
#             return pd.DataFrame()
            
#         except Exception as e:
#             print(f"⚠️ Failed to load {filename}: {e}")
#             return pd.DataFrame()

#     def _download_and_load_file(self, blob_path: str, data_type: str) -> pd.DataFrame:
#         """Download and load individual CSV file from Firebase"""
#         try:
#             blob = self.bucket.blob(blob_path)
#             if not blob.exists():
#                 print(f"📝 File not found: {blob_path}")
#                 return pd.DataFrame()
            
#             file_content = blob.download_as_bytes()
#             filename = os.path.basename(blob_path)
            
#             print(f"📥 Downloading {filename}...")
#             df = self._load_csv_with_schema(file_content, data_type, filename)
            
#             if not df.empty:
#                 print(f"✅ Successfully loaded {filename} with {len(df)} rows, {len(df.columns)} columns")
#                 print(f"   Columns: {list(df.columns)}")
#             else:
#                 print(f"❌ Failed to load valid data from {filename}")
            
#             return df
            
#         except Exception as e:
#             print(f"❌ Error downloading {blob_path}: {e}")
#             return pd.DataFrame()

#     def load_track_data(self, track_name: str) -> Dict[str, pd.DataFrame]:
#         """Load all data types for a specific track from dataset_files/ folder"""
#         try:
#             # Check cache first
#             if self._check_track_data_cached(track_name):
#                 return self._load_cached_data(track_name)
            
#             print(f"🏁 Loading data for track: {track_name}")
            
#             track_data = {}
#             data_types = ['pit_data', 'race_data', 'telemetry_data', 'weather_data']
            
#             for data_type in data_types:
#                 # Construct blob path
#                 blob_path = f"dataset_files/{track_name}_{data_type}.csv"
#                 df = self._download_and_load_file(blob_path, data_type)
#                 track_data[data_type] = df
            
#             # Cache the loaded data
#             self._cache_track_data(track_name, track_data)
            
#             # Report loading statistics
#             loaded_count = sum(1 for df in track_data.values() if not df.empty)
#             print(f"📊 Loaded {loaded_count} data types for {track_name}")
            
#             return track_data
            
#         except Exception as e:
#             print(f"❌ Failed to load {track_name}: {e}")
#             return self._return_empty_data()

#     def _check_track_data_cached(self, track_name: str) -> bool:
#         """Check if track data is cached"""
#         cache_file = f'/kaggle/working/{track_name}_cached.pkl'
#         return os.path.exists(cache_file)

#     def _load_cached_data(self, track_name: str) -> Dict[str, pd.DataFrame]:
#         """Load data from cache"""
#         cache_file = f'/kaggle/working/{track_name}_cached.pkl'
#         print(f"📂 Loading cached data for {track_name}")
#         with open(cache_file, 'rb') as f:
#             return joblib.load(f)

#     def _cache_track_data(self, track_name: str, data: Dict[str, pd.DataFrame]):
#         """Cache track data"""
#         try:
#             cache_file = f'/kaggle/working/{track_name}_cached.pkl'
#             with open(cache_file, 'wb') as f:
#                 joblib.dump(data, f)
#             print(f"💾 Cached {track_name} data")
#         except Exception as e:
#             print(f"⚠️ Could not cache {track_name}: {e}")

#     def load_all_tracks(self, tracks: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
#         """Load multiple tracks efficiently"""
#         all_data = {}
        
#         for i, track in enumerate(tracks, 1):
#             try:
#                 print(f"\n🏁 Processing track {i}/{len(tracks)}: {track}")
#                 all_data[track] = self.load_track_data(track)
#                 time.sleep(1)  # Avoid overwhelming Firebase
#             except Exception as e:
#                 print(f"⚠️ Failed to load {track}: {e}")
#                 all_data[track] = self._return_empty_data()
        
#         return all_data

#     def _return_empty_data(self) -> Dict[str, pd.DataFrame]:
#         """Return empty data structure with proper schemas"""
#         return {
#             'pit_data': pd.DataFrame(),
#             'race_data': pd.DataFrame(),
#             'telemetry_data': pd.DataFrame(),
#             'weather_data': pd.DataFrame()
#         }

   
#     def list_available_tracks(self) -> List[str]:
#         """List all available tracks in dataset_files folder"""
#         tracks = set()
#         try:
#             blobs = self.bucket.list_blobs(prefix="dataset_files/")
#             for blob in blobs:
#                 if blob.name.endswith('.csv'):
#                     # Extract track name from filename (e.g., "dataset_files/road_america_pit_data.csv" -> "road_america")
#                     filename = os.path.basename(blob.name)
#                     # Remove file extension and split by underscore
#                     base_name = filename.replace('.csv', '')
#                     parts = base_name.split('_')
                    
#                     # Handle special case for road_america
#                     if len(parts) >= 2 and parts[0] == 'road' and parts[1] == 'america':
#                         track_name = 'road_america'
#                     else:
#                         track_name = parts[0]  # Default to first part for simple names
                    
#                     if track_name:
#                         tracks.add(track_name)
            
#             print(f"📦 Found {len(tracks)} tracks: {sorted(list(tracks))}")
#             return sorted(list(tracks))
            
#         except Exception as e:
#             print(f"❌ Failed to list tracks: {e}")
#             return []

#     def load_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
#         """Load all available tracks"""
#         available_tracks = self.list_available_tracks()
#         return self.load_all_tracks(available_tracks)

#     def validate_data_quality(self, track_data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
#         """Validate data quality with schema compliance"""
#         validation = {}
        
#         for data_type, df in track_data.items():
#             if df.empty:
#                 validation[data_type] = {'status': 'missing', 'rows': 0, 'columns': 0}
#             else:
#                 # Check schema compliance
#                 schema = self.schemas[data_type]
#                 missing_required = [col for col in schema['required_cols'] if col not in df.columns]
                
#                 validation[data_type] = {
#                     'status': 'loaded',
#                     'rows': len(df),
#                     'columns': len(df.columns),
#                     'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
#                     'schema_compliant': len(missing_required) == 0,
#                     'missing_required_cols': missing_required,
#                     'actual_columns': list(df.columns)
#                 }
        
#         return validation
    
#     def upload_models_to_firebase(self, models_dir: str = "outputs/models"):
#         """Upload all trained models to Firebase Storage after training completes"""
#         try:
#             if not os.path.exists(models_dir):
#                 print(f"❌ Models directory not found: {models_dir}")
#                 return False
                
#             model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
#             if not model_files:
#                 print("❌ No model files found to upload")
#                 return False
                
#             for model_file in model_files:
#                 local_path = os.path.join(models_dir, model_file)
#                 blob_path = f"trained_models/{model_file}"
                
#                 blob = self.bucket.blob(blob_path)
#                 blob.upload_from_filename(local_path)
#                 print(f"✅ Uploaded {model_file} to Firebase Storage")
                
#             print("🎉 All models successfully uploaded to Firebase!")
#             return True
#         except Exception as e:
#             print(f"❌ Failed to upload models: {e}")
#             return False
    
#     def download_models_from_firebase(self, local_dir: str = "outputs/models"):
#         """Download models from Firebase Storage"""
#         try:
#             os.makedirs(local_dir, exist_ok=True)
#             blobs = self.bucket.list_blobs(prefix="trained_models/")
            
#             downloaded_count = 0
#             for blob in blobs:
#                 if blob.name.endswith('.pkl'):
#                     local_path = os.path.join(local_dir, os.path.basename(blob.name))
#                     blob.download_to_filename(local_path)
#                     downloaded_count += 1
#                     print(f"✅ Downloaded {os.path.basename(blob.name)} from Firebase")
            
#             if downloaded_count > 0:
#                 print(f"🎉 Downloaded {downloaded_count} models from Firebase!")
#             else:
#                 print("📝 No models found in Firebase Storage")
                
#             return downloaded_count > 0
#         except Exception as e:
#             print(f"❌ Failed to download models: {e}")
#             return False






















# import firebase_admin
# from firebase_admin import credentials, storage
# import pandas as pd
# import io
# import os, json, base64
# import joblib
# from typing import List, Dict, Optional
# import time
# import numpy as np
# from datetime import datetime

# class FirebaseDataLoader:
#     def __init__(self, bucket_name: str):
#         if not firebase_admin._apps:
#             cred_json = base64.b64decode(os.getenv("FIREBASE_CREDENTIALS_BASE64"))
#             cred_dict = json.loads(cred_json)
#             cred = credentials.Certificate(cred_dict)
#             firebase_admin.initialize_app(cred, {
#                 'storageBucket': bucket_name
#             })
#         self.bucket = storage.bucket()
        
#         # Define strict schemas for each data type
#         self.schemas = {
#             'pit_data': {
#                 'required_cols': ['NUMBER', 'DRIVER_NUMBER', 'LAP_NUMBER', 'LAP_TIME'],
#                 'dtype_mapping': {
#                     'NUMBER': 'int64', 'DRIVER_NUMBER': 'int64', 'LAP_NUMBER': 'int64',
#                     'LAP_TIME': 'str', 'LAP_IMPROVEMENT': 'float64', 'KPH': 'float64',
#                     'TOP_SPEED': 'float64', 'PIT_TIME': 'float64', 'S1_SECONDS': 'float64',
#                     'S2_SECONDS': 'float64', 'S3_SECONDS': 'float64'
#                 },
#                 'default_values': {
#                     'LAP_IMPROVEMENT': 0.0, 'KPH': 0.0, 'TOP_SPEED': 0.0,
#                     'PIT_TIME': 0.0, 'S1_SECONDS': 0.0, 'S2_SECONDS': 0.0, 'S3_SECONDS': 0.0
#                 }
#             },
#             'race_data': {
#                 'required_cols': ['POSITION', 'NUMBER', 'STATUS', 'LAPS', 'TOTAL_TIME'],
#                 'dtype_mapping': {
#                     'POSITION': 'int64', 'NUMBER': 'int64', 'LAPS': 'int64',
#                     'TOTAL_TIME': 'str', 'FL_LAPNUM': 'int64', 'FL_TIME': 'str',
#                     'FL_KPH': 'float64'
#                 },
#                 'default_values': {
#                     'FL_LAPNUM': 0, 'FL_KPH': 0.0, 'GAP_FIRST': '0.000',
#                     'GAP_PREVIOUS': '0.000'
#                 }
#             },
#             'telemetry_data': {
#                 'required_cols': ['timestamp', 'vehicle_id', 'lap', 'speed'],
#                 'dtype_mapping': {
#                     'timestamp': 'datetime64[ns]', 'vehicle_id': 'str', 'lap': 'int64',
#                     'outing': 'int64', 'accx_can': 'float64', 'accy_can': 'float64',
#                     'gear': 'int64', 'speed': 'float64'
#                 },
#                 'default_values': {
#                     'accx_can': 0.0, 'accy_can': 0.0, 'gear': 0, 'outing': 0
#                 }
#             },
#             'weather_data': {
#                 'required_cols': ['TIME_UTC_SECONDS', 'AIR_TEMP', 'TRACK_TEMP'],
#                 'dtype_mapping': {
#                     'TIME_UTC_SECONDS': 'int64', 'TIME_UTC_STR': 'str',
#                     'AIR_TEMP': 'float64', 'TRACK_TEMP': 'float64',
#                     'HUMIDITY': 'float64', 'PRESSURE': 'float64',
#                     'WIND_SPEED': 'float64', 'WIND_DIRECTION': 'int64',
#                     'RAIN': 'int64'
#                 },
#                 'default_values': {
#                     'TRACK_TEMP': 0.0, 'HUMIDITY': 0.0, 'PRESSURE': 0.0,
#                     'WIND_SPEED': 0.0, 'WIND_DIRECTION': 0, 'RAIN': 0
#                 }
#             }
#         }

#     def _enforce_schema(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
#         """Enforce strict schema on DataFrame"""
#         if df.empty:
#             return df
            
#         schema = self.schemas[data_type]
#         df_clean = df.copy()
        
#         # Ensure required columns exist
#         for col in schema['required_cols']:
#             if col not in df_clean.columns:
#                 raise ValueError(f"Missing required column '{col}' for {data_type}")
        
#         # Add missing columns with default values
#         for col, default_val in schema['default_values'].items():
#             if col not in df_clean.columns:
#                 df_clean[col] = default_val
        
#         # Convert data types
#         for col, dtype in schema['dtype_mapping'].items():
#             if col in df_clean.columns:
#                 try:
#                     if dtype == 'datetime64[ns]' and col == 'timestamp':
#                         df_clean[col] = pd.to_datetime(df_clean[col])
#                     else:
#                         df_clean[col] = df_clean[col].astype(dtype)
#                 except Exception as e:
#                     print(f"⚠️ Failed to convert {col} to {dtype}: {e}")
#                     # Fill with default if conversion fails
#                     if col in schema['default_values']:
#                         df_clean[col] = schema['default_values'][col]
        
#         # Remove completely empty columns
#         df_clean = df_clean.dropna(axis=1, how='all')
        
#         # Fill remaining NaN values
#         for col in df_clean.columns:
#             if df_clean[col].isna().any():
#                 if col in schema['default_values']:
#                     df_clean[col].fillna(schema['default_values'][col], inplace=True)
#                 elif df_clean[col].dtype in ['float64', 'int64']:
#                     df_clean[col].fillna(0, inplace=True)
#                 elif df_clean[col].dtype == 'object':
#                     df_clean[col].fillna('', inplace=True)
        
#         return df_clean

#     def _load_csv_with_schema(self, file_content: bytes, data_type: str, filename: str) -> pd.DataFrame:
#         """Load CSV file with schema enforcement"""
#         try:
#             # Try different separators
#             for sep in [';', ',', '\t']:
#                 try:
#                     df = pd.read_csv(io.BytesIO(file_content), sep=sep, low_memory=False)
#                     if len(df.columns) > 1 and len(df) > 0:
#                         print(f"✅ Loaded {filename} (sep: {sep})")
#                         return self._enforce_schema(df, data_type)
#                 except Exception:
#                     continue
            
#             print(f"❌ Could not load {filename} with any separator")
#             return pd.DataFrame()
            
#         except Exception as e:
#             print(f"⚠️ Failed to load {filename}: {e}")
#             return pd.DataFrame()

#     def _download_and_load_file(self, blob_path: str, data_type: str) -> pd.DataFrame:
#         """Download and load individual CSV file from Firebase"""
#         try:
#             blob = self.bucket.blob(blob_path)
#             if not blob.exists():
#                 print(f"📝 File not found: {blob_path}")
#                 return pd.DataFrame()
            
#             file_content = blob.download_as_bytes()
#             filename = os.path.basename(blob_path)
            
#             print(f"📥 Downloading {filename}...")
#             df = self._load_csv_with_schema(file_content, data_type, filename)
            
#             if not df.empty:
#                 print(f"✅ Successfully loaded {filename} with {len(df)} rows")
#             else:
#                 print(f"❌ Failed to load valid data from {filename}")
            
#             return df
            
#         except Exception as e:
#             print(f"❌ Error downloading {blob_path}: {e}")
#             return pd.DataFrame()

#     def load_track_data(self, track_name: str) -> Dict[str, pd.DataFrame]:
#         """Load all data types for a specific track from dataset_files/ folder"""
#         try:
#             # Check cache first
#             if self._check_track_data_cached(track_name):
#                 return self._load_cached_data(track_name)
            
#             print(f"🏁 Loading data for track: {track_name}")
            
#             track_data = {}
#             data_types = ['pit_data', 'race_data', 'telemetry_data', 'weather_data']
            
#             for data_type in data_types:
#                 # Construct blob path
#                 blob_path = f"dataset_files/{track_name}_{data_type}.csv"
#                 df = self._download_and_load_file(blob_path, data_type)
#                 track_data[data_type] = df
            
#             # Cache the loaded data
#             self._cache_track_data(track_name, track_data)
            
#             # Report loading statistics
#             loaded_count = sum(1 for df in track_data.values() if not df.empty)
#             print(f"📊 Loaded {loaded_count} data types for {track_name}")
            
#             return track_data
            
#         except Exception as e:
#             print(f"❌ Failed to load {track_name}: {e}")
#             return self._return_empty_data()

#     def _check_track_data_cached(self, track_name: str) -> bool:
#         """Check if track data is cached"""
#         cache_file = f'/kaggle/working/{track_name}_cached.pkl'
#         return os.path.exists(cache_file)

#     def _load_cached_data(self, track_name: str) -> Dict[str, pd.DataFrame]:
#         """Load data from cache"""
#         cache_file = f'/kaggle/working/{track_name}_cached.pkl'
#         print(f"📂 Loading cached data for {track_name}")
#         with open(cache_file, 'rb') as f:
#             return joblib.load(f)

#     def _cache_track_data(self, track_name: str, data: Dict[str, pd.DataFrame]):
#         """Cache track data"""
#         try:
#             cache_file = f'/kaggle/working/{track_name}_cached.pkl'
#             with open(cache_file, 'wb') as f:
#                 joblib.dump(data, f)
#             print(f"💾 Cached {track_name} data")
#         except Exception as e:
#             print(f"⚠️ Could not cache {track_name}: {e}")

#     def load_all_tracks_optimized(self, tracks: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
#         """Load multiple tracks efficiently"""
#         all_data = {}
        
#         cached_tracks = [t for t in tracks if self._check_track_data_cached(t)]
#         missing_tracks = [t for t in tracks if not self._check_track_data_cached(t)]
        
#         print(f"📊 Track Status: {len(cached_tracks)} cached, {len(missing_tracks)} to download")
        
#         # Load cached tracks
#         for track in cached_tracks:
#             all_data[track] = self.load_track_data(track)
        
#         # Download missing tracks
#         for i, track in enumerate(missing_tracks, 1):
#             try:
#                 print(f"\n🏁 Downloading track {i}/{len(missing_tracks)}: {track}")
#                 all_data[track] = self.load_track_data(track)
#             except Exception as e:
#                 print(f"⚠️ Failed to load {track}: {e}")
#                 all_data[track] = self._return_empty_data()
        
#         return all_data

#     def _return_empty_data(self) -> Dict[str, pd.DataFrame]:
#         """Return empty data structure with proper schemas"""
#         return {
#             'pit_data': pd.DataFrame(),
#             'race_data': pd.DataFrame(),
#             'telemetry_data': pd.DataFrame(),
#             'weather_data': pd.DataFrame()
#         }

#     def list_available_tracks(self) -> List[str]:
#         """List all available tracks in dataset_files folder"""
#         tracks = set()
#         try:
#             blobs = self.bucket.list_blobs(prefix="dataset_files/")
#             for blob in blobs:
#                 if blob.name.endswith('.csv'):
#                     # Extract track name from filename (e.g., "dataset_files/monza_pit_data.csv" -> "monza")
#                     filename = os.path.basename(blob.name)
#                     track_name = filename.split('_')[0]  # Get part before first underscore
#                     if track_name:
#                         tracks.add(track_name)
            
#             print(f"📦 Found {len(tracks)} tracks: {sorted(list(tracks))}")
#             return sorted(list(tracks))
            
#         except Exception as e:
#             print(f"❌ Failed to list tracks: {e}")
#             return []

#     def load_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
#         """Load all available tracks"""
#         available_tracks = self.list_available_tracks()
#         return self.load_all_tracks_optimized(available_tracks)

#     def validate_data_quality(self, track_data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
#         """Validate data quality with schema compliance"""
#         validation = {}
        
#         for data_type, df in track_data.items():
#             if df.empty:
#                 validation[data_type] = {'status': 'missing', 'rows': 0, 'columns': 0}
#             else:
#                 # Check schema compliance
#                 schema = self.schemas[data_type]
#                 missing_required = [col for col in schema['required_cols'] if col not in df.columns]
                
#                 validation[data_type] = {
#                     'status': 'loaded',
#                     'rows': len(df),
#                     'columns': len(df.columns),
#                     'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
#                     'schema_compliant': len(missing_required) == 0,
#                     'missing_required_cols': missing_required
#                 }
        
#         return validation

#     def upload_models_to_firebase(self, models_dir: str = "outputs/models"):
#         """Upload trained models to Firebase"""
#         try:
#             if not os.path.exists(models_dir):
#                 print(f"❌ Models directory not found: {models_dir}")
#                 return False
                
#             model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
#             for model_file in model_files:
#                 local_path = os.path.join(models_dir, model_file)
#                 blob_path = f"trained_models/{model_file}"
#                 blob = self.bucket.blob(blob_path)
#                 blob.upload_from_filename(local_path)
#                 print(f"✅ Uploaded {model_file}")
                
#             return True
#         except Exception as e:
#             print(f"❌ Failed to upload models: {e}")
#             return False

#     def download_models_from_firebase(self, local_dir: str = "outputs/models"):
#         """Download models from Firebase"""
#         try:
#             os.makedirs(local_dir, exist_ok=True)
#             blobs = self.bucket.list_blobs(prefix="trained_models/")
            
#             downloaded_count = 0
#             for blob in blobs:
#                 if blob.name.endswith('.pkl'):
#                     local_path = os.path.join(local_dir, os.path.basename(blob.name))
#                     blob.download_to_filename(local_path)
#                     downloaded_count += 1
#                     print(f"✅ Downloaded {os.path.basename(blob.name)}")
            
#             return downloaded_count > 0
#         except Exception as e:
#             print(f"❌ Failed to download models: {e}")
#             return False