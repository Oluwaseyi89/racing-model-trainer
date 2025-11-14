import firebase_admin
from firebase_admin import credentials, storage
import pandas as pd
import zipfile
import io
import os, json, base64
from typing import List, Dict

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
    
    def _find_files_by_patterns(self, file_list: List[str], patterns: List[List[str]]) -> Dict[str, str]:
        """Find files matching specific patterns for different data types"""
        found_files = {}
        
        # Define search patterns for each data type
        search_patterns = {
            'lap_data': [['lap_ti', 'lap_time', 'laptime', 'timing', 'analysis']],
            'race_data': [['results', 'classification', 'standings', 'provisional']],
            'weather_data': [['weather', 'environment', 'conditions']],
            'telemetry_data': [['telemetry', 'tele', 'sensor', 'vbox']]
        }
        
        for data_type, patterns_list in search_patterns.items():
            for pattern_group in patterns_list:
                for file_path in file_list:
                    # Extract just the filename for pattern matching
                    filename = os.path.basename(file_path).lower()
                    if any(pattern in filename for pattern in pattern_group):
                        # Check if it's a CSV file
                        if filename.endswith('.csv') or filename.endswith('.csf'):
                            found_files[data_type] = file_path
                            break
                if data_type in found_files:
                    break
        
        return found_files
    
    def _extract_all_files_from_zip(self, zip_data: bytes) -> Dict[str, pd.DataFrame]:
        """Extract and load all CSV files from a zip archive with nested directories"""
        data_frames = {}
        
        try:
            with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as z:
                # Get all files in zip (including nested directories)
                all_files = []
                for file_info in z.infolist():
                    if not file_info.is_dir():
                        all_files.append(file_info.filename)
                
                print(f"📂 Found {len(all_files)} files in zip")
                
                # Find CSV files using pattern matching
                found_files = self._find_files_by_patterns(all_files, [])
                print(f"🔍 Pattern matching found: {found_files}")
                
                # Load each found file
                for data_type, file_path in found_files.items():
                    try:
                        with z.open(file_path) as f:
                            file_content = f.read()
                            
                            # Try different separators and encodings
                            for sep in [';', ',', '\t']:
                                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                                    try:
                                        # Reset to beginning of file content for each attempt
                                        df = pd.read_csv(io.BytesIO(file_content), sep=sep, low_memory=False, encoding=encoding)
                                        if len(df.columns) > 1 and len(df) > 0:  # Valid CSV with data
                                            data_frames[data_type] = df
                                            print(f"✅ Loaded {data_type} from {file_path} (sep: {sep}, encoding: {encoding})")
                                            break
                                    except UnicodeDecodeError:
                                        continue
                                    except Exception as e:
                                        continue
                                if data_type in data_frames:
                                    break
                                    
                            if data_type not in data_frames:
                                print(f"❌ Could not load {data_type} from {file_path} with any separator/encoding")
                                
                    except Exception as e:
                        print(f"⚠️ Failed to load {data_type} from {file_path}: {e}")
                
        except Exception as e:
            print(f"❌ Failed to process zip file: {e}")
        
        return data_frames
    
    def load_track_data(self, track_name: str) -> Dict[str, pd.DataFrame]:
        """Load all data files for a specific track from Firebase Storage"""
        try:
            blob_path = f"datasets/{track_name}.zip"
            print(f"📥 Attempting to download: {blob_path}")
            
            blob = self.bucket.blob(blob_path)
            
            # Check if blob exists
            if not blob.exists():
                print(f"❌ Blob does not exist: {blob_path}")
                return self._return_empty_data()
                
            zip_data = blob.download_as_bytes()
            print(f"✅ Downloaded {track_name}.zip ({len(zip_data)} bytes) from Firebase Storage")
            
            # Extract and load all files from zip
            track_data = self._extract_all_files_from_zip(zip_data)
            
            # Ensure we have the expected data types
            expected_types = ['lap_data', 'race_data', 'weather_data', 'telemetry_data']
            for data_type in expected_types:
                if data_type not in track_data:
                    track_data[data_type] = pd.DataFrame()
                    print(f"📝 Created empty DataFrame for {data_type}")
            
            # Report what was loaded
            loaded_count = sum(1 for df in track_data.values() if not df.empty)
            print(f"📊 Loaded {loaded_count} data types for {track_name}")
            
            return track_data
            
        except Exception as e:
            print(f"❌ Failed to load {track_name}: {e}")
            return self._return_empty_data()
    
    def _return_empty_data(self) -> Dict[str, pd.DataFrame]:
        """Return empty data structure"""
        return {
            'lap_data': pd.DataFrame(),
            'race_data': pd.DataFrame(),
            'weather_data': pd.DataFrame(),
            'telemetry_data': pd.DataFrame()
        }
    
    def load_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load all available tracks - MAIN METHOD CALLED BY ORCHESTRATOR"""
        available_tracks = self.list_available_tracks()
        print(f"📁 Found {len(available_tracks)} tracks: {available_tracks}")
        return self.load_all_tracks(available_tracks)
    
    def load_all_tracks(self, tracks: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load multiple tracks for combined training"""
        all_data = {}
        for track in tracks:
            try:
                print(f"\n🏁 Loading {track}...")
                all_data[track] = self.load_track_data(track)
            except Exception as e:
                print(f"⚠️ Failed to load {track}: {e}")
                all_data[track] = self._return_empty_data()
        return all_data
    
    def list_available_tracks(self) -> List[str]:
        """List all available tracks in Firebase Storage"""
        tracks = set()
        try:
            blobs = self.bucket.list_blobs(prefix="datasets/")
            for blob in blobs:
                if blob.name.endswith('.zip'):
                    # Extract track name from filename (e.g., "datasets/COTA.zip" -> "COTA")
                    track_name = os.path.basename(blob.name).replace('.zip', '')
                    tracks.add(track_name)
                    print(f"📦 Found: {track_name}")
        except Exception as e:
            print(f"❌ Failed to list tracks: {e}")
        
        return sorted(list(tracks))
    
    def validate_data_quality(self, track_data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """Validate the quality and completeness of loaded data"""
        validation = {}
        
        for data_type, df in track_data.items():
            if df.empty:
                validation[data_type] = {'status': 'missing', 'rows': 0, 'columns': 0}
            else:
                validation[data_type] = {
                    'status': 'loaded',
                    'rows': len(df),
                    'columns': len(df.columns),
                    'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0
                }
        
        return validation













# import firebase_admin
# from firebase_admin import credentials, storage
# import pandas as pd
# import zipfile
# import io
# import os, json, base64
# from typing import List, Dict

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
    
#     def _find_files_by_patterns(self, file_list: List[str], patterns: List[List[str]]) -> Dict[str, str]:
#         """Find files matching specific patterns for different data types"""
#         found_files = {}
        
#         # Define search patterns for each data type
#         search_patterns = {
#             'lap_data': [['lap_ti', 'lap_time', 'laptime', 'timing', 'analysis']],
#             'race_data': [['results', 'classification', 'standings', 'provisional']],
#             'weather_data': [['weather', 'environment', 'conditions']],
#             'telemetry_data': [['telemetry', 'tele', 'sensor', 'vbox']]
#         }
        
#         for data_type, patterns_list in search_patterns.items():
#             for pattern_group in patterns_list:
#                 for file_path in file_list:
#                     # Extract just the filename for pattern matching
#                     filename = os.path.basename(file_path).lower()
#                     if any(pattern in filename for pattern in pattern_group):
#                         # Check if it's a CSV file
#                         if filename.endswith('.csv') or filename.endswith('.csf'):
#                             found_files[data_type] = file_path
#                             break
#                 if data_type in found_files:
#                     break
        
#         return found_files
    
#     def _extract_all_files_from_zip(self, zip_data: bytes) -> Dict[str, pd.DataFrame]:
#         """Extract and load all CSV files from a zip archive with nested directories"""
#         data_frames = {}
        
#         try:
#             with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as z:
#                 # Get all files in zip (including nested directories)
#                 all_files = []
#                 for file_info in z.infolist():
#                     if not file_info.is_dir():
#                         all_files.append(file_info.filename)
                
#                 # Find CSV files using pattern matching
#                 found_files = self._find_files_by_patterns(all_files, [])
                
#                 # Load each found file
#                 for data_type, file_path in found_files.items():
#                     try:
#                         with z.open(file_path) as f:
#                             # Try different separators and encodings
#                             for sep in [';', ',', '\t']:
#                                 for encoding in ['utf-8', 'latin-1']:
#                                     try:
#                                         df = pd.read_csv(f, sep=sep, low_memory=False, encoding=encoding)
#                                         if len(df.columns) > 1:  # Valid CSV with multiple columns
#                                             data_frames[data_type] = df
#                                             print(f"✅ Loaded {data_type} from {file_path}")
#                                             break
#                                     except UnicodeDecodeError:
#                                         continue
#                                     except:
#                                         continue
#                                 if data_type in data_frames:
#                                     break
#                     except Exception as e:
#                         print(f"⚠️ Failed to load {data_type} from {file_path}: {e}")
                
#         except Exception as e:
#             print(f"❌ Failed to process zip file: {e}")
        
#         return data_frames
    
#     def load_track_data(self, track_name: str) -> Dict[str, pd.DataFrame]:
#         """Load all data files for a specific track from Firebase Storage"""
#         try:
#             blob = self.bucket.blob(f"datasets/{track_name}.zip")
#             zip_data = blob.download_as_bytes()
#             print(f"✅ Downloaded {track_name}.zip from Firebase Storage")
            
#             # Extract and load all files from zip
#             track_data = self._extract_all_files_from_zip(zip_data)
            
#             # Ensure we have the expected data types
#             expected_types = ['lap_data', 'race_data', 'weather_data', 'telemetry_data']
#             for data_type in expected_types:
#                 if data_type not in track_data:
#                     track_data[data_type] = pd.DataFrame()
#                     print(f"📝 Created empty DataFrame for {data_type}")
            
#             # Report what was loaded
#             loaded_count = sum(1 for df in track_data.values() if not df.empty)
#             print(f"📊 Loaded {loaded_count} data types for {track_name}")
            
#             return track_data
            
#         except Exception as e:
#             print(f"❌ Failed to load {track_name}: {e}")
#             # Return empty structure on failure
#             return {
#                 'lap_data': pd.DataFrame(),
#                 'race_data': pd.DataFrame(),
#                 'weather_data': pd.DataFrame(),
#                 'telemetry_data': pd.DataFrame()
#             }
    
#     def load_all_tracks(self, tracks: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
#         """Load multiple tracks for combined training"""
#         all_data = {}
#         for track in tracks:
#             try:
#                 all_data[track] = self.load_track_data(track)
#             except Exception as e:
#                 print(f"⚠️ Failed to load {track}: {e}")
#                 all_data[track] = {
#                     'lap_data': pd.DataFrame(),
#                     'race_data': pd.DataFrame(),
#                     'weather_data': pd.DataFrame(),
#                     'telemetry_data': pd.DataFrame()
#                 }
#         return all_data
    
#     def list_available_tracks(self) -> List[str]:
#         """List all available tracks in Firebase Storage"""
#         tracks = set()
#         try:
#             blobs = self.bucket.list_blobs(prefix="datasets/")
#             for blob in blobs:
#                 if blob.name.endswith('.zip'):
#                     # Extract track name from filename (e.g., "datasets/COTA.zip" -> "COTA")
#                     track_name = os.path.basename(blob.name).replace('.zip', '')
#                     tracks.add(track_name)
#         except Exception as e:
#             print(f"❌ Failed to list tracks: {e}")
        
#         return sorted(list(tracks))
    
#     def validate_data_quality(self, track_data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
#         """Validate the quality and completeness of loaded data"""
#         validation = {}
        
#         for data_type, df in track_data.items():
#             if df.empty:
#                 validation[data_type] = {'status': 'missing', 'rows': 0, 'columns': 0}
#             else:
#                 validation[data_type] = {
#                     'status': 'loaded',
#                     'rows': len(df),
#                     'columns': len(df.columns),
#                     'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0
#                 }
        
#         return validation

# # Example usage
# if __name__ == "__main__":
#     # Initialize with your Firebase bucket name
#     loader = FirebaseDataLoader("your-firebase-bucket-name")
    
#     # List available tracks
#     available_tracks = loader.list_available_tracks()
#     print(f"Available tracks: {available_tracks}")
    
#     # Load specific track
#     if available_tracks:
#         track_data = loader.load_track_data(available_tracks[0])
        
#         # Validate data quality
#         quality_report = loader.validate_data_quality(track_data)
#         for data_type, report in quality_report.items():
#             print(f"{data_type}: {report}")















# import firebase_admin
# from firebase_admin import credentials, storage
# import pandas as pd
# import zipfile
# import io
# import os, json, base64

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
    
#     def _find_file_by_pattern(self, file_list, patterns):
#         """Find file matching any of the patterns"""
#         for pattern in patterns:
#             for file in file_list:
#                 if any(p in file.lower() for p in pattern):
#                     return file
#         return None
    
#     def load_track_data(self, track_name: str) -> dict:
#         """Load all data files for a specific track"""
#         blob = self.bucket.blob(f"datasets/{track_name}.zip")
#         zip_data = blob.download_as_bytes()
        
#         with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
#             files = z.namelist()
            
#             # Find files by multiple possible patterns
#             lap_file = self._find_file_by_pattern(files, [['lap'], ['laptime'], ['lap_time']])
#             race_file = self._find_file_by_pattern(files, [['race'], ['result'], ['classification'], ['standings']])
#             weather_file = self._find_file_by_pattern(files, [['weather'], ['environment'], ['conditions']])
            
#             return {
#                 'lap_data': pd.read_csv(z.open(lap_file), sep=';') if lap_file else pd.DataFrame(),
#                 'race_data': pd.read_csv(z.open(race_file), sep=';') if race_file else pd.DataFrame(),
#                 'weather_data': pd.read_csv(z.open(weather_file), sep=';') if weather_file else pd.DataFrame()
#             }
    
#     def load_all_tracks(self, tracks: list) -> dict:
#         """Load multiple tracks for combined training"""
#         all_data = {}
#         for track in tracks:
#             try:
#                 all_data[track] = self.load_track_data(track)
#             except Exception as e:
#                 print(f"⚠️ Failed to load {track}: {e}")
#                 all_data[track] = {'lap_data': pd.DataFrame(), 'race_data': pd.DataFrame(), 'weather_data': pd.DataFrame()}
#         return all_data













# import firebase_admin
# from firebase_admin import credentials, storage
# import pandas as pd
# import zipfile
# import io
# import os, json, base64


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
    
#     def load_track_data(self, track_name: str) -> dict:
#         """Load all data files for a specific track"""
#         blob = self.bucket.blob(f"datasets/{track_name}.zip")
#         zip_data = blob.download_as_bytes()
        
#         with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
#             return {
#                 'lap_data': pd.read_csv(z.open('lap_times.csv'), sep=';'),
#                 'race_data': pd.read_csv(z.open('race_results.csv'), sep=';'),
#                 'weather_data': pd.read_csv(z.open('weather.csv'), sep=';')
#             }
    
#     def load_all_tracks(self, tracks: list) -> dict:
#         """Load multiple tracks for combined training"""
#         all_data = {}
#         for track in tracks:
#             all_data[track] = self.load_track_data(track)
#         return all_data