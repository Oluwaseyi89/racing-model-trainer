# import os
# import io
# import json
# import base64
# import zipfile
# from typing import List, Dict, Optional
# import pandas as pd
# import joblib
# import firebase_admin
# from firebase_admin import credentials, storage


# class FirebaseDataLoader:
#     """Robust loader for track data and model artifacts from Firebase Storage."""

#     DATA_TYPES = ['lap_data', 'race_data', 'weather_data', 'telemetry_data']

#     SEARCH_PATTERNS = {
#         'lap_data': [['lap_ti', 'lap_time', 'laptime', 'timing', 'analysis']],
#         'race_data': [['results', 'classification', 'standings', 'provisional']],
#         'weather_data': [['weather', 'environment', 'conditions']],
#         'telemetry_data': [['telemetry', 'tele', 'sensor', 'vbox']]
#     }

#     def __init__(self, bucket_name: str, debug: bool = False, cache_dir: str = '/kaggle/working'):
#         self.debug = debug
#         self.cache_dir = cache_dir

#         if not firebase_admin._apps:
#             cred_json_b64 = os.getenv("FIREBASE_CREDENTIALS_BASE64")
#             if not cred_json_b64:
#                 raise ValueError("Firebase credentials not found in environment variable 'FIREBASE_CREDENTIALS_BASE64'")
#             cred_json = base64.b64decode(cred_json_b64)
#             cred_dict = json.loads(cred_json)
#             cred = credentials.Certificate(cred_dict)
#             firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})

#         self.bucket = storage.bucket()
#         self._log(f"✅ Firebase initialized for bucket: {bucket_name}")

#     # --------------------
#     # Logging helper
#     # --------------------
#     def _log(self, *args):
#         if self.debug:
#             print(*args)

#     # --------------------
#     # Core loading helpers
#     # --------------------
#     def _find_files_by_patterns(self, file_list: List[str]) -> Dict[str, List[str]]:
#         """Return dictionary of files categorized by data type using SEARCH_PATTERNS."""
#         found_files = {dt: [] for dt in self.DATA_TYPES}

#         for data_type, pattern_groups in self.SEARCH_PATTERNS.items():
#             for pattern_group in pattern_groups:
#                 for file_path in file_list:
#                     filename = os.path.basename(file_path).lower()
#                     if any(p in filename for p in pattern_group) and filename.endswith(('.csv', '.csf')):
#                         if file_path not in found_files[data_type]:
#                             found_files[data_type].append(file_path)

#         return found_files

#     def _load_csv_file(self, zip_file: zipfile.ZipFile, file_path: str) -> pd.DataFrame:
#         """Load a single CSV file with multiple encoding and separator attempts."""
#         try:
#             with zip_file.open(file_path) as f:
#                 content = f.read()
#         except Exception as e:
#             self._log(f"⚠️ Could not open {file_path}: {e}")
#             return pd.DataFrame()

#         for sep in [';', ',', '\t']:
#             for enc in ['utf-8', 'latin-1', 'cp1252']:
#                 try:
#                     df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=enc, low_memory=False)
#                     if len(df.columns) > 1 and len(df) > 0:
#                         self._log(f"✅ Loaded {file_path} (sep='{sep}', encoding='{enc}')")
#                         return df
#                 except Exception:
#                     continue

#         self._log(f"❌ Failed to load {file_path} with all separators/encodings")
#         return pd.DataFrame()

#     def _extract_all_files_from_zip(self, zip_data: bytes) -> Dict[str, pd.DataFrame]:
#         """Extract all CSVs from zip, categorize by data type, and combine."""
#         data_frames = {dt: pd.DataFrame() for dt in self.DATA_TYPES}

#         try:
#             with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as z:
#                 all_files = [f.filename for f in z.infolist() if not f.is_dir()]
#                 self._log(f"📂 Found {len(all_files)} files in zip")

#                 categorized_files = self._find_files_by_patterns(all_files)

#                 for data_type, files in categorized_files.items():
#                     if not files:
#                         self._log(f"📝 No files found for {data_type}")
#                         continue
#                     dfs = [self._load_csv_file(z, f) for f in files]
#                     dfs = [d for d in dfs if not d.empty]
#                     if dfs:
#                         data_frames[data_type] = pd.concat(dfs, ignore_index=True)
#                         self._log(f"📊 Loaded {len(dfs)} files into {data_type} ({len(data_frames[data_type])} rows)")

#         except Exception as e:
#             self._log(f"❌ Failed to process zip file: {e}")

#         return data_frames

#     # --------------------
#     # Cache & Firebase helpers
#     # --------------------
#     def _cache_path(self, track_name: str) -> str:
#         return os.path.join(self.cache_dir, f"{track_name}_cached.pkl")

#     def _check_cache(self, track_name: str) -> bool:
#         return os.path.exists(self._cache_path(track_name))

#     def _check_exists_firebase(self, track_name: str) -> bool:
#         blob = self.bucket.blob(f"datasets/{track_name}.zip")
#         return blob.exists()

#     def _return_empty_data(self) -> Dict[str, pd.DataFrame]:
#         return {dt: pd.DataFrame() for dt in self.DATA_TYPES}

#     # --------------------
#     # Public track loading
#     # --------------------
#     def load_track_data(self, track_name: str) -> Dict[str, pd.DataFrame]:
#         """Load a single track, cache locally if not already cached."""
#         try:
#             # Load from cache
#             if self._check_cache(track_name):
#                 self._log(f"📂 Loading cached data for {track_name}")
#                 return joblib.load(self._cache_path(track_name))

#             # Check Firebase existence
#             if not self._check_exists_firebase(track_name):
#                 self._log(f"❌ Track {track_name} not found in Firebase")
#                 return self._return_empty_data()

#             # Download zip
#             blob = self.bucket.blob(f"datasets/{track_name}.zip")
#             zip_data = blob.download_as_bytes(timeout=300)
#             self._log(f"✅ Downloaded {track_name}.zip ({len(zip_data)} bytes)")

#             # Extract CSVs
#             track_data = self._extract_all_files_from_zip(zip_data)

#             # Cache
#             try:
#                 joblib.dump(track_data, self._cache_path(track_name))
#                 self._log(f"💾 Cached {track_name} for future use")
#             except Exception as e:
#                 self._log(f"⚠️ Failed to cache {track_name}: {e}")

#             return track_data

#         except Exception as e:
#             self._log(f"❌ Failed to load {track_name}: {e}")
#             return self._return_empty_data()

#     def load_multiple_tracks(self, tracks: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
#         """Load multiple tracks efficiently, only download missing ones."""
#         all_data = {}
#         for i, track in enumerate(tracks, 1):
#             self._log(f"🏁 Loading track {i}/{len(tracks)}: {track}")
#             all_data[track] = self.load_track_data(track)
#         return all_data

#     def list_available_tracks(self) -> List[str]:
#         """List all track zip files in Firebase."""
#         tracks = set()
#         try:
#             blobs = self.bucket.list_blobs(prefix="datasets/")
#             for b in blobs:
#                 if b.name.endswith('.zip'):
#                     track_name = os.path.basename(b.name).replace('.zip', '')
#                     tracks.add(track_name)
#         except Exception as e:
#             self._log(f"❌ Failed to list tracks: {e}")
#         return sorted(list(tracks))

#     # --------------------
#     # Validation
#     # --------------------
#     def validate_data_quality(self, track_data: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
#         """Return a summary of data quality for each data type."""
#         summary = {}
#         for dt, df in track_data.items():
#             if df.empty:
#                 summary[dt] = {'status': 'missing', 'rows': 0, 'columns': 0}
#             else:
#                 null_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
#                 summary[dt] = {'status': 'loaded', 'rows': len(df), 'columns': len(df.columns), 'null_percentage': null_pct}
#         return summary

#     # --------------------
#     # Model upload/download
#     # --------------------
#     def upload_models(self, models_dir: str = "outputs/models") -> bool:
#         """Upload all .pkl models in models_dir to Firebase."""
#         if not os.path.exists(models_dir):
#             self._log(f"❌ Models dir not found: {models_dir}")
#             return False
#         model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
#         if not model_files:
#             self._log("❌ No model files to upload")
#             return False
#         for f in model_files:
#             self.bucket.blob(f"trained_models/{f}").upload_from_filename(os.path.join(models_dir, f))
#             self._log(f"✅ Uploaded {f}")
#         self._log("🎉 All models uploaded")
#         return True

#     def download_models(self, local_dir: str = "outputs/models") -> bool:
#         """Download all .pkl models from Firebase to local_dir."""
#         os.makedirs(local_dir, exist_ok=True)
#         blobs = self.bucket.list_blobs(prefix="trained_models/")
#         count = 0
#         for b in blobs:
#             if b.name.endswith('.pkl'):
#                 b.download_to_filename(os.path.join(local_dir, os.path.basename(b.name)))
#                 self._log(f"✅ Downloaded {os.path.basename(b.name)}")
#                 count += 1
#         if count == 0:
#             self._log("📝 No models found in Firebase")
#         else:
#             self._log(f"🎉 Downloaded {count} models")
#         return count > 0




















import firebase_admin
from firebase_admin import credentials, storage
import pandas as pd
import zipfile
import io
import os, json, base64
import joblib
from typing import List, Dict, Optional
import time

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
    
    def _find_files_by_patterns(self, file_list: List[str], patterns: List[List[str]]) -> Dict[str, List[str]]:
        """Find ALL files matching specific patterns for different data types"""
        found_files = {
            'lap_data': [],
            'race_data': [], 
            'weather_data': [],
            'telemetry_data': []
        }
        
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
                            if file_path not in found_files[data_type]:  # Avoid duplicates
                                found_files[data_type].append(file_path)
        
        return found_files
    
    # def _load_csv_file(self, zip_file: zipfile.ZipFile, file_path: str) -> pd.DataFrame:
    #     """Load a single CSV file with multiple encoding attempts"""
    #     try:
    #         with zip_file.open(file_path) as f:
    #             file_content = f.read()
                
    #             # Try different separators and encodings
    #             for sep in [';', ',', '\t']:
    #                 for encoding in ['utf-8', 'latin-1', 'cp1252']:
    #                     try:
    #                         df = pd.read_csv(io.BytesIO(file_content), sep=sep, low_memory=False, encoding=encoding)
    #                         if len(df.columns) > 1 and len(df) > 0:  # Valid CSV with data
    #                             print(f"✅ Loaded from {file_path} (sep: {sep}, encoding: {encoding})")
    #                             return df
    #                     except UnicodeDecodeError:
    #                         continue
    #                     except Exception:
    #                         continue
                
    #             print(f"❌ Could not load {file_path} with any separator/encoding")
    #             return pd.DataFrame()
                
    #     except Exception as e:
    #         print(f"⚠️ Failed to load {file_path}: {e}")
    #         return pd.DataFrame()
    
    
    
    
    #------------------------------------------------------------------
    # _load_csv_file improvement

    def _load_csv_file(self, zip_file: zipfile.ZipFile, file_path: str) -> pd.DataFrame:
        """Load a single CSV file with proper error handling and validation"""
        try:
            with zip_file.open(file_path) as f:
                # Read sample first to detect encoding/separator
                sample_content = f.read(8192)  # 8KB sample
                # f.seek(0)  # Reset for full read
                
                # Detect encoding and separator more intelligently
                detected_sep, detected_encoding = self._detect_csv_format(sample_content)
                
                if detected_sep and detected_encoding:
                    df = pd.read_csv(f, sep=detected_sep, encoding=detected_encoding, 
                                low_memory=False, on_bad_lines='warn')
                    
                    if self._validate_dataframe(df, file_path):
                        print(f"✅ Loaded {file_path} (sep: {detected_sep}, encoding: {detected_encoding})")
                        return df
                
                # Fallback with early termination
                return self._fallback_load(f, file_path)
                
        except KeyError:
            raise FileNotFoundError(f"File {file_path} not found in zip")
        except Exception as e:
            raise ValueError(f"Failed to load {file_path}: {str(e)}") from e

    def _detect_csv_format(self, sample_content: bytes) -> tuple:
        """Intelligently detect CSV separator and encoding"""
        # Use chardet for encoding detection
        import chardet
        encoding_result = chardet.detect(sample_content)
        
        # Detect separator from first few lines
        decoded_sample = sample_content.decode(encoding_result['encoding'] or 'utf-8', errors='ignore')
        first_line = decoded_sample.split('\n')[0] if '\n' in decoded_sample else decoded_sample
        
        separators = {';': first_line.count(';'), 
                    ',': first_line.count(','), 
                    '\t': first_line.count('\t')}
        best_sep = max(separators, key=separators.get) if max(separators.values()) > 0 else ','
        
        return best_sep, encoding_result['encoding']

    def _validate_dataframe(self, df: pd.DataFrame, file_path: str) -> bool:
        """Validate DataFrame structure and content"""
        if df.empty:
            print(f"⚠️ Empty file: {file_path}")
            return True  # Empty but valid
        
        # Check for reasonable structure
        if len(df.columns) == 0:
            print(f"❌ No columns found: {file_path}")
            return False
            
        # Check for completely null data
        if df.isnull().all().all():
            print(f"⚠️ All null data: {file_path}")
            
        return True


    # _load_csv_file improvement ends
    #--------------------------------------------------------------------

    
    # def _extract_all_files_from_zip(self, zip_data: bytes) -> Dict[str, pd.DataFrame]:
    #     """Extract and load ALL CSV files from a zip archive with nested directories"""
    #     data_frames = {
    #         'lap_data': pd.DataFrame(),
    #         'race_data': pd.DataFrame(),
    #         'weather_data': pd.DataFrame(),
    #         'telemetry_data': pd.DataFrame()
    #     }
        
    #     try:
    #         with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as z:
    #             # Get all files in zip (including nested directories)
    #             all_files = []
    #             for file_info in z.infolist():
    #                 if not file_info.is_dir():
    #                     all_files.append(file_info.filename)
                
    #             print(f"📂 Found {len(all_files)} files in zip")
                
    #             # Find ALL CSV files using pattern matching
    #             found_files = self._find_files_by_patterns(all_files, [])
                
    #             # Load ALL files for each data type and concatenate them
    #             for data_type, file_paths in found_files.items():
    #                 if not file_paths:
    #                     print(f"📝 No files found for {data_type}")
    #                     continue
                    
    #                 print(f"🔍 Found {len(file_paths)} files for {data_type}: {file_paths}")
                    
    #                 dfs = []
    #                 for file_path in file_paths:
    #                     df = self._load_csv_file(z, file_path)
    #                     if not df.empty:
    #                         dfs.append(df)
                    
    #                 if dfs:
    #                     # Concatenate all dataframes for this data type
    #                     combined_df = pd.concat(dfs, ignore_index=True)
    #                     data_frames[data_type] = combined_df
    #                     print(f"📊 Combined {len(dfs)} files into {data_type} with {len(combined_df)} rows")
    #                 else:
    #                     print(f"❌ No valid data loaded for {data_type}")
                
    #     except Exception as e:
    #         print(f"❌ Failed to process zip file: {e}")
        
    #     return data_frames


    #-----------------------------------------------------------
    # _extract_all_files_from_zip Improvement

    def _extract_all_files_from_zip(self, zip_path: str) -> Dict[str, pd.DataFrame]:
        """Extract and load CSV files with proper memory management and error handling"""
        data_frames = {
            'lap_data': pd.DataFrame(),
            'race_data': pd.DataFrame(),
            'weather_data': pd.DataFrame(),
            'telemetry_data': pd.DataFrame()
        }
        
        processed_files = set()
        errors = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                # Process files incrementally to manage memory
                for file_info in z.infolist():
                    if file_info.is_dir() or not file_info.filename.lower().endswith('.csv'):
                        continue
                    
                    try:
                        # Determine data type from filename/path
                        data_type = self._classify_file_type(file_info.filename)
                        if not data_type:
                            continue
                        
                        # Load single file
                        df = self._load_csv_file(z, file_info.filename)
                        if df.empty:
                            errors.append(f"Empty DataFrame: {file_info.filename}")
                            continue
                        
                        # Validate schema before concatenation
                        if not self._validate_schema_compatibility(df, data_type):
                            errors.append(f"Schema mismatch: {file_info.filename}")
                            continue
                        
                        # Merge with existing data
                        data_frames[data_type] = self._merge_dataframes(
                            data_frames[data_type], df, data_type
                        )
                        processed_files.add(file_info.filename)
                        
                    except Exception as e:
                        errors.append(f"Failed {file_info.filename}: {str(e)}")
                        continue
                
        except zipfile.BadZipFile as e:
            raise ValueError(f"Invalid zip file: {str(e)}") from e
        except Exception as e:
            raise IOError(f"Zip processing failed: {str(e)}") from e
        
        # Return results with metadata
        result = {
            'data_frames': data_frames,
            'processed_files': list(processed_files),
            'errors': errors,
            'success': len(errors) == 0
        }
        
        return result

    def _merge_dataframes(self, existing_df: pd.DataFrame, new_df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Safely merge dataframes with schema validation"""
        if existing_df.empty:
            return new_df
        
        # Check for column compatibility
        common_cols = set(existing_df.columns) & set(new_df.columns)
        if not common_cols:
            raise ValueError(f"No common columns for {data_type}")
        
        # Ensure consistent dtypes for common columns
        for col in common_cols:
            if existing_df[col].dtype != new_df[col].dtype:
                new_df[col] = new_df[col].astype(existing_df[col].dtype)
        
        return pd.concat([existing_df, new_df], ignore_index=True, sort=False)

    # _extract_all_files_from_zip Improvement ends here
    #-----------------------------------------------------------------
    
    def _check_track_data_cached(self, track_name: str) -> bool:
        """Check if track data is already cached on Kaggle"""
        cache_file = f'/kaggle/working/{track_name}_cached.pkl'
        return os.path.exists(cache_file)
    
    def _check_track_exists_firebase(self, track_name: str) -> bool:
        """Check if track exists in Firebase Storage"""
        blob_path = f"datasets/{track_name}.zip"
        blob = self.bucket.blob(blob_path)
        return blob.exists()
    
    # def load_track_data(self, track_name: str) -> Dict[str, pd.DataFrame]:
    #     """Load track data - only download if not already cached"""
    #     try:
    #         # Check for cached data first
    #         if self._check_track_data_cached(track_name):
    #             cache_file = f'/kaggle/working/{track_name}_cached.pkl'
    #             print(f"📂 Loading cached data for {track_name}")
    #             with open(cache_file, 'rb') as f:
    #                 return joblib.load(f)
            
    #         # Check if track exists in Firebase before downloading
    #         if not self._check_track_exists_firebase(track_name):
    #             print(f"❌ Track {track_name} not found in Firebase Storage")
    #             return self._return_empty_data()
            
    #         # Download only if not cached
    #         blob_path = f"datasets/{track_name}.zip"
    #         print(f"📥 Downloading missing dataset: {blob_path}")
            
    #         blob = self.bucket.blob(blob_path)
            
    #         # Download with timeout protection
    #         try:
    #             zip_data = blob.download_as_bytes(timeout=300)  # 5-minute timeout
    #             print(f"✅ Downloaded {track_name}.zip ({len(zip_data)} bytes)")
    #         except Exception as e:
    #             print(f"⏰ Download timeout for {track_name}: {e}")
    #             return self._return_empty_data()
            
    #         # Extract and load files
    #         track_data = self._extract_all_files_from_zip(zip_data)
            
    #         # Cache immediately to prevent data loss on kernel restart
    #         try:
    #             cache_file = f'/kaggle/working/{track_name}_cached.pkl'
    #             with open(cache_file, 'wb') as f:
    #                 joblib.dump(track_data, f)
    #             print(f"💾 Cached {track_name} data for future sessions")
    #         except Exception as e:
    #             print(f"⚠️ Could not cache {track_name}: {e}")
            
    #         # Report what was loaded
    #         loaded_count = sum(1 for df in track_data.values() if not df.empty)
    #         print(f"📊 Loaded {loaded_count} data types for {track_name}")
            
    #         return track_data
            
    #     except Exception as e:
    #         print(f"❌ Failed to load {track_name}: {e}")
    #         return self._return_empty_data()


    #--------------------------------------------------------------
    # load_track_data Improvement

    def load_track_data(self, track_name: str) -> Dict[str, pd.DataFrame]:
        """Load track data - only download if not already cached"""
        try:
            # Check for cached data first (ORIGINAL LOGIC PRESERVED)
            if self._check_track_data_cached(track_name):
                cache_file = f'/kaggle/working/{track_name}_cached.pkl'
                print(f"📂 Loading cached data for {track_name}")
                with open(cache_file, 'rb') as f:
                    cached_data = joblib.load(f)
                    # Add validation before returning cached data
                    if self._is_valid_cached_data(cached_data):
                        return cached_data
                    else:
                        print("⚠️ Cached data invalid, re-downloading")
                        # Remove corrupted cache
                        os.remove(cache_file)
            
            # Original logic continues exactly as before...
            if not self._check_track_exists_firebase(track_name):
                print(f"❌ Track {track_name} not found in Firebase Storage")
                return self._return_empty_data()
            
            # Download with retry (minimal addition)
            zip_data = self._download_with_retry(track_name)
            if zip_data is None:
                return self._return_empty_data()
            
            # Extract files (ORIGINAL)
            track_data = self._extract_all_files_from_zip(zip_data)
            
            # Validate before caching (minimal addition)
            if not any(not df.empty for df in track_data.values()):
                print(f"⚠️ No valid data extracted, skipping cache for {track_name}")
                return track_data  # Return empty but don't cache
            
            # Cache immediately (ORIGINAL LOGIC PRESERVED)
            try:
                cache_file = f'/kaggle/working/{track_name}_cached.pkl'
                with open(cache_file, 'wb') as f:
                    joblib.dump(track_data, f)
                print(f"💾 Cached {track_name} data for future sessions")
            except Exception as e:
                print(f"⚠️ Could not cache {track_name}: {e}")
            
            # Clean up memory
            del zip_data
            
            return track_data
            
        except Exception as e:
            print(f"❌ Failed to load {track_name}: {e}")
            return self._return_empty_data()

    def _download_with_retry(self, track_name: str, max_retries: int = 2) -> Optional[bytes]:
        """Minimal retry logic without changing main flow"""
        blob_path = f"datasets/{track_name}.zip"
        
        for attempt in range(max_retries):
            try:
                blob = self.bucket.blob(blob_path)
                return blob.download_as_bytes(timeout=300)
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"⏰ Download timeout for {track_name}: {e}")
                    return None
                print(f"🔄 Retry {attempt + 1} for {track_name}")
                time.sleep(1)


    # load-track_data Improvement ends
    #-----------------------------------------------------------------
    
    def load_all_tracks_optimized(self, tracks: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load multiple tracks - only download missing datasets"""
        all_data = {}
        total_tracks = len(tracks)
        
        # Check which tracks are already cached
        cached_tracks = [t for t in tracks if self._check_track_data_cached(t)]
        missing_tracks = [t for t in tracks if not self._check_track_data_cached(t)]
        
        print(f"📊 Track Status: {len(cached_tracks)} cached, {len(missing_tracks)} need download")
        
        if cached_tracks:
            print(f"📂 Loading cached tracks: {cached_tracks}")
            for track in cached_tracks:
                all_data[track] = self.load_track_data(track)
        
        if missing_tracks:
            print(f"📥 Downloading missing tracks: {missing_tracks}")
            for i, track in enumerate(missing_tracks, 1):
                try:
                    print(f"\n🏁 Downloading track {i}/{len(missing_tracks)}: {track}")
                    all_data[track] = self.load_track_data(track)
                except Exception as e:
                    print(f"⚠️ Failed to load {track}: {e}")
                    all_data[track] = self._return_empty_data()
        
        return all_data
    
    def _return_empty_data(self) -> Dict[str, pd.DataFrame]:
        """Return empty data structure"""
        return {
            'lap_data': pd.DataFrame(),
            'race_data': pd.DataFrame(),
            'weather_data': pd.DataFrame(),
            'telemetry_data': pd.DataFrame()
        }
    
    def load_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load all available tracks - MAIN METHOD CALLED BY ORCHESTRATOR (backward compatible)"""
        available_tracks = self.list_available_tracks()
        print(f"📁 Found {len(available_tracks)} tracks: {available_tracks}")
        
        # Use optimized loading that only downloads missing datasets
        return self.load_all_tracks_optimized(available_tracks)
    
    def load_all_tracks(self, tracks: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load multiple tracks for combined training with progress tracking (backward compatible)"""
        # This maintains backward compatibility but uses optimized loading internally
        return self.load_all_tracks_optimized(tracks)
    
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
# import zipfile
# import io
# import os, json, base64
# import joblib
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
    
#     def _find_files_by_patterns(self, file_list: List[str], patterns: List[List[str]]) -> Dict[str, List[str]]:
#         """Find ALL files matching specific patterns for different data types"""
#         found_files = {
#             'lap_data': [],
#             'race_data': [], 
#             'weather_data': [],
#             'telemetry_data': []
#         }
        
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
#                             if file_path not in found_files[data_type]:  # Avoid duplicates
#                                 found_files[data_type].append(file_path)
        
#         return found_files
    
#     def _load_csv_file(self, zip_file: zipfile.ZipFile, file_path: str) -> pd.DataFrame:
#         """Load a single CSV file with multiple encoding attempts"""
#         try:
#             with zip_file.open(file_path) as f:
#                 file_content = f.read()
                
#                 # Try different separators and encodings
#                 for sep in [';', ',', '\t']:
#                     for encoding in ['utf-8', 'latin-1', 'cp1252']:
#                         try:
#                             df = pd.read_csv(io.BytesIO(file_content), sep=sep, low_memory=False, encoding=encoding)
#                             if len(df.columns) > 1 and len(df) > 0:  # Valid CSV with data
#                                 print(f"✅ Loaded from {file_path} (sep: {sep}, encoding: {encoding})")
#                                 return df
#                         except UnicodeDecodeError:
#                             continue
#                         except Exception:
#                             continue
                
#                 print(f"❌ Could not load {file_path} with any separator/encoding")
#                 return pd.DataFrame()
                
#         except Exception as e:
#             print(f"⚠️ Failed to load {file_path}: {e}")
#             return pd.DataFrame()
    
#     def _extract_all_files_from_zip(self, zip_data: bytes) -> Dict[str, pd.DataFrame]:
#         """Extract and load ALL CSV files from a zip archive with nested directories"""
#         data_frames = {
#             'lap_data': pd.DataFrame(),
#             'race_data': pd.DataFrame(),
#             'weather_data': pd.DataFrame(),
#             'telemetry_data': pd.DataFrame()
#         }
        
#         try:
#             with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as z:
#                 # Get all files in zip (including nested directories)
#                 all_files = []
#                 for file_info in z.infolist():
#                     if not file_info.is_dir():
#                         all_files.append(file_info.filename)
                
#                 print(f"📂 Found {len(all_files)} files in zip")
                
#                 # Find ALL CSV files using pattern matching
#                 found_files = self._find_files_by_patterns(all_files, [])
                
#                 # Load ALL files for each data type and concatenate them
#                 for data_type, file_paths in found_files.items():
#                     if not file_paths:
#                         print(f"📝 No files found for {data_type}")
#                         continue
                    
#                     print(f"🔍 Found {len(file_paths)} files for {data_type}: {file_paths}")
                    
#                     dfs = []
#                     for file_path in file_paths:
#                         df = self._load_csv_file(z, file_path)
#                         if not df.empty:
#                             dfs.append(df)
                    
#                     if dfs:
#                         # Concatenate all dataframes for this data type
#                         combined_df = pd.concat(dfs, ignore_index=True)
#                         data_frames[data_type] = combined_df
#                         print(f"📊 Combined {len(dfs)} files into {data_type} with {len(combined_df)} rows")
#                     else:
#                         print(f"❌ No valid data loaded for {data_type}")
                
#         except Exception as e:
#             print(f"❌ Failed to process zip file: {e}")
        
#         return data_frames
    
#     def load_track_data(self, track_name: str) -> Dict[str, pd.DataFrame]:
#         """Load track data with caching and timeout protection to prevent kernel restarts"""
#         try:
#             # Check for cached data first
#             cache_file = f'/kaggle/working/{track_name}_cached.pkl'
#             if os.path.exists(cache_file):
#                 print(f"📂 Loading cached data for {track_name}")
#                 with open(cache_file, 'rb') as f:
#                     return joblib.load(f)
            
#             blob_path = f"datasets/{track_name}.zip"
#             print(f"📥 Downloading: {blob_path}")
            
#             blob = self.bucket.blob(blob_path)
            
#             # Check if blob exists
#             if not blob.exists():
#                 print(f"❌ Blob does not exist: {blob_path}")
#                 return self._return_empty_data()
            
#             # Download with timeout protection
#             try:
#                 zip_data = blob.download_as_bytes(timeout=300)  # 5-minute timeout
#                 print(f"✅ Downloaded {track_name}.zip ({len(zip_data)} bytes)")
#             except Exception as e:
#                 print(f"⏰ Download timeout for {track_name}: {e}")
#                 return self._return_empty_data()
            
#             # Extract and load files
#             track_data = self._extract_all_files_from_zip(zip_data)
            
#             # Cache immediately to prevent data loss on kernel restart
#             try:
#                 with open(cache_file, 'wb') as f:
#                     joblib.dump(track_data, f)
#                 print(f"💾 Cached {track_name} data")
#             except Exception as e:
#                 print(f"⚠️ Could not cache {track_name}: {e}")
            
#             # Report what was loaded
#             loaded_count = sum(1 for df in track_data.values() if not df.empty)
#             print(f"📊 Loaded {loaded_count} data types for {track_name}")
            
#             return track_data
            
#         except Exception as e:
#             print(f"❌ Failed to load {track_name}: {e}")
#             return self._return_empty_data()
    
#     def _return_empty_data(self) -> Dict[str, pd.DataFrame]:
#         """Return empty data structure"""
#         return {
#             'lap_data': pd.DataFrame(),
#             'race_data': pd.DataFrame(),
#             'weather_data': pd.DataFrame(),
#             'telemetry_data': pd.DataFrame()
#         }
    
#     def load_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
#         """Load all available tracks - MAIN METHOD CALLED BY ORCHESTRATOR"""
#         available_tracks = self.list_available_tracks()
#         print(f"📁 Found {len(available_tracks)} tracks: {available_tracks}")
#         return self.load_all_tracks(available_tracks)
    
#     def load_all_tracks(self, tracks: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
#         """Load multiple tracks for combined training with progress tracking"""
#         all_data = {}
#         total_tracks = len(tracks)
        
#         for i, track in enumerate(tracks, 1):
#             try:
#                 print(f"\n🏁 Loading track {i}/{total_tracks}: {track}")
#                 all_data[track] = self.load_track_data(track)
                
#                 # Save progress after each track
#                 progress_file = f'/kaggle/working/loading_progress_{i}.pkl'
#                 with open(progress_file, 'wb') as f:
#                     joblib.dump(list(all_data.keys()), f)
#                 print(f"📈 Progress saved: {i}/{total_tracks} tracks loaded")
                
#             except Exception as e:
#                 print(f"⚠️ Failed to load {track}: {e}")
#                 all_data[track] = self._return_empty_data()
        
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
#                     print(f"📦 Found: {track_name}")
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
    
#     def _find_files_by_patterns(self, file_list: List[str], patterns: List[List[str]]) -> Dict[str, List[str]]:
#         """Find ALL files matching specific patterns for different data types"""
#         found_files = {
#             'lap_data': [],
#             'race_data': [], 
#             'weather_data': [],
#             'telemetry_data': []
#         }
        
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
#                             if file_path not in found_files[data_type]:  # Avoid duplicates
#                                 found_files[data_type].append(file_path)
        
#         return found_files
    
#     def _load_csv_file(self, zip_file: zipfile.ZipFile, file_path: str) -> pd.DataFrame:
#         """Load a single CSV file with multiple encoding attempts"""
#         try:
#             with zip_file.open(file_path) as f:
#                 file_content = f.read()
                
#                 # Try different separators and encodings
#                 for sep in [';', ',', '\t']:
#                     for encoding in ['utf-8', 'latin-1', 'cp1252']:
#                         try:
#                             df = pd.read_csv(io.BytesIO(file_content), sep=sep, low_memory=False, encoding=encoding)
#                             if len(df.columns) > 1 and len(df) > 0:  # Valid CSV with data
#                                 print(f"✅ Loaded from {file_path} (sep: {sep}, encoding: {encoding})")
#                                 return df
#                         except UnicodeDecodeError:
#                             continue
#                         except Exception:
#                             continue
                
#                 print(f"❌ Could not load {file_path} with any separator/encoding")
#                 return pd.DataFrame()
                
#         except Exception as e:
#             print(f"⚠️ Failed to load {file_path}: {e}")
#             return pd.DataFrame()
    
#     def _extract_all_files_from_zip(self, zip_data: bytes) -> Dict[str, pd.DataFrame]:
#         """Extract and load ALL CSV files from a zip archive with nested directories"""
#         data_frames = {
#             'lap_data': pd.DataFrame(),
#             'race_data': pd.DataFrame(),
#             'weather_data': pd.DataFrame(),
#             'telemetry_data': pd.DataFrame()
#         }
        
#         try:
#             with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as z:
#                 # Get all files in zip (including nested directories)
#                 all_files = []
#                 for file_info in z.infolist():
#                     if not file_info.is_dir():
#                         all_files.append(file_info.filename)
                
#                 print(f"📂 Found {len(all_files)} files in zip")
                
#                 # Find ALL CSV files using pattern matching
#                 found_files = self._find_files_by_patterns(all_files, [])
                
#                 # Load ALL files for each data type and concatenate them
#                 for data_type, file_paths in found_files.items():
#                     if not file_paths:
#                         print(f"📝 No files found for {data_type}")
#                         continue
                    
#                     print(f"🔍 Found {len(file_paths)} files for {data_type}: {file_paths}")
                    
#                     dfs = []
#                     for file_path in file_paths:
#                         df = self._load_csv_file(z, file_path)
#                         if not df.empty:
#                             dfs.append(df)
                    
#                     if dfs:
#                         # Concatenate all dataframes for this data type
#                         combined_df = pd.concat(dfs, ignore_index=True)
#                         data_frames[data_type] = combined_df
#                         print(f"📊 Combined {len(dfs)} files into {data_type} with {len(combined_df)} rows")
#                     else:
#                         print(f"❌ No valid data loaded for {data_type}")
                
#         except Exception as e:
#             print(f"❌ Failed to process zip file: {e}")
        
#         return data_frames
    
#     def load_track_data(self, track_name: str) -> Dict[str, pd.DataFrame]:
#         """Load all data files for a specific track from Firebase Storage"""
#         try:
#             blob_path = f"datasets/{track_name}.zip"
#             print(f"📥 Attempting to download: {blob_path}")
            
#             blob = self.bucket.blob(blob_path)
            
#             # Check if blob exists
#             if not blob.exists():
#                 print(f"❌ Blob does not exist: {blob_path}")
#                 return self._return_empty_data()
                
#             zip_data = blob.download_as_bytes()
#             print(f"✅ Downloaded {track_name}.zip ({len(zip_data)} bytes) from Firebase Storage")
            
#             # Extract and load all files from zip
#             track_data = self._extract_all_files_from_zip(zip_data)
            
#             # Report what was loaded
#             loaded_count = sum(1 for df in track_data.values() if not df.empty)
#             print(f"📊 Loaded {loaded_count} data types for {track_name}")
            
#             return track_data
            
#         except Exception as e:
#             print(f"❌ Failed to load {track_name}: {e}")
#             return self._return_empty_data()
    
#     def _return_empty_data(self) -> Dict[str, pd.DataFrame]:
#         """Return empty data structure"""
#         return {
#             'lap_data': pd.DataFrame(),
#             'race_data': pd.DataFrame(),
#             'weather_data': pd.DataFrame(),
#             'telemetry_data': pd.DataFrame()
#         }
    
#     def load_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
#         """Load all available tracks - MAIN METHOD CALLED BY ORCHESTRATOR"""
#         available_tracks = self.list_available_tracks()
#         print(f"📁 Found {len(available_tracks)} tracks: {available_tracks}")
#         return self.load_all_tracks(available_tracks)
    
#     def load_all_tracks(self, tracks: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
#         """Load multiple tracks for combined training"""
#         all_data = {}
#         for track in tracks:
#             try:
#                 print(f"\n🏁 Loading {track}...")
#                 all_data[track] = self.load_track_data(track)
#             except Exception as e:
#                 print(f"⚠️ Failed to load {track}: {e}")
#                 all_data[track] = self._return_empty_data()
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
#                     print(f"📦 Found: {track_name}")
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
                
#                 print(f"📂 Found {len(all_files)} files in zip")
                
#                 # Find CSV files using pattern matching
#                 found_files = self._find_files_by_patterns(all_files, [])
#                 print(f"🔍 Pattern matching found: {found_files}")
                
#                 # Load each found file
#                 for data_type, file_path in found_files.items():
#                     try:
#                         with z.open(file_path) as f:
#                             file_content = f.read()
                            
#                             # Try different separators and encodings
#                             for sep in [';', ',', '\t']:
#                                 for encoding in ['utf-8', 'latin-1', 'cp1252']:
#                                     try:
#                                         # Reset to beginning of file content for each attempt
#                                         df = pd.read_csv(io.BytesIO(file_content), sep=sep, low_memory=False, encoding=encoding)
#                                         if len(df.columns) > 1 and len(df) > 0:  # Valid CSV with data
#                                             data_frames[data_type] = df
#                                             print(f"✅ Loaded {data_type} from {file_path} (sep: {sep}, encoding: {encoding})")
#                                             break
#                                     except UnicodeDecodeError:
#                                         continue
#                                     except Exception as e:
#                                         continue
#                                 if data_type in data_frames:
#                                     break
                                    
#                             if data_type not in data_frames:
#                                 print(f"❌ Could not load {data_type} from {file_path} with any separator/encoding")
                                
#                     except Exception as e:
#                         print(f"⚠️ Failed to load {data_type} from {file_path}: {e}")
                
#         except Exception as e:
#             print(f"❌ Failed to process zip file: {e}")
        
#         return data_frames
    
#     def load_track_data(self, track_name: str) -> Dict[str, pd.DataFrame]:
#         """Load all data files for a specific track from Firebase Storage"""
#         try:
#             blob_path = f"datasets/{track_name}.zip"
#             print(f"📥 Attempting to download: {blob_path}")
            
#             blob = self.bucket.blob(blob_path)
            
#             # Check if blob exists
#             if not blob.exists():
#                 print(f"❌ Blob does not exist: {blob_path}")
#                 return self._return_empty_data()
                
#             zip_data = blob.download_as_bytes()
#             print(f"✅ Downloaded {track_name}.zip ({len(zip_data)} bytes) from Firebase Storage")
            
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
#             return self._return_empty_data()
    
#     def _return_empty_data(self) -> Dict[str, pd.DataFrame]:
#         """Return empty data structure"""
#         return {
#             'lap_data': pd.DataFrame(),
#             'race_data': pd.DataFrame(),
#             'weather_data': pd.DataFrame(),
#             'telemetry_data': pd.DataFrame()
#         }
    
#     def load_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
#         """Load all available tracks - MAIN METHOD CALLED BY ORCHESTRATOR"""
#         available_tracks = self.list_available_tracks()
#         print(f"📁 Found {len(available_tracks)} tracks: {available_tracks}")
#         return self.load_all_tracks(available_tracks)
    
#     def load_all_tracks(self, tracks: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
#         """Load multiple tracks for combined training"""
#         all_data = {}
#         for track in tracks:
#             try:
#                 print(f"\n🏁 Loading {track}...")
#                 all_data[track] = self.load_track_data(track)
#             except Exception as e:
#                 print(f"⚠️ Failed to load {track}: {e}")
#                 all_data[track] = self._return_empty_data()
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
#                     print(f"📦 Found: {track_name}")
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