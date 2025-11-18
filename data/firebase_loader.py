import firebase_admin
from firebase_admin import credentials, storage
import pandas as pd
import zipfile
import io
import os, json, base64
import joblib
from typing import List, Dict, Optional
import tempfile
import time


class FirebaseDataLoader:
    def __init__(self, bucket_name: str, enable_firebase: bool = True):
        self.enable_firebase = enable_firebase
        self.bucket = None
        self.initialized = False
        
        if enable_firebase:
            self._initialize_firebase(bucket_name)
    
    def _initialize_firebase(self, bucket_name: str):
        """Initialize Firebase with robust error handling"""
        try:
            if firebase_admin._apps:
                print("✅ Firebase already initialized")
                self.bucket = storage.bucket()
                self.initialized = True
                return
            
            cred_base64 = os.getenv("FIREBASE_CREDENTIALS_BASE64")
            if not cred_base64:
                print("❌ Firebase credentials not found in environment variables")
                self.enable_firebase = False
                return
            
            # Decode and validate credentials
            try:
                cred_json = base64.b64decode(cred_base64)
                cred_dict = json.loads(cred_json)
                
                # Validate required fields
                required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
                if not all(field in cred_dict for field in required_fields):
                    print("❌ Invalid Firebase credentials structure")
                    self.enable_firebase = False
                    return
                    
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})
                self.bucket = storage.bucket()
                self.initialized = True
                print("✅ Firebase initialized successfully")
                
            except (json.JSONDecodeError, base64.binascii.Error) as e:
                print(f"❌ Failed to decode Firebase credentials: {e}")
                self.enable_firebase = False
                
        except Exception as e:
            print(f"❌ Firebase initialization failed: {e}")
            self.enable_firebase = False
    
    def _get_smart_file_patterns(self) -> Dict[str, List[List[str]]]:
        """Define intelligent file patterns for different data types"""
        return {
            'lap_data': [
                ['lap_time', 'laptime', 'timing'],
                ['analysis', 'endurance', 'sections'],
                ['lap', 'laps']
            ],
            'race_data': [
                ['results', 'classification', 'standings'],
                ['provisional', 'official', 'final'],
                ['race', 'session']
            ],
            'weather_data': [
                ['weather', 'environment', 'conditions'],
                ['temperature', 'temp', 'humidity']
            ],
            'telemetry_data': [
                ['telemetry', 'tele', 'sensor'],
                ['vbox', 'data', 'channel'],
                ['accel', 'brake', 'throttle', 'steering']
            ]
        }
    
    def _classify_file_by_content(self, zip_file: zipfile.ZipFile, file_path: str) -> Optional[str]:
        """Classify file by analyzing its content when filename patterns are ambiguous"""
        try:
            with zip_file.open(file_path) as f:
                # Read first few KB to analyze content
                sample_content = f.read(4096).decode('utf-8', errors='ignore')
                
                # Check for column patterns
                if any(col in sample_content.upper() for col in ['LAP_TIME', 'LAP_NUMBER', 'S1_SECONDS']):
                    return 'lap_data'
                elif any(col in sample_content.upper() for col in ['POSITION', 'DRIVER', 'CLASSIFICATION']):
                    return 'race_data'
                elif any(col in sample_content.upper() for col in ['TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WIND']):
                    return 'weather_data'
                elif any(col in sample_content.upper() for col in ['THROTTLE', 'BRAKE', 'ACCEL', 'RPM', 'SPEED']):
                    return 'telemetry_data'
                    
        except Exception:
            pass
            
        return None
    
    def _find_files_intelligently(self, file_list: List[str], zip_file: zipfile.ZipFile) -> Dict[str, List[str]]:
        """Find files using both filename patterns and content analysis"""
        found_files = {data_type: [] for data_type in self._get_smart_file_patterns().keys()}
        classified_files = set()
        
        patterns = self._get_smart_file_patterns()
        
        for file_path in file_list:
            if not file_path.lower().endswith(('.csv', '.txt', '.data')):
                continue
                
            filename = os.path.basename(file_path).lower()
            file_classified = False
            
            # Try filename pattern matching first
            for data_type, pattern_groups in patterns.items():
                for pattern_group in pattern_groups:
                    if any(pattern in filename for pattern in pattern_group):
                        if file_path not in found_files[data_type]:
                            found_files[data_type].append(file_path)
                            classified_files.add(file_path)
                            file_classified = True
                            break
                if file_classified:
                    break
            
            # If not classified by filename, try content analysis
            if not file_classified and file_path not in classified_files:
                content_type = self._classify_file_by_content(zip_file, file_path)
                if content_type and file_path not in found_files[content_type]:
                    found_files[content_type].append(file_path)
                    classified_files.add(file_path)
        
        return found_files
    
    def _robust_csv_loading(self, zip_file: zipfile.ZipFile, file_path: str, max_rows: int = 10000) -> pd.DataFrame:
        """Load CSV files with comprehensive error recovery"""
        attempts = [
            # Try standard CSV first
            {'sep': ',', 'encoding': 'utf-8', 'engine': 'python'},
            {'sep': ',', 'encoding': 'latin-1', 'engine': 'python'},
            {'sep': ';', 'encoding': 'utf-8', 'engine': 'python'},
            {'sep': ';', 'encoding': 'latin-1', 'engine': 'python'},
            {'sep': '\t', 'encoding': 'utf-8', 'engine': 'python'},
            # Try with different parameters for problematic files
            {'sep': None, 'encoding': 'utf-8', 'engine': 'python', 'delim_whitespace': True},
            {'sep': None, 'encoding': 'latin-1', 'engine': 'python', 'delim_whitespace': True},
        ]
        
        best_df = pd.DataFrame()
        best_row_count = 0
        
        for i, params in enumerate(attempts):
            try:
                with zip_file.open(file_path) as f:
                    # Read file content once
                    file_content = f.read()
                    
                    # Try reading with current parameters
                    current_params = params.copy()
                    
                    # For very large files, read only first N rows to test
                    if len(file_content) > 1000000:  # 1MB
                        current_params['nrows'] = 1000
                    
                    df = pd.read_csv(io.BytesIO(file_content), **current_params)
                    
                    # Validate this is a reasonable dataframe
                    if len(df.columns) >= 2 and len(df) > 0:
                        # If we limited rows and it worked, try full file
                        if 'nrows' in current_params and len(df) == current_params['nrows']:
                            try:
                                full_params = params.copy()
                                df_full = pd.read_csv(io.BytesIO(file_content), **full_params)
                                if len(df_full) > len(df):
                                    df = df_full
                            except:
                                pass  # Keep the limited version
                        
                        # Apply row limit to prevent memory issues
                        if len(df) > max_rows:
                            df = df.head(max_rows)
                            print(f"⚠️ Limited {file_path} to {max_rows} rows")
                        
                        # Clean column names
                        df.columns = [str(col).strip().upper() for col in df.columns]
                        
                        print(f"✅ Loaded {file_path} with params {i+1}: {len(df)} rows, {len(df.columns)} cols")
                        return df
                        
            except Exception as e:
                continue
        
        # If all attempts fail, try manual parsing as last resort
        try:
            with zip_file.open(file_path) as f:
                content = f.read().decode('latin-1', errors='ignore')
                lines = content.split('\n')
                
                # Find the first data row
                data_start = 0
                for i, line in enumerate(lines):
                    if len(line.split(',')) >= 2 or len(line.split(';')) >= 2:
                        data_start = i
                        break
                
                if data_start > 0:
                    # Reconstruct with header
                    header = lines[data_start] if data_start < len(lines) else ''
                    data_lines = lines[data_start + 1:data_start + 101]  # First 100 rows
                    
                    if data_lines:
                        reconstructed = [header] + data_lines
                        df = pd.read_csv(io.StringIO('\n'.join(reconstructed)), engine='python')
                        print(f"🔄 Manually parsed {file_path}: {len(df)} rows")
                        return df
                        
        except Exception:
            pass
        
        print(f"❌ All loading attempts failed for {file_path}")
        return pd.DataFrame()
    
    def _validate_and_clean_dataframe(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Validate and clean dataframe based on data type"""
        if df.empty:
            return df
            
        df_clean = df.copy()
        
        try:
            # Remove completely empty rows and columns
            df_clean = df_clean.dropna(how='all').reset_index(drop=True)
            df_clean = df_clean.loc[:, ~df_clean.isna().all()]
            
            # Basic data type-specific validation
            if data_type == 'lap_data':
                # Ensure we have some lap-related columns
                lap_columns = [col for col in df_clean.columns if 'LAP' in col or 'TIME' in col]
                if not lap_columns and len(df_clean) > 0:
                    print(f"⚠️ Lap data missing lap/time columns")
                    
            elif data_type == 'telemetry_data':
                # Check for sensor data columns
                sensor_columns = [col for col in df_clean.columns if any(x in col for x in ['SPEED', 'RPM', 'THROTTLE', 'BRAKE'])]
                if not sensor_columns and len(df_clean) > 0:
                    print(f"⚠️ Telemetry data missing sensor columns")
            
            # Convert numeric columns where possible
            for col in df_clean.columns:
                try:
                    # Skip if already numeric
                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                        continue
                    
                    # Try conversion to numeric
                    converted = pd.to_numeric(df_clean[col], errors='coerce')
                    if converted.notna().sum() > len(converted) * 0.5:  # Mostly numeric
                        df_clean[col] = converted
                except:
                    pass
                    
        except Exception as e:
            print(f"⚠️ Data cleaning failed for {data_type}: {e}")
            
        return df_clean
    
    def _extract_files_with_validation(self, zip_data: bytes) -> Dict[str, pd.DataFrame]:
        """Extract and load files with comprehensive validation"""
        data_frames = {
            'lap_data': pd.DataFrame(),
            'race_data': pd.DataFrame(),
            'weather_data': pd.DataFrame(),
            'telemetry_data': pd.DataFrame()
        }
        
        try:
            with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_file:
                # Get all files
                all_files = [info.filename for info in zip_file.infolist() if not info.is_dir()]
                print(f"📂 Found {len(all_files)} files in zip")
                
                if not all_files:
                    print("❌ Zip file is empty")
                    return data_frames
                
                # Intelligent file classification
                found_files = self._find_files_intelligently(all_files, zip_file)
                
                # Process each data type
                for data_type, file_paths in found_files.items():
                    if not file_paths:
                        print(f"📝 No files classified as {data_type}")
                        continue
                    
                    print(f"🔍 Found {len(file_paths)} files for {data_type}")
                    
                    dfs = []
                    for file_path in file_paths:
                        try:
                            df = self._robust_csv_loading(zip_file, file_path)
                            if not df.empty:
                                df_clean = self._validate_and_clean_dataframe(df, data_type)
                                if not df_clean.empty:
                                    dfs.append(df_clean)
                                    print(f"  ✅ {os.path.basename(file_path)}: {len(df_clean)} rows")
                                else:
                                    print(f"  ⚠️ {os.path.basename(file_path)}: empty after cleaning")
                            else:
                                print(f"  ❌ {os.path.basename(file_path)}: failed to load")
                        except Exception as e:
                            print(f"  ❌ {os.path.basename(file_path)}: {e}")
                            continue
                    
                    # Combine dataframes for this type
                    if dfs:
                        try:
                            combined_df = pd.concat(dfs, ignore_index=True, sort=False)
                            data_frames[data_type] = combined_df
                            print(f"📊 Combined {len(dfs)} files into {data_type}: {len(combined_df)} rows")
                        except Exception as e:
                            print(f"❌ Failed to combine {data_type} data: {e}")
                            # Use the first valid dataframe if combination fails
                            data_frames[data_type] = dfs[0]
                    else:
                        print(f"❌ No valid data for {data_type}")
                        
        except zipfile.BadZipFile:
            print("❌ Invalid zip file format")
        except Exception as e:
            print(f"❌ Failed to process zip file: {e}")
        
        return data_frames
    
    def _check_track_data_cached(self, track_name: str) -> bool:
        """Check if track data is cached with validation"""
        cache_file = f'/kaggle/working/{track_name}_cached.pkl'
        if not os.path.exists(cache_file):
            return False
        
        try:
            # Validate cache file is not corrupted
            with open(cache_file, 'rb') as f:
                data = joblib.load(f)
                if isinstance(data, dict) and all(isinstance(df, pd.DataFrame) for df in data.values()):
                    return True
                else:
                    print(f"⚠️ Invalid cache format for {track_name}")
                    os.remove(cache_file)  # Remove corrupted cache
                    return False
        except Exception:
            print(f"⚠️ Corrupted cache file for {track_name}")
            try:
                os.remove(cache_file)
            except:
                pass
            return False
    
    def _check_track_exists_firebase(self, track_name: str) -> bool:
        """Check if track exists in Firebase with timeout"""
        if not self.initialized:
            return False
            
        try:
            blob_path = f"datasets/{track_name}.zip"
            blob = self.bucket.blob(blob_path)
            
            # Add timeout for the existence check
            import threading
            result = [None]
            
            def check_existence():
                try:
                    result[0] = blob.exists()
                except Exception:
                    result[0] = False
            
            thread = threading.Thread(target=check_existence)
            thread.daemon = True
            thread.start()
            thread.join(timeout=10)  # 10 second timeout
            
            if thread.is_alive():
                print(f"⏰ Timeout checking Firebase for {track_name}")
                return False
                
            return result[0] if result[0] is not None else False
            
        except Exception as e:
            print(f"⚠️ Error checking Firebase for {track_name}: {e}")
            return False
    
    def _download_with_retry(self, blob, max_retries: int = 3) -> Optional[bytes]:
        """Download with retry logic and progress tracking"""
        for attempt in range(max_retries):
            try:
                print(f"📥 Download attempt {attempt + 1}/{max_retries}...")
                start_time = time.time()
                zip_data = blob.download_as_bytes(timeout=120)  # 2-minute timeout
                download_time = time.time() - start_time
                print(f"✅ Download successful ({len(zip_data)} bytes, {download_time:.1f}s)")
                return zip_data
                
            except Exception as e:
                print(f"⚠️ Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print("❌ All download attempts failed")
                    
        return None
    
    def _generate_fallback_data(self, track_name: str) -> Dict[str, pd.DataFrame]:
        """Generate fallback synthetic data when Firebase is unavailable"""
        print(f"🔄 Generating fallback data for {track_name}")
        
        # Generate realistic synthetic data for all types
        def generate_lap_data():
            return pd.DataFrame({
                'NUMBER': [1, 1, 1, 2, 2, 2],
                'LAP_NUMBER': [1, 2, 3, 1, 2, 3],
                'LAP_TIME_SECONDS': [85.2, 84.8, 85.5, 86.1, 85.9, 86.3],
                'S1_SECONDS': [28.1, 27.9, 28.2, 28.4, 28.3, 28.5],
                'S2_SECONDS': [28.5, 28.3, 28.6, 28.8, 28.7, 28.9],
                'S3_SECONDS': [28.6, 28.6, 28.7, 28.9, 28.9, 28.9]
            })
        
        def generate_race_data():
            return pd.DataFrame({
                'NUMBER': [1, 2, 3],
                'POSITION': [1, 2, 3],
                'GAP_FIRST_SECONDS': [0.0, 1.5, 3.2],
                'LAPS': [50, 50, 50]
            })
        
        def generate_weather_data():
            return pd.DataFrame({
                'TIMESTAMP': pd.date_range('2025-01-01', periods=3, freq='30min'),
                'AIR_TEMP': [25.0, 26.0, 27.0],
                'TRACK_TEMP': [30.0, 32.0, 34.0],
                'HUMIDITY': [60.0, 58.0, 55.0]
            })
        
        def generate_telemetry_data():
            return pd.DataFrame({
                'vehicle_number': [1, 1, 1, 2, 2, 2],
                'lap': [1, 1, 1, 2, 2, 2],
                'THROTTLE_POSITION': [80, 85, 75, 78, 82, 79],
                'BRAKE_PRESSURE': [25, 30, 20, 28, 32, 26],
                'KPH': [185, 190, 180, 182, 188, 183]
            })
        
        return {
            'lap_data': generate_lap_data(),
            'race_data': generate_race_data(),
            'weather_data': generate_weather_data(),
            'telemetry_data': generate_telemetry_data()
        }
    
    def load_track_data(self, track_name: str, use_fallback: bool = True) -> Dict[str, pd.DataFrame]:
        """Load track data with comprehensive error handling and fallbacks"""
        try:
            # Try cached data first
            if self._check_track_data_cached(track_name):
                cache_file = f'/kaggle/working/{track_name}_cached.pkl'
                print(f"📂 Loading cached data for {track_name}")
                with open(cache_file, 'rb') as f:
                    cached_data = joblib.load(f)
                    # Validate cached data has some content
                    if any(not df.empty for df in cached_data.values()):
                        return cached_data
                    else:
                        print(f"⚠️ Cached data for {track_name} is empty")
            
            # Try Firebase download if enabled
            if self.enable_firebase and self.initialized:
                if self._check_track_exists_firebase(track_name):
                    blob_path = f"datasets/{track_name}.zip"
                    print(f"📥 Downloading from Firebase: {blob_path}")
                    
                    blob = self.bucket.blob(blob_path)
                    zip_data = self._download_with_retry(blob)
                    
                    if zip_data:
                        track_data = self._extract_files_with_validation(zip_data)
                        
                        # Cache if we got any data
                        if any(not df.empty for df in track_data.values()):
                            cache_file = f'/kaggle/working/{track_name}_cached.pkl'
                            try:
                                with open(cache_file, 'wb') as f:
                                    joblib.dump(track_data, f)
                                print(f"💾 Cached {track_name} data")
                            except Exception as e:
                                print(f"⚠️ Could not cache {track_name}: {e}")
                            
                            return track_data
            
            # Use fallback data if enabled
            if use_fallback:
                print(f"🔄 Using fallback data for {track_name}")
                fallback_data = self._generate_fallback_data(track_name)
                
                # Cache fallback data
                cache_file = f'/kaggle/working/{track_name}_cached.pkl'
                try:
                    with open(cache_file, 'wb') as f:
                        joblib.dump(fallback_data, f)
                except Exception:
                    pass
                    
                return fallback_data
            else:
                return self._return_empty_data()
                
        except Exception as e:
            print(f"❌ Failed to load {track_name}: {e}")
            if use_fallback:
                return self._generate_fallback_data(track_name)
            else:
                return self._return_empty_data()
    
    def load_all_tracks_optimized(self, tracks: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load multiple tracks with progress tracking and error isolation"""
        all_data = {}
        
        print(f"🏁 Loading {len(tracks)} tracks...")
        
        for i, track in enumerate(tracks, 1):
            print(f"\n[{i}/{len(tracks)}] Processing: {track}")
            try:
                track_data = self.load_track_data(track, use_fallback=True)
                all_data[track] = track_data
                
                # Report what was loaded
                loaded_types = [t for t, df in track_data.items() if not df.empty]
                print(f"   ✅ Loaded {len(loaded_types)} data types: {loaded_types}")
                
            except Exception as e:
                print(f"   ❌ Failed to load {track}: {e}")
                all_data[track] = self._generate_fallback_data(track)
        
        return all_data
    
    def _return_empty_data(self) -> Dict[str, pd.DataFrame]:
        """Return empty data structure"""
        return {data_type: pd.DataFrame() for data_type in self._get_smart_file_patterns().keys()}
    
    # Backward compatibility methods
    def load_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        available_tracks = self.list_available_tracks()
        print(f"📁 Found {len(available_tracks)} tracks")
        return self.load_all_tracks_optimized(available_tracks)
    
    def load_all_tracks(self, tracks: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        return self.load_all_tracks_optimized(tracks)
    
    def list_available_tracks(self) -> List[str]:
        """List available tracks with fallback to default tracks"""
        tracks = set()
        
        if self.enable_firebase and self.initialized:
            try:
                blobs = self.bucket.list_blobs(prefix="datasets/")
                for blob in blobs:
                    if blob.name.endswith('.zip'):
                        track_name = os.path.basename(blob.name).replace('.zip', '')
                        tracks.add(track_name)
            except Exception as e:
                print(f"⚠️ Failed to list Firebase tracks: {e}")
        
        # Add default tracks if Firebase failed or no tracks found
        if not tracks:
            default_tracks = [
                'sonoma', 'indianapolis', 'road-america', 'circuit-of-the-americas',
                'sebring', 'virginia-international-raceway', 'barber-motorsports-park'
            ]
            tracks.update(default_tracks)
            print("📝 Using default track list")
        
        return sorted(list(tracks))
    
    def upload_models_to_firebase(self, models_dir: str = "outputs/models") -> bool:
        """Upload models with enhanced error handling"""
        if not self.enable_firebase:
            print("❌ Firebase not enabled, skipping upload")
            return False
            
        try:
            if not os.path.exists(models_dir):
                print(f"❌ Models directory not found: {models_dir}")
                return False
                
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            if not model_files:
                print("❌ No model files found")
                return False
            
            success_count = 0
            for model_file in model_files:
                try:
                    local_path = os.path.join(models_dir, model_file)
                    blob_path = f"trained_models/{model_file}"
                    
                    blob = self.bucket.blob(blob_path)
                    blob.upload_from_filename(local_path, timeout=60)
                    success_count += 1
                    print(f"✅ Uploaded {model_file}")
                    
                except Exception as e:
                    print(f"❌ Failed to upload {model_file}: {e}")
            
            print(f"📊 Uploaded {success_count}/{len(model_files)} models")
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Model upload failed: {e}")
            return False
    
    def download_models_from_firebase(self, local_dir: str = "outputs/models") -> bool:
        """Download models with enhanced error handling"""
        if not self.enable_firebase:
            print("❌ Firebase not enabled, skipping download")
            return False
            
        try:
            os.makedirs(local_dir, exist_ok=True)
            blobs = list(self.bucket.list_blobs(prefix="trained_models/"))
            
            if not blobs:
                print("📝 No models found in Firebase")
                return False
            
            success_count = 0
            for blob in blobs:
                if blob.name.endswith('.pkl'):
                    try:
                        local_path = os.path.join(local_dir, os.path.basename(blob.name))
                        blob.download_to_filename(local_path, timeout=60)
                        success_count += 1
                        print(f"✅ Downloaded {os.path.basename(blob.name)}")
                    except Exception as e:
                        print(f"❌ Failed to download {blob.name}: {e}")
            
            print(f"📊 Downloaded {success_count}/{len(blobs)} models")
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Model download failed: {e}")
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
    
#     def _check_track_data_cached(self, track_name: str) -> bool:
#         """Check if track data is already cached on Kaggle"""
#         cache_file = f'/kaggle/working/{track_name}_cached.pkl'
#         return os.path.exists(cache_file)
    
#     def _check_track_exists_firebase(self, track_name: str) -> bool:
#         """Check if track exists in Firebase Storage"""
#         blob_path = f"datasets/{track_name}.zip"
#         blob = self.bucket.blob(blob_path)
#         return blob.exists()
    
#     def load_track_data(self, track_name: str) -> Dict[str, pd.DataFrame]:
#         """Load track data - only download if not already cached"""
#         try:
#             # Check for cached data first
#             if self._check_track_data_cached(track_name):
#                 cache_file = f'/kaggle/working/{track_name}_cached.pkl'
#                 print(f"📂 Loading cached data for {track_name}")
#                 with open(cache_file, 'rb') as f:
#                     return joblib.load(f)
            
#             # Check if track exists in Firebase before downloading
#             if not self._check_track_exists_firebase(track_name):
#                 print(f"❌ Track {track_name} not found in Firebase Storage")
#                 return self._return_empty_data()
            
#             # Download only if not cached
#             blob_path = f"datasets/{track_name}.zip"
#             print(f"📥 Downloading missing dataset: {blob_path}")
            
#             blob = self.bucket.blob(blob_path)
            
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
#                 cache_file = f'/kaggle/working/{track_name}_cached.pkl'
#                 with open(cache_file, 'wb') as f:
#                     joblib.dump(track_data, f)
#                 print(f"💾 Cached {track_name} data for future sessions")
#             except Exception as e:
#                 print(f"⚠️ Could not cache {track_name}: {e}")
            
#             # Report what was loaded
#             loaded_count = sum(1 for df in track_data.values() if not df.empty)
#             print(f"📊 Loaded {loaded_count} data types for {track_name}")
            
#             return track_data
            
#         except Exception as e:
#             print(f"❌ Failed to load {track_name}: {e}")
#             return self._return_empty_data()
    
#     def load_all_tracks_optimized(self, tracks: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
#         """Load multiple tracks - only download missing datasets"""
#         all_data = {}
#         total_tracks = len(tracks)
        
#         # Check which tracks are already cached
#         cached_tracks = [t for t in tracks if self._check_track_data_cached(t)]
#         missing_tracks = [t for t in tracks if not self._check_track_data_cached(t)]
        
#         print(f"📊 Track Status: {len(cached_tracks)} cached, {len(missing_tracks)} need download")
        
#         if cached_tracks:
#             print(f"📂 Loading cached tracks: {cached_tracks}")
#             for track in cached_tracks:
#                 all_data[track] = self.load_track_data(track)
        
#         if missing_tracks:
#             print(f"📥 Downloading missing tracks: {missing_tracks}")
#             for i, track in enumerate(missing_tracks, 1):
#                 try:
#                     print(f"\n🏁 Downloading track {i}/{len(missing_tracks)}: {track}")
#                     all_data[track] = self.load_track_data(track)
#                 except Exception as e:
#                     print(f"⚠️ Failed to load {track}: {e}")
#                     all_data[track] = self._return_empty_data()
        
#         return all_data
    
#     def _return_empty_data(self) -> Dict[str, pd.DataFrame]:
#         """Return empty data structure"""
#         return {
#             'lap_data': pd.DataFrame(),
#             'race_data': pd.DataFrame(),
#             'weather_data': pd.DataFrame(),
#             'telemetry_data': pd.DataFrame()
#         }
    
#     def load_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
#         """Load all available tracks - MAIN METHOD CALLED BY ORCHESTRATOR (backward compatible)"""
#         available_tracks = self.list_available_tracks()
#         print(f"📁 Found {len(available_tracks)} tracks: {available_tracks}")
        
#         # Use optimized loading that only downloads missing datasets
#         return self.load_all_tracks_optimized(available_tracks)
    
#     def load_all_tracks(self, tracks: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
#         """Load multiple tracks for combined training with progress tracking (backward compatible)"""
#         # This maintains backward compatibility but uses optimized loading internally
#         return self.load_all_tracks_optimized(tracks)
    
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