import re
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

class DataPreprocessor:
    """
    Enhanced Data Preprocessor with robust error handling and logging.
    Maintains compatibility with FirebaseDataLoader schemas while adding resilience.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # Minimum required columns for each data type
        self.required_columns = {
            'pit_data': ['NUMBER', 'LAP_NUMBER', 'LAP_TIME'],
            'race_data': ['NUMBER', 'POSITION', 'LAPS'],
            'weather_data': ['AIR_TEMP', 'TRACK_TEMP'],
            'telemetry_data': ['vehicle_id', 'lap', 'speed']
        }

    def _log(self, level: str, message: str):
        """Unified logging method"""
        if self.debug:
            print(f"{level} {message}")
        if level == "❌":
            self.logger.error(message)
        elif level == "⚠️":
            self.logger.warning(message)
        else:
            self.logger.info(message)

    # --------------------
    # Enhanced Public Preprocessors
    # --------------------
    
    def preprocess_pit_data(self, pit_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Enhanced pit data preprocessing with robust error handling"""
        if pit_df is None or pit_df.empty:
            self._log("⚠️", "Pit data is None or empty, returning empty DataFrame")
            return pd.DataFrame()

        try:
            df = pit_df.copy()
            df = self._normalize_dataframe(df)
            
            # Log original state for debugging
            self._log("🔧", f"Original pit data: {len(df)} rows, {len(df.columns)} columns")
            self._log("🔧", f"Pit data columns: {list(df.columns)}")

            # Enhanced time column conversion with validation
            time_columns = ['LAP_TIME', 'S1', 'S2', 'S3', 'PIT_TIME', 'FL_TIME']
            converted_time_cols = []
            
            for col in time_columns:
                if col in df.columns:
                    new_col = f"{col}_SECONDS"
                    df[new_col] = df[col].apply(self.time_to_seconds)
                    converted_time_cols.append(new_col)
                    # Validate conversion
                    valid_conversions = df[new_col].notna().sum()
                    self._log("🔧", f"Converted {col} -> {new_col}: {valid_conversions}/{len(df)} valid")

            # Convert intermediate timing columns
            intermediate_times = ['IM1a_time', 'IM1_time', 'IM2a_time', 'IM2_time', 'IM3a_time', 'FL_time']
            for col in intermediate_times:
                if col in df.columns:
                    df[f"{col}_SECONDS"] = df[col].apply(self.time_to_seconds)

            # Convert elapsed columns
            elapsed_columns = ['IM1a_elapsed', 'IM1_elapsed', 'IM2a_elapsed', 'IM2_elapsed', 'IM3a_elapsed', 'FL_elapsed']
            for col in elapsed_columns:
                if col in df.columns:
                    df[f"{col}_SECONDS"] = df[col].apply(self.time_to_seconds)

            # Enhanced numeric column processing with validation
            numeric_columns = [
                'NUMBER', 'DRIVER_NUMBER', 'LAP_NUMBER', 'LAP_IMPROVEMENT', 
                'S1_IMPROVEMENT', 'S2_IMPROVEMENT', 'S3_IMPROVEMENT',
                'KPH', 'TOP_SPEED', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'
            ]
            
            numeric_conversions = {}
            for col in numeric_columns:
                if col in df.columns:
                    original_non_null = df[col].notna().sum()
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    final_non_null = (df[col] != 0).sum()
                    numeric_conversions[col] = (original_non_null, final_non_null)

            if self.debug:
                for col, (orig, final) in numeric_conversions.items():
                    self._log("🔧", f"Numeric {col}: {final}/{orig} valid values")

            # Handle large sector times
            large_sector_cols = ['S1_LARGE', 'S2_LARGE', 'S3_LARGE']
            for col in large_sector_cols:
                if col in df.columns:
                    df[f"{col}_SECONDS"] = df[col].apply(self.time_to_seconds)

            # Enhanced performance metrics with validation
            if 'LAP_TIME_SECONDS' in df.columns and 'NUMBER' in df.columns:
                try:
                    df['PERFORMANCE_DROP'] = df.groupby('NUMBER')['LAP_TIME_SECONDS'].transform(
                        lambda x: x - x.min() if x.notna().any() else 0
                    )
                    df['CONSISTENCY'] = df.groupby('NUMBER')['LAP_TIME_SECONDS'].transform(
                        lambda x: x.std() if x.notna().any() and len(x) > 1 else 0
                    ).fillna(0)
                    
                    # Validate performance metrics
                    valid_performance = df['PERFORMANCE_DROP'].notna().sum()
                    valid_consistency = df['CONSISTENCY'].notna().sum()
                    self._log("🔧", f"Performance metrics: {valid_performance} performance drops, {valid_consistency} consistency values")
                    
                except Exception as e:
                    self._log("⚠️", f"Performance metric calculation failed: {e}")

            # Data quality report
            total_cells = len(df) * len(df.columns)
            null_cells = df.isnull().sum().sum()
            null_percentage = (null_cells / total_cells * 100) if total_cells > 0 else 0
            
            self._log("✅", f"Processed pit data: {len(df)} rows, {len(df.columns)} columns, {null_percentage:.1f}% nulls")
            
            return df

        except Exception as e:
            self._log("❌", f"Pit data preprocessing failed: {e}")
            return pd.DataFrame()

    def preprocess_race_data(self, race_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Enhanced race data preprocessing with robust error handling"""
        if race_df is None or race_df.empty:
            self._log("⚠️", "Race data is None or empty, returning empty DataFrame")
            return pd.DataFrame()

        try:
            df = race_df.copy()
            df = self._normalize_dataframe(df)
            
            self._log("🔧", f"Original race data: {len(df)} rows, {len(df.columns)} columns")

            # Enhanced position handling
            if 'POSITION' in df.columns:
                original_positions = df['POSITION'].nunique()
                df['POSITION'] = pd.to_numeric(df['POSITION'], errors='coerce').fillna(999)  # Use high number for invalid
                valid_positions = (df['POSITION'] != 999).sum()
                self._log("🔧", f"Position processing: {valid_positions}/{len(df)} valid positions")

            # Enhanced time conversion with validation
            time_conversions = {}
            if 'TOTAL_TIME' in df.columns:
                df['TOTAL_TIME_SECONDS'] = df['TOTAL_TIME'].apply(self.time_to_seconds)
                time_conversions['TOTAL_TIME'] = df['TOTAL_TIME_SECONDS'].notna().sum()

            if 'FL_TIME' in df.columns:
                df['FL_TIME_SECONDS'] = df['FL_TIME'].apply(self.time_to_seconds)
                time_conversions['FL_TIME'] = df['FL_TIME_SECONDS'].notna().sum()

            if self.debug and time_conversions:
                for col, valid_count in time_conversions.items():
                    self._log("🔧", f"Time conversion {col}: {valid_count}/{len(df)} valid")

            # Enhanced gap parsing
            gap_columns = ['GAP_FIRST', 'GAP_PREVIOUS']
            gap_conversions = {}
            for col in gap_columns:
                if col in df.columns:
                    df[f"{col}_SECONDS"] = df[col].apply(self.gap_to_seconds)
                    gap_conversions[col] = df[f"{col}_SECONDS"].notna().sum()

            if self.debug and gap_conversions:
                for col, valid_count in gap_conversions.items():
                    self._log("🔧", f"Gap conversion {col}: {valid_count}/{len(df)} valid")

            # Enhanced numeric processing
            numeric_columns = ['NUMBER', 'LAPS', 'FL_LAPNUM', 'FL_KPH']
            numeric_stats = {}
            for col in numeric_columns:
                if col in df.columns:
                    original_non_null = df[col].notna().sum()
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    final_non_null = (df[col] != 0).sum()
                    numeric_stats[col] = (original_non_null, final_non_null)

            if self.debug:
                for col, (orig, final) in numeric_stats.items():
                    self._log("🔧", f"Numeric {col}: {final}/{orig} valid values")

            # Enhanced status handling
            if 'STATUS' in df.columns:
                unique_statuses = df['STATUS'].nunique()
                df['STATUS'] = df['STATUS'].astype(str).fillna('Unknown')
                self._log("🔧", f"Status processing: {unique_statuses} unique status values")

            self._log("✅", f"Processed race data: {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            self._log("❌", f"Race data preprocessing failed: {e}")
            return pd.DataFrame()

    def preprocess_weather_data(self, weather_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Enhanced weather data preprocessing with robust timestamp handling"""
        if weather_df is None or weather_df.empty:
            self._log("⚠️", "Weather data is None or empty, returning empty DataFrame")
            return pd.DataFrame()

        try:
            df = weather_df.copy()
            df = self._normalize_dataframe(df)
            
            self._log("🔧", f"Original weather data: {len(df)} rows, {len(df.columns)} columns")

            # Enhanced timestamp creation with multiple fallbacks
            timestamp_created = False
            if 'TIME_UTC_SECONDS' in df.columns:
                df['timestamp'] = pd.to_datetime(df['TIME_UTC_SECONDS'], unit='s', errors='coerce')
                valid_timestamps = df['timestamp'].notna().sum()
                self._log("🔧", f"TIME_UTC_SECONDS conversion: {valid_timestamps}/{len(df)} valid")
                timestamp_created = valid_timestamps > 0

            if not timestamp_created and 'TIME_UTC_STR' in df.columns:
                # Try multiple datetime formats
                formats = [
                    '%m/%d/%Y %I:%M:%S %p',
                    '%Y-%m-%d %H:%M:%S',
                    '%m/%d/%Y %H:%M:%S'
                ]
                
                for fmt in formats:
                    df['timestamp'] = pd.to_datetime(df['TIME_UTC_STR'], format=fmt, errors='coerce')
                    if df['timestamp'].notna().sum() > 0:
                        valid_timestamps = df['timestamp'].notna().sum()
                        self._log("🔧", f"TIME_UTC_STR conversion (format {fmt}): {valid_timestamps}/{len(df)} valid")
                        timestamp_created = True
                        break

            # Fallback: synthetic timestamps if no valid timestamps created
            if not timestamp_created:
                start_time = datetime.now() - timedelta(hours=2)
                df['timestamp'] = [start_time + timedelta(seconds=i*60) for i in range(len(df))]
                self._log("⚠️", "Using synthetic timestamps for weather data")

            # Enhanced weather metric processing
            weather_metrics = ['AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 'PRESSURE', 'WIND_SPEED', 'WIND_DIRECTION', 'RAIN']
            metric_stats = {}
            
            for col in weather_metrics:
                if col in df.columns:
                    original_non_null = df[col].notna().sum()
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Use reasonable defaults for critical metrics
                    if col in ['AIR_TEMP', 'TRACK_TEMP'] and df[col].isna().any():
                        default_temp = 25.0 if col == 'AIR_TEMP' else 30.0
                        df[col] = df[col].fillna(default_temp)
                        self._log("🔧", f"Applied default {default_temp} for missing {col}")
                    
                    final_non_null = df[col].notna().sum()
                    metric_stats[col] = (original_non_null, final_non_null)

            if self.debug:
                for col, (orig, final) in metric_stats.items():
                    self._log("🔧", f"Weather metric {col}: {final}/{orig} valid values")

            # Remove rows with invalid timestamps
            original_count = len(df)
            df = df.dropna(subset=['timestamp'])
            removed_count = original_count - len(df)
            
            if removed_count > 0:
                self._log("⚠️", f"Removed {removed_count} rows with invalid timestamps")

            self._log("✅", f"Processed weather data: {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            self._log("❌", f"Weather data preprocessing failed: {e}")
            return pd.DataFrame()

    def preprocess_telemetry_data(self, telemetry_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Enhanced telemetry data preprocessing with robust timestamp handling"""
        if telemetry_df is None or telemetry_df.empty:
            self._log("⚠️", "Telemetry data is None or empty, returning empty DataFrame")
            return pd.DataFrame()

        try:
            df = telemetry_df.copy()
            df = self._normalize_dataframe(df)
            
            self._log("🔧", f"Original telemetry data: {len(df)} rows, {len(df.columns)} columns")

            # Enhanced timestamp processing
            if 'timestamp' in df.columns:
                original_timestamps = df['timestamp'].notna().sum()
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                valid_timestamps = df['timestamp'].notna().sum()
                self._log("🔧", f"Timestamp processing: {valid_timestamps}/{original_timestamps} valid")
                
                # Remove invalid timestamps
                original_count = len(df)
                df = df.dropna(subset=['timestamp'])
                removed_count = original_count - len(df)
                if removed_count > 0:
                    self._log("⚠️", f"Removed {removed_count} rows with invalid timestamps")

            # Enhanced numeric column processing
            numeric_columns = ['lap', 'outing', 'accx_can', 'accy_can', 'gear', 'speed']
            numeric_stats = {}
            
            for col in numeric_columns:
                if col in df.columns:
                    original_non_null = df[col].notna().sum()
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    final_non_null = (df[col] != 0).sum()
                    numeric_stats[col] = (original_non_null, final_non_null)

            if self.debug:
                for col, (orig, final) in numeric_stats.items():
                    self._log("🔧", f"Telemetry numeric {col}: {final}/{orig} valid values")

            # Enhanced derived metrics with validation
            if all(col in df.columns for col in ['accx_can', 'accy_can']):
                try:
                    df['total_acceleration'] = np.sqrt(df['accx_can']**2 + df['accy_can']**2)
                    valid_acceleration = df['total_acceleration'].notna().sum()
                    self._log("🔧", f"Acceleration metric: {valid_acceleration}/{len(df)} valid")
                except Exception as e:
                    self._log("⚠️", f"Acceleration calculation failed: {e}")

            self._log("✅", f"Processed telemetry data: {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            self._log("❌", f"Telemetry data preprocessing failed: {e}")
            return pd.DataFrame()

    # --------------------
    # Enhanced Batch Processing
    # --------------------
    
    def preprocess_track_data(self, track_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Enhanced batch preprocessing with comprehensive validation
        """
        processed_data = {}
        processing_stats = {}
        
        self._log("🔧", f"Starting track data preprocessing with {len(track_data)} data types")

        for data_type, df in track_data.items():
            try:
                if data_type == 'pit_data':
                    processed_data[data_type] = self.preprocess_pit_data(df)
                elif data_type == 'race_data':
                    processed_data[data_type] = self.preprocess_race_data(df)
                elif data_type == 'weather_data':
                    processed_data[data_type] = self.preprocess_weather_data(df)
                elif data_type == 'telemetry_data':
                    processed_data[data_type] = self.preprocess_telemetry_data(df)
                else:
                    self._log("⚠️", f"Unknown data type: {data_type}")
                    processed_data[data_type] = df  # Pass through unchanged
                    
                # Track processing statistics
                result_df = processed_data[data_type]
                processing_stats[data_type] = {
                    'original_rows': len(df) if df is not None else 0,
                    'processed_rows': len(result_df),
                    'success': not result_df.empty
                }
                
            except Exception as e:
                self._log("❌", f"Failed to process {data_type}: {e}")
                processed_data[data_type] = pd.DataFrame()
                processing_stats[data_type] = {
                    'original_rows': len(df) if df is not None else 0,
                    'processed_rows': 0,
                    'success': False,
                    'error': str(e)
                }

        # Enhanced status reporting
        successful_types = sum(1 for stats in processing_stats.values() if stats['success'])
        total_rows_processed = sum(stats['processed_rows'] for stats in processing_stats.values())
        
        self._log("📊", f"Preprocessing complete: {successful_types}/{len(track_data)} data types successful")
        self._log("📊", f"Total rows processed: {total_rows_processed}")
        
        if self.debug:
            for data_type, stats in processing_stats.items():
                status_icon = "✅" if stats['success'] else "❌"
                self._log("📊", f"{status_icon} {data_type}: {stats['processed_rows']}/{stats['original_rows']} rows")
                if 'error' in stats:
                    self._log("📊", f"     Error: {stats['error']}")

        return processed_data

    def preprocess_all_tracks(self, all_track_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Enhanced multi-track preprocessing with comprehensive reporting
        """
        processed_tracks = {}
        track_stats = {}
        
        self._log("🚀", f"Starting multi-track preprocessing for {len(all_track_data)} tracks")

        for track_name, track_data in all_track_data.items():
            self._log("🔧", f"\nProcessing track: {track_name}")
            
            try:
                processed_tracks[track_name] = self.preprocess_track_data(track_data)
                
                # Calculate track-level statistics
                successful_data_types = sum(1 for df in processed_tracks[track_name].values() if not df.empty)
                total_rows = sum(len(df) for df in processed_tracks[track_name].values())
                
                track_stats[track_name] = {
                    'successful_data_types': successful_data_types,
                    'total_data_types': len(track_data),
                    'total_rows': total_rows,
                    'success': successful_data_types > 0
                }
                
                status_icon = "✅" if track_stats[track_name]['success'] else "❌"
                self._log("📊", f"{status_icon} {track_name}: {successful_data_types}/{len(track_data)} data types, {total_rows} total rows")
                
            except Exception as e:
                self._log("❌", f"Failed to process track {track_name}: {e}")
                processed_tracks[track_name] = {}
                track_stats[track_name] = {
                    'successful_data_types': 0,
                    'total_data_types': len(track_data),
                    'total_rows': 0,
                    'success': False,
                    'error': str(e)
                }

        # Enhanced summary report
        successful_tracks = sum(1 for stats in track_stats.values() if stats['success'])
        total_processed_rows = sum(stats['total_rows'] for stats in track_stats.values())
        
        self._log("🎉", f"\nMulti-track preprocessing complete!")
        self._log("📊", f"Successful tracks: {successful_tracks}/{len(all_track_data)}")
        self._log("📊", f"Total processed rows: {total_processed_rows}")
        
        return processed_tracks

    # --------------------
    # Enhanced Helper Methods
    # --------------------
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced dataframe normalization with comprehensive cleaning"""
        try:
            df = df.copy()
            original_columns = list(df.columns)
            
            # Clean column names while preserving structure
            new_cols = {}
            for c in df.columns:
                cleaned = self._clean_col(str(c))
                new_cols[c] = cleaned
            
            df = df.rename(columns=new_cols)

            # Handle duplicate columns after normalization
            cols = df.columns.tolist()
            seen = {}
            unique_cols = []
            
            for c in cols:
                if c in seen:
                    seen[c] += 1
                    new_name = f"{c}_{seen[c]}"
                    unique_cols.append(new_name)
                    self._log("🔧", f"Renamed duplicate column: {c} -> {new_name}")
                else:
                    seen[c] = 0
                    unique_cols.append(c)
            
            df.columns = unique_cols

            # Enhanced string column cleaning
            for c in df.select_dtypes(include=['object']).columns:
                try:
                    df[c] = df[c].astype(str).str.strip().replace({
                        'None': '', 'nan': '', 'NaN': '', 'null': '', 'NULL': '', '': np.nan
                    })
                except Exception as e:
                    self._log("⚠️", f"Failed to clean string column {c}: {e}")

            if self.debug and set(original_columns) != set(df.columns):
                self._log("🔧", f"Column normalization: {original_columns} -> {list(df.columns)}")

            return df

        except Exception as e:
            self._log("❌", f"Dataframe normalization failed: {e}")
            return df  # Return original if normalization fails

    def _clean_col(self, col: str) -> str:
        """Enhanced column name cleaning"""
        if not isinstance(col, str):
            return str(col)
        
        # Remove BOM and whitespace
        col = col.lstrip('\ufeff').strip()
        
        # Enhanced character replacement
        col = re.sub(r'[^0-9A-Za-z_]+', '_', col)
        col = re.sub(r'__+', '_', col)
        col = col.strip('_')
        
        return col.upper()

    def time_to_seconds(self, time_str: Any) -> float:
        """Enhanced time conversion with comprehensive format support"""
        if pd.isna(time_str) or time_str == 0:
            return 0.0
            
        try:
            s = str(time_str).strip()
            if s == '' or s.upper() in {'-', 'NULL', 'NONE', 'NAN', 'NaN'}:
                return 0.0

            # Handle numeric strings directly
            try:
                return float(s)
            except ValueError:
                pass

            # Remove + sign and other prefixes
            s = s.lstrip('+').lstrip('-')

            # Enhanced format handling
            parts = s.split(':')
            
            if len(parts) == 3:  # HH:MM:SS.ms
                hours, minutes, seconds = parts
                return float(hours) * 3600.0 + float(minutes) * 60.0 + float(seconds)
            elif len(parts) == 2:  # MM:SS.ms
                minutes, seconds = parts
                return float(minutes) * 60.0 + float(seconds)
            else:
                # Try direct conversion for decimal seconds
                return float(s)
                
        except (ValueError, TypeError, AttributeError) as e:
            if self.debug:
                self._log("⚠️", f"Time conversion failed for '{time_str}': {e}")
            return 0.0

    def gap_to_seconds(self, gap_str: Any) -> float:
        """Enhanced gap conversion with comprehensive parsing"""
        if pd.isna(gap_str) or gap_str == 0:
            return 0.0
            
        try:
            s = str(gap_str).strip()
            if s == '' or s.upper() in {'-', 'NULL', 'NONE', 'NAN'}:
                return 0.0
                
            # Remove leading +/-
            s = s.lstrip('+').lstrip('-')
            
            # Handle colon format
            if ':' in s:
                return self.time_to_seconds(s)
                
            # Handle decimal format
            return float(s)
            
        except (ValueError, TypeError) as e:
            if self.debug:
                self._log("⚠️", f"Gap conversion failed for '{gap_str}': {e}")
            return 0.0

    def validate_processed_data(self, processed_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Enhanced data validation with comprehensive reporting
        """
        validation_results = {}
        
        self._log("🔍", "Starting data validation...")
        
        for track_name, track_data in processed_data.items():
            track_validation = {}
            
            for data_type, df in track_data.items():
                if df.empty:
                    track_validation[data_type] = {
                        'status': 'empty', 
                        'rows': 0, 
                        'columns': 0,
                        'message': 'No data available'
                    }
                else:
                    # Check for required columns
                    required_cols = self.required_columns.get(data_type, [])
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    # Calculate data quality metrics
                    null_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                    zero_percentage = ((df == 0).sum().sum() / (len(df) * len(df.columns))) * 100
                    
                    status = 'valid' if not missing_cols else 'missing_columns'
                    
                    track_validation[data_type] = {
                        'status': status,
                        'rows': len(df),
                        'columns': len(df.columns),
                        'missing_required_cols': missing_cols,
                        'null_percentage': null_percentage,
                        'zero_percentage': zero_percentage,
                        'data_quality': 'good' if null_percentage < 10 else 'fair' if null_percentage < 30 else 'poor'
                    }
            
            validation_results[track_name] = track_validation

        # Enhanced validation summary
        self._print_enhanced_validation_summary(validation_results)
        
        return validation_results

    def _print_enhanced_validation_summary(self, validation_results: Dict[str, Any]):
        """Enhanced validation summary with detailed reporting"""
        print("\n" + "="*70)
        print("ENHANCED DATA VALIDATION SUMMARY")
        print("="*70)
        
        total_tracks = len(validation_results)
        valid_tracks = 0
        
        for track_name, track_validation in validation_results.items():
            track_valid = True
            print(f"\n🏁 {track_name.upper()}:")
            
            for data_type, validation in track_validation.items():
                status_icon = "✅" if validation['status'] == 'valid' else "❌"
                quality_icon = "🟢" if validation.get('data_quality') == 'good' else "🟡" if validation.get('data_quality') == 'fair' else "🔴"
                
                print(f"  {status_icon} {quality_icon} {data_type}:")
                print(f"     Rows: {validation['rows']}, Columns: {validation['columns']}")
                
                if validation['status'] != 'valid':
                    print(f"     ❌ Missing: {validation['missing_required_cols']}")
                    track_valid = False
                
                if 'null_percentage' in validation:
                    print(f"     📊 Nulls: {validation['null_percentage']:.1f}%, Zeros: {validation['zero_percentage']:.1f}%")
            
            if track_valid:
                valid_tracks += 1
        
        print(f"\n📈 OVERALL SUMMARY:")
        print(f"   ✅ Valid Tracks: {valid_tracks}/{total_tracks}")
        print(f"   📊 Success Rate: {(valid_tracks/total_tracks*100):.1f}%")

    def get_data_quality_report(self, processed_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report
        """
        quality_report = {
            'summary': {},
            'tracks': {},
            'recommendations': []
        }
        
        validation_results = self.validate_processed_data(processed_data)
        
        # Calculate overall statistics
        total_rows = 0
        total_columns = 0
        tracks_with_issues = 0
        
        for track_name, track_validation in validation_results.items():
            track_report = {
                'data_types': {},
                'issues': []
            }
            
            for data_type, validation in track_validation.items():
                track_report['data_types'][data_type] = {
                    'rows': validation['rows'],
                    'columns': validation['columns'],
                    'status': validation['status'],
                    'data_quality': validation.get('data_quality', 'unknown')
                }
                
                total_rows += validation['rows']
                total_columns += validation['columns']
                
                # Track issues
                if validation['status'] != 'valid':
                    track_report['issues'].append(f"{data_type}: missing {validation['missing_required_cols']}")
                
                if validation.get('null_percentage', 0) > 20:
                    track_report['issues'].append(f"{data_type}: high null percentage ({validation['null_percentage']:.1f}%)")
            
            quality_report['tracks'][track_name] = track_report
            
            if track_report['issues']:
                tracks_with_issues += 1
        
        # Generate summary
        quality_report['summary'] = {
            'total_tracks': len(validation_results),
            'tracks_with_issues': tracks_with_issues,
            'total_rows': total_rows,
            'total_columns': total_columns,
            'overall_quality': 'good' if tracks_with_issues == 0 else 'fair' if tracks_with_issues <= len(validation_results) / 2 else 'poor'
        }
        
        # Generate recommendations
        if tracks_with_issues > 0:
            quality_report['recommendations'].append(f"Address data issues in {tracks_with_issues} tracks")
        
        if total_rows == 0:
            quality_report['recommendations'].append("No data available - check data sources")
        
        return quality_report





















# import re
# import pandas as pd
# import numpy as np
# from typing import Dict, Any, Optional, List
# from datetime import datetime

# class DataPreprocessor:
#     """
#     Data preprocessor consistent with FirebaseDataLoader schemas.
#     Uses EXACT column names from the data structures you provided.
#     """

#     @staticmethod
#     def _debug_print(debug: bool, *args, **kwargs):
#         if debug:
#             print(*args, **kwargs)

#     # --------------------
#     # Public preprocessors - Updated for FirebaseDataLoader consistency
#     # --------------------
    
#     @staticmethod
#     def preprocess_pit_data(pit_df: Optional[pd.DataFrame], debug: bool = False) -> pd.DataFrame:
#         """Preprocess pit data using EXACT column names from FirebaseDataLoader"""
#         if pit_df is None or pit_df.empty:
#             DataPreprocessor._debug_print(debug, "⚠️ Pit data is None or empty, returning empty DataFrame")
#             return pd.DataFrame()

#         df = pit_df.copy()
#         df = DataPreprocessor._normalize_dataframe(df, debug=debug)

#         # Convert time columns to seconds using EXACT column names
#         time_columns = ['LAP_TIME', 'S1', 'S2', 'S3', 'PIT_TIME', 'FL_TIME']
#         for col in time_columns:
#             if col in df.columns:
#                 df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.time_to_seconds)

#         # Convert intermediate timing columns to seconds
#         intermediate_times = ['IM1a_time', 'IM1_time', 'IM2a_time', 'IM2_time', 'IM3a_time', 'FL_time']
#         for col in intermediate_times:
#             if col in df.columns:
#                 df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.time_to_seconds)

#         # Convert elapsed columns to seconds
#         elapsed_columns = ['IM1a_elapsed', 'IM1_elapsed', 'IM2a_elapsed', 'IM2_elapsed', 'IM3a_elapsed', 'FL_elapsed']
#         for col in elapsed_columns:
#             if col in df.columns:
#                 df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.time_to_seconds)

#         # Ensure numeric columns using EXACT names
#         numeric_columns = [
#             'NUMBER', 'DRIVER_NUMBER', 'LAP_NUMBER', 'LAP_IMPROVEMENT', 
#             'S1_IMPROVEMENT', 'S2_IMPROVEMENT', 'S3_IMPROVEMENT',
#             'KPH', 'TOP_SPEED', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'
#         ]
#         for col in numeric_columns:
#             if col in df.columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

#         # Handle large sector times if present
#         large_sector_cols = ['S1_LARGE', 'S2_LARGE', 'S3_LARGE']
#         for col in large_sector_cols:
#             if col in df.columns:
#                 df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.time_to_seconds)

#         # Create performance metrics
#         if 'LAP_TIME_SECONDS' in df.columns:
#             df['PERFORMANCE_DROP'] = df.groupby('NUMBER')['LAP_TIME_SECONDS'].transform(lambda x: x - x.min())
#             df['CONSISTENCY'] = df.groupby('NUMBER')['LAP_TIME_SECONDS'].transform('std').fillna(0)

#         DataPreprocessor._debug_print(debug, f"✅ Processed pit data: {len(df)} rows, {len(df.columns)} columns")
#         return df

#     @staticmethod
#     def preprocess_race_data(race_df: Optional[pd.DataFrame], debug: bool = False) -> pd.DataFrame:
#         """Preprocess race data using EXACT column names from FirebaseDataLoader"""
#         if race_df is None or race_df.empty:
#             DataPreprocessor._debug_print(debug, "⚠️ Race data is None or empty, returning empty DataFrame")
#             return pd.DataFrame()

#         df = race_df.copy()
#         df = DataPreprocessor._normalize_dataframe(df, debug=debug)

#         # Ensure position is numeric
#         if 'POSITION' in df.columns:
#             df['POSITION'] = pd.to_numeric(df['POSITION'], errors='coerce').fillna(0)

#         # Convert time columns to seconds using EXACT names
#         if 'TOTAL_TIME' in df.columns:
#             df['TOTAL_TIME_SECONDS'] = df['TOTAL_TIME'].apply(DataPreprocessor.time_to_seconds)

#         if 'FL_TIME' in df.columns:
#             df['FL_TIME_SECONDS'] = df['FL_TIME'].apply(DataPreprocessor.time_to_seconds)

#         # Parse gap columns
#         gap_columns = ['GAP_FIRST', 'GAP_PREVIOUS']
#         for col in gap_columns:
#             if col in df.columns:
#                 df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.gap_to_seconds)

#         # Ensure numeric columns using EXACT names
#         numeric_columns = [
#             'NUMBER', 'LAPS', 'FL_LAPNUM', 'FL_KPH'
#         ]
#         for col in numeric_columns:
#             if col in df.columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

#         # Handle status and classification columns
#         if 'STATUS' in df.columns:
#             df['STATUS'] = df['STATUS'].astype(str).fillna('Unknown')

#         DataPreprocessor._debug_print(debug, f"✅ Processed race data: {len(df)} rows, {len(df.columns)} columns")
#         return df

#     @staticmethod
#     def preprocess_weather_data(weather_df: Optional[pd.DataFrame], debug: bool = False) -> pd.DataFrame:
#         """Preprocess weather data using EXACT column names from FirebaseDataLoader"""
#         if weather_df is None or weather_df.empty:
#             DataPreprocessor._debug_print(debug, "⚠️ Weather data is None or empty, returning empty DataFrame")
#             return pd.DataFrame()

#         df = weather_df.copy()
#         df = DataPreprocessor._normalize_dataframe(df, debug=debug)

#         # Create timestamp from EXACT column names
#         if 'TIME_UTC_SECONDS' in df.columns:
#             df['timestamp'] = pd.to_datetime(df['TIME_UTC_SECONDS'], unit='s', errors='coerce')
#         elif 'TIME_UTC_STR' in df.columns:
#             df['timestamp'] = pd.to_datetime(df['TIME_UTC_STR'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

#         # Ensure all weather metrics are numeric using EXACT names
#         weather_metrics = [
#             'AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 'PRESSURE', 
#             'WIND_SPEED', 'WIND_DIRECTION', 'RAIN'
#         ]
#         for col in weather_metrics:
#             if col in df.columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

#         # Remove rows with invalid timestamps
#         df = df.dropna(subset=['timestamp'])

#         DataPreprocessor._debug_print(debug, f"✅ Processed weather data: {len(df)} rows, {len(df.columns)} columns")
#         return df

#     @staticmethod
#     def preprocess_telemetry_data(telemetry_df: Optional[pd.DataFrame], debug: bool = False) -> pd.DataFrame:
#         """Preprocess telemetry data using EXACT column names from FirebaseDataLoader"""
#         if telemetry_df is None or telemetry_df.empty:
#             DataPreprocessor._debug_print(debug, "⚠️ Telemetry data is None or empty, returning empty DataFrame")
#             return pd.DataFrame()

#         df = telemetry_df.copy()
#         df = DataPreprocessor._normalize_dataframe(df, debug=debug)

#         # Convert timestamp using EXACT column name
#         if 'timestamp' in df.columns:
#             df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

#         # Ensure numeric columns using EXACT names
#         numeric_columns = [
#             'lap', 'outing', 'accx_can', 'accy_can', 'gear', 'speed'
#         ]
#         for col in numeric_columns:
#             if col in df.columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

#         # Remove rows with invalid timestamps
#         if 'timestamp' in df.columns:
#             df = df.dropna(subset=['timestamp'])

#         # Create derived metrics
#         if all(col in df.columns for col in ['accx_can', 'accy_can']):
#             df['total_acceleration'] = np.sqrt(df['accx_can']**2 + df['accy_can']**2)

#         DataPreprocessor._debug_print(debug, f"✅ Processed telemetry data: {len(df)} rows, {len(df.columns)} columns")
#         return df

#     # --------------------
#     # Batch preprocessing for FirebaseDataLoader output
#     # --------------------
    
#     @staticmethod
#     def preprocess_track_data(track_data: Dict[str, pd.DataFrame], debug: bool = False) -> Dict[str, pd.DataFrame]:
#         """
#         Preprocess all data types for a single track using EXACT FirebaseDataLoader structure
#         """
#         processed_data = {}
        
#         # Preprocess each data type using EXACT keys from FirebaseDataLoader
#         if 'pit_data' in track_data:
#             processed_data['pit_data'] = DataPreprocessor.preprocess_pit_data(track_data['pit_data'], debug)
        
#         if 'race_data' in track_data:
#             processed_data['race_data'] = DataPreprocessor.preprocess_race_data(track_data['race_data'], debug)
        
#         if 'weather_data' in track_data:
#             processed_data['weather_data'] = DataPreprocessor.preprocess_weather_data(track_data['weather_data'], debug)
        
#         if 'telemetry_data' in track_data:
#             processed_data['telemetry_data'] = DataPreprocessor.preprocess_telemetry_data(track_data['telemetry_data'], debug)

#         # Report processing status
#         for data_type, df in processed_data.items():
#             status = "✅" if not df.empty else "❌"
#             DataPreprocessor._debug_print(debug, f"{status} {data_type}: {len(df)} rows")

#         return processed_data

#     @staticmethod
#     def preprocess_all_tracks(all_track_data: Dict[str, Dict[str, pd.DataFrame]], debug: bool = False) -> Dict[str, Dict[str, pd.DataFrame]]:
#         """
#         Preprocess data for all tracks using EXACT FirebaseDataLoader structure
#         """
#         processed_tracks = {}
        
#         for track_name, track_data in all_track_data.items():
#             DataPreprocessor._debug_print(debug, f"\n🔧 Processing track: {track_name}")
#             processed_tracks[track_name] = DataPreprocessor.preprocess_track_data(track_data, debug)
        
#         # Summary report
#         total_tracks = len(processed_tracks)
#         successful_tracks = sum(1 for track_data in processed_tracks.values() 
#                               if any(not df.empty for df in track_data.values()))
        
#         DataPreprocessor._debug_print(debug, f"\n📊 Preprocessing Summary: {successful_tracks}/{total_tracks} tracks processed successfully")
        
#         return processed_tracks

#     # --------------------
#     # Helper methods
#     # --------------------
    
#     @staticmethod
#     def _normalize_dataframe(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
#         """Clean column names while preserving EXACT naming from FirebaseDataLoader"""
#         # Clean column names but preserve the exact structure
#         new_cols = {}
#         for c in df.columns:
#             cleaned = DataPreprocessor._clean_col(str(c))
#             new_cols[c] = cleaned
#         df = df.rename(columns=new_cols)

#         # Handle duplicate columns after normalization
#         cols = df.columns.tolist()
#         seen = {}
#         unique_cols = []
#         for c in cols:
#             if c in seen:
#                 seen[c] += 1
#                 new_name = f"{c}_{seen[c]}"
#                 unique_cols.append(new_name)
#             else:
#                 seen[c] = 0
#                 unique_cols.append(c)
#         df.columns = unique_cols

#         # Clean string columns
#         for c in df.select_dtypes(include=['object']).columns:
#             try:
#                 df[c] = df[c].astype(str).str.strip().replace({
#                     'None': '', 'nan': '', 'NaN': '', 'null': '', 'NULL': ''
#                 })
#             except Exception:
#                 pass

#         DataPreprocessor._debug_print(debug, f"Normalized columns: {list(df.columns)}")
#         return df

#     @staticmethod
#     def _clean_col(col: str) -> str:
#         """Normalize column names while preserving key naming conventions"""
#         if not isinstance(col, str):
#             return str(col)
        
#         # Remove BOM if present
#         col = col.lstrip('\ufeff').strip()
        
#         # Replace multiple non-alphanum with underscore, but preserve existing structure
#         col = re.sub(r'[^0-9A-Za-z_]+', '_', col)
#         col = re.sub(r'__+', '_', col)
        
#         return col.upper()

#     @staticmethod
#     def time_to_seconds(time_str: Any) -> float:
#         """
#         Convert time strings to seconds - consistent with FirebaseDataLoader implementation
#         Handles formats like: '1:54.168', '2:13.691', '1:37.428'
#         """
#         if pd.isna(time_str) or time_str == 0:
#             return 0.0
            
#         s = str(time_str).strip()
#         if s == '' or s.upper() in {'-', 'NULL', 'NONE', 'NAN'}:
#             return 0.0

#         # If already numeric
#         try:
#             return float(s)
#         except ValueError:
#             pass

#         # Remove + sign used in gaps
#         s = s.lstrip('+')

#         # Handle MM:SS.ms format (most common in your data)
#         parts = s.split(':')
#         try:
#             if len(parts) == 2:
#                 # MM:SS.ms
#                 minutes = float(parts[0])
#                 seconds = float(parts[1])
#                 return minutes * 60.0 + seconds
#             elif len(parts) == 3:
#                 # HH:MM:SS.ms
#                 hours = float(parts[0])
#                 minutes = float(parts[1])
#                 seconds = float(parts[2])
#                 return hours * 3600.0 + minutes * 60.0 + seconds
#             else:
#                 # Assume seconds
#                 return float(s)
#         except (ValueError, TypeError):
#             return 0.0

#     @staticmethod
#     def gap_to_seconds(gap_str: Any) -> float:
#         """Convert gap strings to seconds - consistent implementation"""
#         if pd.isna(gap_str) or gap_str == 0:
#             return 0.0
            
#         s = str(gap_str).strip()
#         if s == '' or s.upper() in {'-', 'NULL', 'NONE', 'NAN'}:
#             return 0.0
            
#         # Remove leading +
#         s = s.lstrip('+')
        
#         # If it contains colon, use time conversion
#         if ':' in s:
#             return DataPreprocessor.time_to_seconds(s)
            
#         # Try direct numeric conversion
#         try:
#             return float(s)
#         except (ValueError, TypeError):
#             return 0.0

#     @staticmethod
#     def validate_processed_data(processed_data: Dict[str, Dict[str, pd.DataFrame]], debug: bool = False) -> Dict[str, Any]:
#         """
#         Validate that processed data maintains consistency with FirebaseDataLoader schemas
#         """
#         validation_results = {}
        
#         for track_name, track_data in processed_data.items():
#             track_validation = {}
            
#             for data_type, df in track_data.items():
#                 if df.empty:
#                     track_validation[data_type] = {'status': 'empty', 'rows': 0, 'columns': 0}
#                 else:
#                     # Check for required columns based on data type
#                     required_cols = DataPreprocessor._get_required_columns(data_type)
#                     missing_cols = [col for col in required_cols if col not in df.columns]
                    
#                     track_validation[data_type] = {
#                         'status': 'valid' if not missing_cols else 'missing_columns',
#                         'rows': len(df),
#                         'columns': len(df.columns),
#                         'missing_required_cols': missing_cols,
#                         'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
#                     }
            
#             validation_results[track_name] = track_validation
        
#         # Print validation summary
#         if debug:
#             DataPreprocessor._print_validation_summary(validation_results)
        
#         return validation_results

#     @staticmethod
#     def _get_required_columns(data_type: str) -> List[str]:
#         """Get required columns for each data type based on FirebaseDataLoader schemas"""
#         required_columns = {
#             'pit_data': ['NUMBER', 'LAP_NUMBER', 'LAP_TIME'],
#             'race_data': ['POSITION', 'NUMBER', 'LAPS', 'TOTAL_TIME'],
#             'weather_data': ['TIME_UTC_SECONDS', 'AIR_TEMP', 'TRACK_TEMP'],
#             'telemetry_data': ['timestamp', 'vehicle_id', 'lap', 'speed']
#         }
#         return required_columns.get(data_type, [])

#     @staticmethod
#     def _print_validation_summary(validation_results: Dict[str, Any]):
#         """Print formatted validation summary"""
#         print("\n" + "="*60)
#         print("DATA VALIDATION SUMMARY")
#         print("="*60)
        
#         for track_name, track_validation in validation_results.items():
#             print(f"\n🏁 {track_name.upper()}:")
#             for data_type, validation in track_validation.items():
#                 status_icon = "✅" if validation['status'] == 'valid' else "❌"
#                 print(f"  {status_icon} {data_type}: {validation['rows']} rows, {validation['columns']} cols")
#                 if validation['missing_required_cols']:
#                     print(f"     Missing: {validation['missing_required_cols']}")