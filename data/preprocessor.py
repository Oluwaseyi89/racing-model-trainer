import re
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

class DataPreprocessor:
    """
    Data preprocessor consistent with FirebaseDataLoader schemas.
    Uses EXACT column names from the data structures you provided.
    """

    @staticmethod
    def _debug_print(debug: bool, *args, **kwargs):
        if debug:
            print(*args, **kwargs)

    # --------------------
    # Public preprocessors - Updated for FirebaseDataLoader consistency
    # --------------------
    
    @staticmethod
    def preprocess_pit_data(pit_df: Optional[pd.DataFrame], debug: bool = False) -> pd.DataFrame:
        """Preprocess pit data using EXACT column names from FirebaseDataLoader"""
        if pit_df is None or pit_df.empty:
            DataPreprocessor._debug_print(debug, "⚠️ Pit data is None or empty, returning empty DataFrame")
            return pd.DataFrame()

        df = pit_df.copy()
        df = DataPreprocessor._normalize_dataframe(df, debug=debug)

        # Convert time columns to seconds using EXACT column names
        time_columns = ['LAP_TIME', 'S1', 'S2', 'S3', 'PIT_TIME', 'FL_TIME']
        for col in time_columns:
            if col in df.columns:
                df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.time_to_seconds)

        # Convert intermediate timing columns to seconds
        intermediate_times = ['IM1a_time', 'IM1_time', 'IM2a_time', 'IM2_time', 'IM3a_time', 'FL_time']
        for col in intermediate_times:
            if col in df.columns:
                df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.time_to_seconds)

        # Convert elapsed columns to seconds
        elapsed_columns = ['IM1a_elapsed', 'IM1_elapsed', 'IM2a_elapsed', 'IM2_elapsed', 'IM3a_elapsed', 'FL_elapsed']
        for col in elapsed_columns:
            if col in df.columns:
                df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.time_to_seconds)

        # Ensure numeric columns using EXACT names
        numeric_columns = [
            'NUMBER', 'DRIVER_NUMBER', 'LAP_NUMBER', 'LAP_IMPROVEMENT', 
            'S1_IMPROVEMENT', 'S2_IMPROVEMENT', 'S3_IMPROVEMENT',
            'KPH', 'TOP_SPEED', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Handle large sector times if present
        large_sector_cols = ['S1_LARGE', 'S2_LARGE', 'S3_LARGE']
        for col in large_sector_cols:
            if col in df.columns:
                df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.time_to_seconds)

        # Create performance metrics
        if 'LAP_TIME_SECONDS' in df.columns:
            df['PERFORMANCE_DROP'] = df.groupby('NUMBER')['LAP_TIME_SECONDS'].transform(lambda x: x - x.min())
            df['CONSISTENCY'] = df.groupby('NUMBER')['LAP_TIME_SECONDS'].transform('std').fillna(0)

        DataPreprocessor._debug_print(debug, f"✅ Processed pit data: {len(df)} rows, {len(df.columns)} columns")
        return df

    @staticmethod
    def preprocess_race_data(race_df: Optional[pd.DataFrame], debug: bool = False) -> pd.DataFrame:
        """Preprocess race data using EXACT column names from FirebaseDataLoader"""
        if race_df is None or race_df.empty:
            DataPreprocessor._debug_print(debug, "⚠️ Race data is None or empty, returning empty DataFrame")
            return pd.DataFrame()

        df = race_df.copy()
        df = DataPreprocessor._normalize_dataframe(df, debug=debug)

        # Ensure position is numeric
        if 'POSITION' in df.columns:
            df['POSITION'] = pd.to_numeric(df['POSITION'], errors='coerce').fillna(0)

        # Convert time columns to seconds using EXACT names
        if 'TOTAL_TIME' in df.columns:
            df['TOTAL_TIME_SECONDS'] = df['TOTAL_TIME'].apply(DataPreprocessor.time_to_seconds)

        if 'FL_TIME' in df.columns:
            df['FL_TIME_SECONDS'] = df['FL_TIME'].apply(DataPreprocessor.time_to_seconds)

        # Parse gap columns
        gap_columns = ['GAP_FIRST', 'GAP_PREVIOUS']
        for col in gap_columns:
            if col in df.columns:
                df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.gap_to_seconds)

        # Ensure numeric columns using EXACT names
        numeric_columns = [
            'NUMBER', 'LAPS', 'FL_LAPNUM', 'FL_KPH'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Handle status and classification columns
        if 'STATUS' in df.columns:
            df['STATUS'] = df['STATUS'].astype(str).fillna('Unknown')

        DataPreprocessor._debug_print(debug, f"✅ Processed race data: {len(df)} rows, {len(df.columns)} columns")
        return df

    @staticmethod
    def preprocess_weather_data(weather_df: Optional[pd.DataFrame], debug: bool = False) -> pd.DataFrame:
        """Preprocess weather data using EXACT column names from FirebaseDataLoader"""
        if weather_df is None or weather_df.empty:
            DataPreprocessor._debug_print(debug, "⚠️ Weather data is None or empty, returning empty DataFrame")
            return pd.DataFrame()

        df = weather_df.copy()
        df = DataPreprocessor._normalize_dataframe(df, debug=debug)

        # Create timestamp from EXACT column names
        if 'TIME_UTC_SECONDS' in df.columns:
            df['timestamp'] = pd.to_datetime(df['TIME_UTC_SECONDS'], unit='s', errors='coerce')
        elif 'TIME_UTC_STR' in df.columns:
            df['timestamp'] = pd.to_datetime(df['TIME_UTC_STR'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

        # Ensure all weather metrics are numeric using EXACT names
        weather_metrics = [
            'AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 'PRESSURE', 
            'WIND_SPEED', 'WIND_DIRECTION', 'RAIN'
        ]
        for col in weather_metrics:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Remove rows with invalid timestamps
        df = df.dropna(subset=['timestamp'])

        DataPreprocessor._debug_print(debug, f"✅ Processed weather data: {len(df)} rows, {len(df.columns)} columns")
        return df

    @staticmethod
    def preprocess_telemetry_data(telemetry_df: Optional[pd.DataFrame], debug: bool = False) -> pd.DataFrame:
        """Preprocess telemetry data using EXACT column names from FirebaseDataLoader"""
        if telemetry_df is None or telemetry_df.empty:
            DataPreprocessor._debug_print(debug, "⚠️ Telemetry data is None or empty, returning empty DataFrame")
            return pd.DataFrame()

        df = telemetry_df.copy()
        df = DataPreprocessor._normalize_dataframe(df, debug=debug)

        # Convert timestamp using EXACT column name
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Ensure numeric columns using EXACT names
        numeric_columns = [
            'lap', 'outing', 'accx_can', 'accy_can', 'gear', 'speed'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Remove rows with invalid timestamps
        if 'timestamp' in df.columns:
            df = df.dropna(subset=['timestamp'])

        # Create derived metrics
        if all(col in df.columns for col in ['accx_can', 'accy_can']):
            df['total_acceleration'] = np.sqrt(df['accx_can']**2 + df['accy_can']**2)

        DataPreprocessor._debug_print(debug, f"✅ Processed telemetry data: {len(df)} rows, {len(df.columns)} columns")
        return df

    # --------------------
    # Batch preprocessing for FirebaseDataLoader output
    # --------------------
    
    @staticmethod
    def preprocess_track_data(track_data: Dict[str, pd.DataFrame], debug: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Preprocess all data types for a single track using EXACT FirebaseDataLoader structure
        """
        processed_data = {}
        
        # Preprocess each data type using EXACT keys from FirebaseDataLoader
        if 'pit_data' in track_data:
            processed_data['pit_data'] = DataPreprocessor.preprocess_pit_data(track_data['pit_data'], debug)
        
        if 'race_data' in track_data:
            processed_data['race_data'] = DataPreprocessor.preprocess_race_data(track_data['race_data'], debug)
        
        if 'weather_data' in track_data:
            processed_data['weather_data'] = DataPreprocessor.preprocess_weather_data(track_data['weather_data'], debug)
        
        if 'telemetry_data' in track_data:
            processed_data['telemetry_data'] = DataPreprocessor.preprocess_telemetry_data(track_data['telemetry_data'], debug)

        # Report processing status
        for data_type, df in processed_data.items():
            status = "✅" if not df.empty else "❌"
            DataPreprocessor._debug_print(debug, f"{status} {data_type}: {len(df)} rows")

        return processed_data

    @staticmethod
    def preprocess_all_tracks(all_track_data: Dict[str, Dict[str, pd.DataFrame]], debug: bool = False) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Preprocess data for all tracks using EXACT FirebaseDataLoader structure
        """
        processed_tracks = {}
        
        for track_name, track_data in all_track_data.items():
            DataPreprocessor._debug_print(debug, f"\n🔧 Processing track: {track_name}")
            processed_tracks[track_name] = DataPreprocessor.preprocess_track_data(track_data, debug)
        
        # Summary report
        total_tracks = len(processed_tracks)
        successful_tracks = sum(1 for track_data in processed_tracks.values() 
                              if any(not df.empty for df in track_data.values()))
        
        DataPreprocessor._debug_print(debug, f"\n📊 Preprocessing Summary: {successful_tracks}/{total_tracks} tracks processed successfully")
        
        return processed_tracks

    # --------------------
    # Helper methods
    # --------------------
    
    @staticmethod
    def _normalize_dataframe(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
        """Clean column names while preserving EXACT naming from FirebaseDataLoader"""
        # Clean column names but preserve the exact structure
        new_cols = {}
        for c in df.columns:
            cleaned = DataPreprocessor._clean_col(str(c))
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
            else:
                seen[c] = 0
                unique_cols.append(c)
        df.columns = unique_cols

        # Clean string columns
        for c in df.select_dtypes(include=['object']).columns:
            try:
                df[c] = df[c].astype(str).str.strip().replace({
                    'None': '', 'nan': '', 'NaN': '', 'null': '', 'NULL': ''
                })
            except Exception:
                pass

        DataPreprocessor._debug_print(debug, f"Normalized columns: {list(df.columns)}")
        return df

    @staticmethod
    def _clean_col(col: str) -> str:
        """Normalize column names while preserving key naming conventions"""
        if not isinstance(col, str):
            return str(col)
        
        # Remove BOM if present
        col = col.lstrip('\ufeff').strip()
        
        # Replace multiple non-alphanum with underscore, but preserve existing structure
        col = re.sub(r'[^0-9A-Za-z_]+', '_', col)
        col = re.sub(r'__+', '_', col)
        
        return col.upper()

    @staticmethod
    def time_to_seconds(time_str: Any) -> float:
        """
        Convert time strings to seconds - consistent with FirebaseDataLoader implementation
        Handles formats like: '1:54.168', '2:13.691', '1:37.428'
        """
        if pd.isna(time_str) or time_str == 0:
            return 0.0
            
        s = str(time_str).strip()
        if s == '' or s.upper() in {'-', 'NULL', 'NONE', 'NAN'}:
            return 0.0

        # If already numeric
        try:
            return float(s)
        except ValueError:
            pass

        # Remove + sign used in gaps
        s = s.lstrip('+')

        # Handle MM:SS.ms format (most common in your data)
        parts = s.split(':')
        try:
            if len(parts) == 2:
                # MM:SS.ms
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60.0 + seconds
            elif len(parts) == 3:
                # HH:MM:SS.ms
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600.0 + minutes * 60.0 + seconds
            else:
                # Assume seconds
                return float(s)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def gap_to_seconds(gap_str: Any) -> float:
        """Convert gap strings to seconds - consistent implementation"""
        if pd.isna(gap_str) or gap_str == 0:
            return 0.0
            
        s = str(gap_str).strip()
        if s == '' or s.upper() in {'-', 'NULL', 'NONE', 'NAN'}:
            return 0.0
            
        # Remove leading +
        s = s.lstrip('+')
        
        # If it contains colon, use time conversion
        if ':' in s:
            return DataPreprocessor.time_to_seconds(s)
            
        # Try direct numeric conversion
        try:
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def validate_processed_data(processed_data: Dict[str, Dict[str, pd.DataFrame]], debug: bool = False) -> Dict[str, Any]:
        """
        Validate that processed data maintains consistency with FirebaseDataLoader schemas
        """
        validation_results = {}
        
        for track_name, track_data in processed_data.items():
            track_validation = {}
            
            for data_type, df in track_data.items():
                if df.empty:
                    track_validation[data_type] = {'status': 'empty', 'rows': 0, 'columns': 0}
                else:
                    # Check for required columns based on data type
                    required_cols = DataPreprocessor._get_required_columns(data_type)
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    track_validation[data_type] = {
                        'status': 'valid' if not missing_cols else 'missing_columns',
                        'rows': len(df),
                        'columns': len(df.columns),
                        'missing_required_cols': missing_cols,
                        'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                    }
            
            validation_results[track_name] = track_validation
        
        # Print validation summary
        if debug:
            DataPreprocessor._print_validation_summary(validation_results)
        
        return validation_results

    @staticmethod
    def _get_required_columns(data_type: str) -> List[str]:
        """Get required columns for each data type based on FirebaseDataLoader schemas"""
        required_columns = {
            'pit_data': ['NUMBER', 'LAP_NUMBER', 'LAP_TIME'],
            'race_data': ['POSITION', 'NUMBER', 'LAPS', 'TOTAL_TIME'],
            'weather_data': ['TIME_UTC_SECONDS', 'AIR_TEMP', 'TRACK_TEMP'],
            'telemetry_data': ['timestamp', 'vehicle_id', 'lap', 'speed']
        }
        return required_columns.get(data_type, [])

    @staticmethod
    def _print_validation_summary(validation_results: Dict[str, Any]):
        """Print formatted validation summary"""
        print("\n" + "="*60)
        print("DATA VALIDATION SUMMARY")
        print("="*60)
        
        for track_name, track_validation in validation_results.items():
            print(f"\n🏁 {track_name.upper()}:")
            for data_type, validation in track_validation.items():
                status_icon = "✅" if validation['status'] == 'valid' else "❌"
                print(f"  {status_icon} {data_type}: {validation['rows']} rows, {validation['columns']} cols")
                if validation['missing_required_cols']:
                    print(f"     Missing: {validation['missing_required_cols']}")