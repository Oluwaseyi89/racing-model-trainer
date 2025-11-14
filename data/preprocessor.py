import pandas as pd
import numpy as np
from typing import Dict, Any
import re

class DataPreprocessor:
    @staticmethod
    def preprocess_lap_data(lap_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare lap telemetry data for multiple data formats"""
        if lap_df is None or lap_df.empty:
            print("⚠️ Lap data is None or empty, returning empty DataFrame")
            return pd.DataFrame()
            
        df = lap_df.copy()
        
        # Handle different column naming conventions
        df = DataPreprocessor._standardize_column_names(df)
        
        # Convert various time formats to seconds
        time_columns = ['LAP_TIME', 'TIME', 'FL_TIME', 'S1', 'S2', 'S3', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']
        for col in time_columns:
            if col in df.columns:
                df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.time_to_seconds)
        
        # Convert gap strings to seconds
        gap_columns = ['GAP_FIRST', 'GAP_PREVIOUS']
        for col in gap_columns:
            if col in df.columns:
                df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.gap_to_seconds)
        
        # Handle different lap number column names
        lap_num_cols = ['LAP_NUMBER', 'LAP', 'LAPNUM', 'FL_LAPNUM']
        for col in lap_num_cols:
            if col in df.columns:
                df['LAP_NUMBER'] = pd.to_numeric(df[col], errors='coerce')
                break
        
        # Convert speed columns
        if 'KPH' in df.columns:
            df['KPH'] = pd.to_numeric(df['KPH'], errors='coerce')
        
        # Calculate performance metrics if we have lap times
        if 'LAP_TIME_SECONDS' in df.columns:
            df['PERFORMANCE_DROP'] = df.groupby('NUMBER')['LAP_TIME_SECONDS'].transform(
                lambda x: x - x.min()
            )
            df['CONSISTENCY'] = df.groupby('NUMBER')['LAP_TIME_SECONDS'].transform('std')
        
        # Handle sector time improvements
        improvement_cols = ['S1_IMPROVEMENT', 'S2_IMPROVEMENT', 'S3_IMPROVEMENT', 'LAP_IMPROVEMENT']
        for col in improvement_cols:
            if col in df.columns:
                df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.time_to_seconds)
        
        # Clean numeric columns
        numeric_cols = ['POSITION', 'POS', 'NUMBER', 'DRIVER_NUMBER', 'LAPS', 'TOTAL_LAPS']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    @staticmethod
    def preprocess_race_data(race_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare race results data"""
        # FIXED: Added proper None and empty DataFrame handling
        if race_df is None:
            print("⚠️ Race data is None, returning empty DataFrame")
            return pd.DataFrame()
            
        if race_df.empty:
            print("⚠️ Race data is empty, returning empty DataFrame")
            return pd.DataFrame()
            
        df = race_df.copy()
        
        # Standardize column names
        df = DataPreprocessor._standardize_column_names(df)
        
        # FIXED: Handle position columns with proper validation
        pos_cols = ['POSITION', 'POS', 'PIC']
        position_found = False
        
        for col in pos_cols:
            if col in df.columns and df[col] is not None:
                try:
                    # FIXED: Check if the column has valid data before processing
                    if not df[col].empty:
                        df['POSITION'] = pd.to_numeric(df[col], errors='coerce')
                        position_found = True
                        print(f"✅ Using position column: {col}")
                        break
                except Exception as e:
                    print(f"❌ Error processing position column {col}: {e}")
                    continue
        
        # FIXED: Create default positions if no valid position column found
        if not position_found:
            print("⚠️ No valid position column found, creating default positions")
            if len(df) > 0:
                df['POSITION'] = range(1, len(df) + 1)
            else:
                df['POSITION'] = pd.Series(dtype='float64')
        
        # Convert total race time
        if 'TOTAL_TIME' in df.columns:
            df['TOTAL_TIME_SECONDS'] = df['TOTAL_TIME'].apply(DataPreprocessor.race_time_to_seconds)
        
        # Convert gap strings
        gap_cols = ['GAP_FIRST', 'GAP_PREVIOUS']
        for col in gap_cols:
            if col in df.columns and df[col] is not None:
                try:
                    df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.gap_to_seconds)
                except Exception as e:
                    print(f"❌ Error processing gap column {col}: {e}")
                    df[f"{col}_SECONDS"] = 0.0
        
        # Handle best lap data
        if 'BEST_LAP_TIME' in df.columns:
            df['BEST_LAP_SECONDS'] = df['BEST_LAP_TIME'].apply(DataPreprocessor.time_to_seconds)
        elif 'FL_TIME' in df.columns:
            df['BEST_LAP_SECONDS'] = df['FL_TIME'].apply(DataPreprocessor.time_to_seconds)
        
        # Clean numeric columns with error handling
        numeric_cols = ['NUMBER', 'LAPS', 'FL_LAPNUM', 'BEST_LAP_NUM']
        for col in numeric_cols:
            if col in df.columns and df[col] is not None:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    print(f"❌ Error processing numeric column {col}: {e}")
                    df[col] = np.nan
        
        print(f"✅ Processed race data with {len(df)} entries")
        return df

    @staticmethod
    def preprocess_weather_data(weather_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare weather data"""
        if weather_df is None or weather_df.empty:
            print("⚠️ Weather data is None or empty, returning empty DataFrame")
            return pd.DataFrame()
            
        df = weather_df.copy()
        
        # Standardize column names
        df = DataPreprocessor._standardize_column_names(df)
        
        # Convert timestamps
        timestamp_cols = ['TIME_UTC_STR', 'TIME_UTC_SECONDS', 'timestamp']
        for col in timestamp_cols:
            if col in df.columns:
                if col == 'TIME_UTC_SECONDS':
                    df['timestamp'] = pd.to_datetime(df[col], unit='s')
                else:
                    df['timestamp'] = pd.to_datetime(df[col], errors='coerce')
                break
        
        # Clean numeric weather columns
        weather_cols = ['AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 'PRESSURE', 'WIND_SPEED', 'WIND_DIRECTION', 'RAIN']
        for col in weather_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    @staticmethod
    def preprocess_telemetry_data(telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare telemetry data"""
        if telemetry_df is None or telemetry_df.empty:
            print("⚠️ Telemetry data is None or empty, returning empty DataFrame")
            return pd.DataFrame()
            
        df = telemetry_df.copy()
        
        # Handle different telemetry column formats
        telemetry_mappings = {
            'accx_can': 'LONGITUDINAL_ACCEL',
            'accy_can': 'LATERAL_ACCEL', 
            'aps': 'THROTTLE_POSITION',
            'pbrake_f': 'BRAKE_PRESSURE_FRONT',
            'pbrake_r': 'BRAKE_PRESSURE_REAR',
            'gear': 'GEAR',
            'Steering_Angle': 'STEERING_ANGLE'
        }
        
        for old_col, new_col in telemetry_mappings.items():
            if old_col in df.columns:
                df[new_col] = pd.to_numeric(df[old_col], errors='coerce')
        
        # Calculate derived metrics
        if all(col in df.columns for col in ['BRAKE_PRESSURE_FRONT', 'BRAKE_PRESSURE_REAR']):
            df['TOTAL_BRAKE_PRESSURE'] = (df['BRAKE_PRESSURE_FRONT'] + df['BRAKE_PRESSURE_REAR']) / 2
        
        if all(col in df.columns for col in ['LONGITUDINAL_ACCEL', 'LATERAL_ACCEL']):
            df['TOTAL_ACCEL'] = np.sqrt(df['LONGITUDINAL_ACCEL']**2 + df['LATERAL_ACCEL']**2)
        
        # Convert timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        return df

    @staticmethod
    def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different data formats"""
        column_mappings = {
            'LAP': 'LAP_NUMBER',
            'LAPNUM': 'LAP_NUMBER', 
            'FL_LAPNUM': 'BEST_LAP_NUM',
            'TIME': 'LAP_TIME',
            'FL_TIME': 'BEST_LAP_TIME',
            'POS': 'POSITION',
            'PIC': 'POSITION',
            'DRIVER_NUMBER': 'NUMBER',
            'TOTAL_LAPS': 'LAPS'
        }
        
        df = df.rename(columns=column_mappings)
        return df

    @staticmethod
    def time_to_seconds(time_str: Any) -> float:
        """Convert various time formats to seconds"""
        if pd.isna(time_str) or time_str in ['', '-', 'NULL']:
            return np.nan
        
        time_str = str(time_str).strip()
        
        # Handle already numeric values
        try:
            return float(time_str)
        except ValueError:
            pass
        
        # Handle MM:SS.sss format (1:39.496)
        if ':' in time_str and '.' in time_str:
            try:
                parts = time_str.split(':')
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            except:
                pass
        
        # Handle MM:SS format
        if ':' in time_str:
            try:
                parts = time_str.split(':')
                if len(parts) == 2:
                    return float(parts[0]) * 60 + float(parts[1])
                elif len(parts) == 3:  # HH:MM:SS
                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            except:
                pass
        
        # Handle decimal seconds directly
        try:
            return float(time_str)
        except:
            return np.nan

    @staticmethod
    def gap_to_seconds(gap_str: Any) -> float:
        """Convert gap strings to seconds"""
        if pd.isna(gap_str) or gap_str in ['', '-', 'NULL']:
            return 0.0
        
        gap_str = str(gap_str).strip()
        
        # Remove + sign if present
        gap_str = gap_str.replace('+', '')
        
        # Handle already numeric values
        try:
            return float(gap_str)
        except ValueError:
            pass
        
        # Handle time format gaps (e.g., +0:00.234)
        if ':' in gap_str:
            return DataPreprocessor.time_to_seconds(gap_str.replace('+', ''))
        
        return 0.0

    @staticmethod
    def race_time_to_seconds(time_str: Any) -> float:
        """Convert race total time (MM:SS.sss) to seconds"""
        if pd.isna(time_str) or time_str in ['', '-']:
            return np.nan
        
        time_str = str(time_str).strip()
        
        # Handle formats like "46:41.553"
        if ':' in time_str and '.' in time_str:
            try:
                parts = time_str.split(':')
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            except:
                pass
        
        return DataPreprocessor.time_to_seconds(time_str)

    @staticmethod
    def merge_session_data(processed_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Merge all processed data types into a unified structure"""
        merged_data = {}
        
        # Ensure all data types are present
        for data_type in ['lap_data', 'race_data', 'weather_data', 'telemetry_data']:
            merged_data[data_type] = processed_data.get(data_type, pd.DataFrame())
        
        return merged_data




















# import pandas as pd
# import numpy as np
# from typing import Dict, Any
# import re

# class DataPreprocessor:
#     @staticmethod
#     def preprocess_lap_data(lap_df: pd.DataFrame) -> pd.DataFrame:
#         """Clean and prepare lap telemetry data for multiple data formats"""
#         if lap_df.empty:
#             return lap_df
            
#         df = lap_df.copy()
        
#         # Handle different column naming conventions
#         df = DataPreprocessor._standardize_column_names(df)
        
#         # Convert various time formats to seconds
#         time_columns = ['LAP_TIME', 'TIME', 'FL_TIME', 'S1', 'S2', 'S3', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']
#         for col in time_columns:
#             if col in df.columns:
#                 df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.time_to_seconds)
        
#         # Convert gap strings to seconds
#         gap_columns = ['GAP_FIRST', 'GAP_PREVIOUS']
#         for col in gap_columns:
#             if col in df.columns:
#                 df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.gap_to_seconds)
        
#         # Handle different lap number column names
#         lap_num_cols = ['LAP_NUMBER', 'LAP', 'LAPNUM', 'FL_LAPNUM']
#         for col in lap_num_cols:
#             if col in df.columns:
#                 df['LAP_NUMBER'] = pd.to_numeric(df[col], errors='coerce')
#                 break
        
#         # Convert speed columns
#         if 'KPH' in df.columns:
#             df['KPH'] = pd.to_numeric(df['KPH'], errors='coerce')
        
#         # Calculate performance metrics if we have lap times
#         if 'LAP_TIME_SECONDS' in df.columns:
#             df['PERFORMANCE_DROP'] = df.groupby('NUMBER')['LAP_TIME_SECONDS'].transform(
#                 lambda x: x - x.min()
#             )
#             df['CONSISTENCY'] = df.groupby('NUMBER')['LAP_TIME_SECONDS'].transform('std')
        
#         # Handle sector time improvements
#         improvement_cols = ['S1_IMPROVEMENT', 'S2_IMPROVEMENT', 'S3_IMPROVEMENT', 'LAP_IMPROVEMENT']
#         for col in improvement_cols:
#             if col in df.columns:
#                 df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.time_to_seconds)
        
#         # Clean numeric columns
#         numeric_cols = ['POSITION', 'POS', 'NUMBER', 'DRIVER_NUMBER', 'LAPS', 'TOTAL_LAPS']
#         for col in numeric_cols:
#             if col in df.columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
        
#         return df

#     @staticmethod
#     def preprocess_race_data(race_df: pd.DataFrame) -> pd.DataFrame:
#         """Clean and prepare race results data"""
#         if race_df.empty:
#             return race_df
            
#         df = race_df.copy()
        
#         # Standardize column names
#         df = DataPreprocessor._standardize_column_names(df)
        
#         # Handle position columns
#         pos_cols = ['POSITION', 'POS', 'PIC']
#         for col in pos_cols:
#             if col in df.columns:
#                 df['POSITION'] = pd.to_numeric(df[col], errors='coerce')
#                 break
        
#         # Convert total race time
#         if 'TOTAL_TIME' in df.columns:
#             df['TOTAL_TIME_SECONDS'] = df['TOTAL_TIME'].apply(DataPreprocessor.race_time_to_seconds)
        
#         # Convert gap strings
#         gap_cols = ['GAP_FIRST', 'GAP_PREVIOUS']
#         for col in gap_cols:
#             if col in df.columns:
#                 df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.gap_to_seconds)
        
#         # Handle best lap data
#         if 'BEST_LAP_TIME' in df.columns:
#             df['BEST_LAP_SECONDS'] = df['BEST_LAP_TIME'].apply(DataPreprocessor.time_to_seconds)
#         elif 'FL_TIME' in df.columns:
#             df['BEST_LAP_SECONDS'] = df['FL_TIME'].apply(DataPreprocessor.time_to_seconds)
        
#         # Clean numeric columns
#         numeric_cols = ['NUMBER', 'LAPS', 'FL_LAPNUM', 'BEST_LAP_NUM']
#         for col in numeric_cols:
#             if col in df.columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
        
#         return df

#     @staticmethod
#     def preprocess_weather_data(weather_df: pd.DataFrame) -> pd.DataFrame:
#         """Clean and prepare weather data"""
#         if weather_df.empty:
#             return weather_df
            
#         df = weather_df.copy()
        
#         # Standardize column names
#         df = DataPreprocessor._standardize_column_names(df)
        
#         # Convert timestamps
#         timestamp_cols = ['TIME_UTC_STR', 'TIME_UTC_SECONDS', 'timestamp']
#         for col in timestamp_cols:
#             if col in df.columns:
#                 if col == 'TIME_UTC_SECONDS':
#                     df['timestamp'] = pd.to_datetime(df[col], unit='s')
#                 else:
#                     df['timestamp'] = pd.to_datetime(df[col], errors='coerce')
#                 break
        
#         # Clean numeric weather columns
#         weather_cols = ['AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 'PRESSURE', 'WIND_SPEED', 'WIND_DIRECTION', 'RAIN']
#         for col in weather_cols:
#             if col in df.columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
        
#         return df

#     @staticmethod
#     def preprocess_telemetry_data(telemetry_df: pd.DataFrame) -> pd.DataFrame:
#         """Clean and prepare telemetry data"""
#         if telemetry_df.empty:
#             return telemetry_df
            
#         df = telemetry_df.copy()
        
#         # Handle different telemetry column formats
#         telemetry_mappings = {
#             'accx_can': 'LONGITUDINAL_ACCEL',
#             'accy_can': 'LATERAL_ACCEL', 
#             'aps': 'THROTTLE_POSITION',
#             'pbrake_f': 'BRAKE_PRESSURE_FRONT',
#             'pbrake_r': 'BRAKE_PRESSURE_REAR',
#             'gear': 'GEAR',
#             'Steering_Angle': 'STEERING_ANGLE'
#         }
        
#         for old_col, new_col in telemetry_mappings.items():
#             if old_col in df.columns:
#                 df[new_col] = pd.to_numeric(df[old_col], errors='coerce')
        
#         # Calculate derived metrics
#         if all(col in df.columns for col in ['BRAKE_PRESSURE_FRONT', 'BRAKE_PRESSURE_REAR']):
#             df['TOTAL_BRAKE_PRESSURE'] = (df['BRAKE_PRESSURE_FRONT'] + df['BRAKE_PRESSURE_REAR']) / 2
        
#         if all(col in df.columns for col in ['LONGITUDINAL_ACCEL', 'LATERAL_ACCEL']):
#             df['TOTAL_ACCEL'] = np.sqrt(df['LONGITUDINAL_ACCEL']**2 + df['LATERAL_ACCEL']**2)
        
#         # Convert timestamps
#         if 'timestamp' in df.columns:
#             df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
#         return df

#     @staticmethod
#     def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
#         """Standardize column names across different data formats"""
#         column_mappings = {
#             'LAP': 'LAP_NUMBER',
#             'LAPNUM': 'LAP_NUMBER', 
#             'FL_LAPNUM': 'BEST_LAP_NUM',
#             'TIME': 'LAP_TIME',
#             'FL_TIME': 'BEST_LAP_TIME',
#             'POS': 'POSITION',
#             'PIC': 'POSITION',
#             'DRIVER_NUMBER': 'NUMBER',
#             'TOTAL_LAPS': 'LAPS'
#         }
        
#         df = df.rename(columns=column_mappings)
#         return df

#     @staticmethod
#     def time_to_seconds(time_str: Any) -> float:
#         """Convert various time formats to seconds"""
#         if pd.isna(time_str) or time_str in ['', '-', 'NULL']:
#             return np.nan
        
#         time_str = str(time_str).strip()
        
#         # Handle already numeric values
#         try:
#             return float(time_str)
#         except ValueError:
#             pass
        
#         # Handle MM:SS.sss format (1:39.496)
#         if ':' in time_str and '.' in time_str:
#             try:
#                 parts = time_str.split(':')
#                 minutes = float(parts[0])
#                 seconds = float(parts[1])
#                 return minutes * 60 + seconds
#             except:
#                 pass
        
#         # Handle MM:SS format
#         if ':' in time_str:
#             try:
#                 parts = time_str.split(':')
#                 if len(parts) == 2:
#                     return float(parts[0]) * 60 + float(parts[1])
#                 elif len(parts) == 3:  # HH:MM:SS
#                     return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
#             except:
#                 pass
        
#         # Handle decimal seconds directly
#         try:
#             return float(time_str)
#         except:
#             return np.nan

#     @staticmethod
#     def gap_to_seconds(gap_str: Any) -> float:
#         """Convert gap strings to seconds"""
#         if pd.isna(gap_str) or gap_str in ['', '-', 'NULL']:
#             return 0.0
        
#         gap_str = str(gap_str).strip()
        
#         # Remove + sign if present
#         gap_str = gap_str.replace('+', '')
        
#         # Handle already numeric values
#         try:
#             return float(gap_str)
#         except ValueError:
#             pass
        
#         # Handle time format gaps (e.g., +0:00.234)
#         if ':' in gap_str:
#             return DataPreprocessor.time_to_seconds(gap_str.replace('+', ''))
        
#         return 0.0

#     @staticmethod
#     def race_time_to_seconds(time_str: Any) -> float:
#         """Convert race total time (MM:SS.sss) to seconds"""
#         if pd.isna(time_str) or time_str in ['', '-']:
#             return np.nan
        
#         time_str = str(time_str).strip()
        
#         # Handle formats like "46:41.553"
#         if ':' in time_str and '.' in time_str:
#             try:
#                 parts = time_str.split(':')
#                 minutes = float(parts[0])
#                 seconds = float(parts[1])
#                 return minutes * 60 + seconds
#             except:
#                 pass
        
#         return DataPreprocessor.time_to_seconds(time_str)

#     @staticmethod
#     def merge_session_data(processed_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
#         """Merge all processed data types into a unified structure"""
#         merged_data = {}
        
#         # Ensure all data types are present
#         for data_type in ['lap_data', 'race_data', 'weather_data', 'telemetry_data']:
#             merged_data[data_type] = processed_data.get(data_type, pd.DataFrame())
        
#         return merged_data
























# import pandas as pd
# import numpy as np

# class DataPreprocessor:
#     @staticmethod
#     def preprocess_lap_data(lap_df: pd.DataFrame) -> pd.DataFrame:
#         """Clean and prepare lap telemetry data"""
#         df = lap_df.copy()
        
#         # Convert time strings to seconds
#         df['LAP_TIME_SECONDS'] = df['LAP_TIME'].apply(DataPreprocessor.time_to_seconds)
#         df['S1_SECONDS'] = df['S1'].apply(DataPreprocessor.time_to_seconds)
#         df['S2_SECONDS'] = df['S2'].apply(DataPreprocessor.time_to_seconds)
#         df['S3_SECONDS'] = df['S3'].apply(DataPreprocessor.time_to_seconds)
        
#         # Calculate performance metrics
#         df['PERFORMANCE_DROP'] = df['LAP_TIME_SECONDS'] - df['LAP_TIME_SECONDS'].min()
#         df['TIRE_AGE'] = df['LAP_NUMBER']
        
#         return df
    
#     @staticmethod
#     def time_to_seconds(time_str: str) -> float:
#         """Convert MM:SS.sss format to seconds"""
#         if pd.isna(time_str) or time_str == '':
#             return np.nan
#         parts = time_str.split(':')
#         if len(parts) == 2:
#             return float(parts[0]) * 60 + float(parts[1])
#         return float(parts[0])