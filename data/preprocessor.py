import re
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass


@dataclass
class DataSchema:
    """Unified schema definition for all data types"""
    required_columns: Set[str]
    optional_columns: Set[str]
    default_values: Dict[str, Any]
    column_dtypes: Dict[str, type]
    time_columns: Set[str]


class SchemaRegistry:
    """Registry for data schemas used across all trainers"""
    
    # Unified schemas based on trainer requirements
    LAP_DATA_SCHEMA = DataSchema(
        required_columns={'NUMBER', 'LAP_NUMBER', 'LAP_TIME_SECONDS'},
        optional_columns={'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS', 'KPH', 'POSITION', 
                         'PERFORMANCE_DROP', 'CONSISTENCY', 'timestamp'},
        default_values={
            'NUMBER': 1, 'LAP_NUMBER': 1, 'LAP_TIME_SECONDS': 60.0,
            'S1_SECONDS': 20.0, 'S2_SECONDS': 20.0, 'S3_SECONDS': 20.0,
            'KPH': 100.0, 'POSITION': 1, 'PERFORMANCE_DROP': 0.0, 'CONSISTENCY': 1.0
        },
        column_dtypes={
            'NUMBER': 'int64', 'LAP_NUMBER': 'int64', 'LAP_TIME_SECONDS': 'float64',
            'S1_SECONDS': 'float64', 'S2_SECONDS': 'float64', 'S3_SECONDS': 'float64',
            'KPH': 'float64', 'POSITION': 'int64', 'PERFORMANCE_DROP': 'float64', 
            'CONSISTENCY': 'float64'
        },
        time_columns={'timestamp'}
    )
    
    RACE_DATA_SCHEMA = DataSchema(
        required_columns={'NUMBER', 'POSITION'},
        optional_columns={'GAP_FIRST_SECONDS', 'GAP_PREVIOUS_SECONDS', 'TOTAL_TIME_SECONDS',
                         'BEST_LAP_SECONDS', 'LAPS', 'FL_LAPNUM'},
        default_values={
            'NUMBER': 1, 'POSITION': 1, 'GAP_FIRST_SECONDS': 0.0, 'GAP_PREVIOUS_SECONDS': 0.0,
            'TOTAL_TIME_SECONDS': 3600.0, 'BEST_LAP_SECONDS': 60.0, 'LAPS': 50, 'FL_LAPNUM': 1
        },
        column_dtypes={
            'NUMBER': 'int64', 'POSITION': 'int64', 'GAP_FIRST_SECONDS': 'float64',
            'GAP_PREVIOUS_SECONDS': 'float64', 'TOTAL_TIME_SECONDS': 'float64',
            'BEST_LAP_SECONDS': 'float64', 'LAPS': 'int64', 'FL_LAPNUM': 'int64'
        },
        time_columns=set()
    )
    
    WEATHER_DATA_SCHEMA = DataSchema(
        required_columns={'timestamp'},
        optional_columns={'AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 'PRESSURE', 
                         'WIND_SPEED', 'WIND_DIRECTION', 'RAIN'},
        default_values={
            'AIR_TEMP': 25.0, 'TRACK_TEMP': 30.0, 'HUMIDITY': 50.0, 'PRESSURE': 1013.0,
            'WIND_SPEED': 0.0, 'WIND_DIRECTION': 0.0, 'RAIN': 0.0
        },
        column_dtypes={
            'AIR_TEMP': 'float64', 'TRACK_TEMP': 'float64', 'HUMIDITY': 'float64',
            'PRESSURE': 'float64', 'WIND_SPEED': 'float64', 'WIND_DIRECTION': 'float64',
            'RAIN': 'float64', 'timestamp': 'datetime64[ns]'
        },
        time_columns={'timestamp'}
    )
    
    TELEMETRY_DATA_SCHEMA = DataSchema(
        required_columns={'vehicle_number', 'lap'},
        optional_columns={'THROTTLE_POSITION', 'BRAKE_PRESSURE_FRONT', 'BRAKE_PRESSURE_REAR',
                         'LONGITUDINAL_ACCEL', 'LATERAL_ACCEL', 'GEAR', 'KPH', 'TOP_SPEED',
                         'STEERING_ANGLE', 'TOTAL_BRAKE_PRESSURE', 'TOTAL_ACCEL', 'timestamp'},
        default_values={
            'vehicle_number': 1, 'lap': 1, 'THROTTLE_POSITION': 60.0, 
            'BRAKE_PRESSURE_FRONT': 25.0, 'BRAKE_PRESSURE_REAR': 25.0,
            'LONGITUDINAL_ACCEL': 0.0, 'LATERAL_ACCEL': 0.0, 'GEAR': 3,
            'KPH': 100.0, 'TOP_SPEED': 120.0, 'STEERING_ANGLE': 0.0,
            'TOTAL_BRAKE_PRESSURE': 25.0, 'TOTAL_ACCEL': 0.0
        },
        column_dtypes={
            'vehicle_number': 'int64', 'lap': 'int64', 'THROTTLE_POSITION': 'float64',
            'BRAKE_PRESSURE_FRONT': 'float64', 'BRAKE_PRESSURE_REAR': 'float64',
            'LONGITUDINAL_ACCEL': 'float64', 'LATERAL_ACCEL': 'float64', 'GEAR': 'int64',
            'KPH': 'float64', 'TOP_SPEED': 'float64', 'STEERING_ANGLE': 'float64',
            'TOTAL_BRAKE_PRESSURE': 'float64', 'TOTAL_ACCEL': 'float64', 'timestamp': 'datetime64[ns]'
        },
        time_columns={'timestamp'}
    )


class DataPreprocessor:
    """
    Robust preprocessor with unified schema validation and automatic missing data handling.
    - Uses strict schema validation for consistent data across all trainers
    - Automatically generates missing columns with appropriate defaults
    - Handles mixed data types and track variations seamlessly
    """

    # Canonical telemetry mapping with case-insensitive support
    TELEMETRY_MAP = {
        'accx_can': 'LONGITUDINAL_ACCEL',
        'accy_can': 'LATERAL_ACCEL', 
        'aps': 'THROTTLE_POSITION',
        'pbrake_f': 'BRAKE_PRESSURE_FRONT',
        'pbrake_r': 'BRAKE_PRESSURE_REAR',
        'pbrk_f': 'BRAKE_PRESSURE_FRONT',
        'pbrk_r': 'BRAKE_PRESSURE_REAR',
        'pbrk_rr': 'BRAKE_PRESSURE_REAR',
        'pbrace_f': 'BRAKE_PRESSURE_FRONT',
        'pbrace_r': 'BRAKE_PRESSURE_REAR',
        'gear': 'GEAR',
        'steering_angle': 'STEERING_ANGLE',
        'vbox_speed': 'KPH'
    }

    def __init__(self, debug: bool = False):
        self.debug = debug
        self._validated_schemas = {}
        self._first_valid_datasets = {}

    def _log(self, message: str):
        if self.debug:
            print(f"🔧 {message}")

    # --------------------
    # PUBLIC PREPROCESSORS
    # --------------------
    
    def preprocess_lap_data(self, lap_df: Optional[pd.DataFrame], track_name: str = "unknown") -> pd.DataFrame:
        """Preprocess lap data with schema validation and automatic column generation"""
        return self._preprocess_with_schema(
            lap_df, SchemaRegistry.LAP_DATA_SCHEMA, 'lap_data', track_name,
            self._enhance_lap_data
        )

    def preprocess_race_data(self, race_df: Optional[pd.DataFrame], track_name: str = "unknown") -> pd.DataFrame:
        """Preprocess race data with schema validation"""
        return self._preprocess_with_schema(
            race_df, SchemaRegistry.RACE_DATA_SCHEMA, 'race_data', track_name
        )

    def preprocess_weather_data(self, weather_df: Optional[pd.DataFrame], track_name: str = "unknown") -> pd.DataFrame:
        """Preprocess weather data with schema validation and timestamp handling"""
        return self._preprocess_with_schema(
            weather_df, SchemaRegistry.WEATHER_DATA_SCHEMA, 'weather_data', track_name,
            self._enhance_weather_data
        )

    def preprocess_telemetry_data(self, telemetry_df: Optional[pd.DataFrame], track_name: str = "unknown") -> pd.DataFrame:
        """Preprocess telemetry data with schema validation and column mapping"""
        return self._preprocess_with_schema(
            telemetry_df, SchemaRegistry.TELEMETRY_DATA_SCHEMA, 'telemetry_data', track_name,
            self._enhance_telemetry_data
        )

    # --------------------
    # CORE PROCESSING ENGINE
    # --------------------
    
    def _preprocess_with_schema(self, df: Optional[pd.DataFrame], schema: DataSchema, 
                               data_type: str, track_name: str, 
                               enhancer_func: callable = None) -> pd.DataFrame:
        """Core preprocessing with schema validation and enhancement"""
        try:
            # Handle empty input
            if df is None or df.empty:
                self._log(f"Empty {data_type} for {track_name}, generating schema-compliant data")
                return self._generate_schema_compliant_data(schema, data_type, track_name)

            # Store first valid dataset for schema learning
            if data_type not in self._first_valid_datasets:
                self._first_valid_datasets[data_type] = df.copy()
                self._log(f"📚 Learned schema from first valid {data_type} dataset")

            # Normalize and validate
            normalized_df = self._normalize_dataframe(df, data_type)
            validated_df = self._validate_and_enforce_schema(normalized_df, schema, data_type, track_name)
            
            # Apply data-specific enhancements
            if enhancer_func:
                validated_df = enhancer_func(validated_df, track_name)
            
            self._log(f"✅ Processed {data_type} for {track_name}: {validated_df.shape}")
            return validated_df

        except Exception as e:
            self._log(f"❌ {data_type} preprocessing failed for {track_name}: {e}")
            return self._generate_schema_compliant_data(schema, data_type, track_name)

    def _normalize_dataframe(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Normalize column names and ensure consistent structure"""
        df = df.copy()
        
        # Clean and normalize column names
        new_columns = {}
        for col in df.columns:
            cleaned = self._clean_column_name(col)
            new_columns[col] = cleaned
        
        df = df.rename(columns=new_columns)
        
        # Ensure unique column names
        df.columns = self._ensure_unique_columns(df.columns)
        
        # Convert object columns to proper types
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = self._safe_column_conversion(df[col])
        
        return df

    def _validate_and_enforce_schema(self, df: pd.DataFrame, schema: DataSchema, 
                                   data_type: str, track_name: str) -> pd.DataFrame:
        """Enforce schema compliance by adding missing columns and validating types"""
        result_df = df.copy()
        
        # Add missing required columns with defaults
        for col in schema.required_columns:
            if col not in result_df.columns:
                default_val = schema.default_values.get(col, 0)
                result_df[col] = default_val
                self._log(f"➕ Added missing required column '{col}' with default {default_val}")

        # Add missing optional columns with defaults
        for col in schema.optional_columns:
            if col not in result_df.columns:
                default_val = schema.default_values.get(col, 0)
                result_df[col] = default_val

        # Enforce data types
        for col, dtype in schema.column_dtypes.items():
            if col in result_df.columns:
                try:
                    if dtype == 'datetime64[ns]' and col in schema.time_columns:
                        result_df[col] = self._safe_datetime_parse(result_df[col])
                    else:
                        result_df[col] = result_df[col].astype(dtype)
                except (ValueError, TypeError) as e:
                    self._log(f"⚠️ Type conversion failed for {col}: {e}, using default")
                    result_df[col] = schema.default_values.get(col, 0)
        
        # Ensure required columns exist and are not empty
        missing_required = [col for col in schema.required_columns if col not in result_df.columns]
        if missing_required:
            self._log(f"❌ Missing required columns after processing: {missing_required}")
            for col in missing_required:
                result_df[col] = schema.default_values.get(col, 0)

        return result_df

    def _generate_schema_compliant_data(self, schema: DataSchema, data_type: str, track_name: str) -> pd.DataFrame:
        """Generate minimal schema-compliant data when input is invalid"""
        n_samples = 10  # Minimal dataset size
        
        data = {}
        for col in schema.required_columns | schema.optional_columns:
            default_val = schema.default_values.get(col, 0)
            if col in schema.time_columns:
                data[col] = pd.date_range('2025-01-01', periods=n_samples, freq='1min')
            elif isinstance(default_val, (int, np.integer)):
                data[col] = np.ones(n_samples, dtype=int) * default_val
            elif isinstance(default_val, (float, np.floating)):
                data[col] = np.ones(n_samples, dtype=float) * default_val
            else:
                data[col] = [default_val] * n_samples
        
        self._log(f"🔄 Generated schema-compliant {data_type} for {track_name}")
        return pd.DataFrame(data)

    # --------------------
    # DATA ENHANCEMENTS
    # --------------------
    
    def _enhance_lap_data(self, df: pd.DataFrame, track_name: str) -> pd.DataFrame:
        """Add calculated metrics to lap data"""
        result_df = df.copy()
        
        # Calculate performance metrics if lap time available
        if 'LAP_TIME_SECONDS' in result_df.columns:
            if 'NUMBER' in result_df.columns:
                result_df['PERFORMANCE_DROP'] = result_df.groupby('NUMBER')['LAP_TIME_SECONDS'].transform(
                    lambda x: x - x.min() if x.min() > 0 else 0.0
                )
                result_df['CONSISTENCY'] = result_df.groupby('NUMBER')['LAP_TIME_SECONDS'].transform('std').fillna(1.0)
        
        # Ensure time columns are properly formatted
        time_columns = ['LAP_TIME_SECONDS', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']
        for col in time_columns:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(60.0)
        
        return result_df

    def _enhance_weather_data(self, df: pd.DataFrame, track_name: str) -> pd.DataFrame:
        """Enhance weather data with calculated metrics"""
        result_df = df.copy()
        
        # Calculate derived weather metrics
        if all(col in result_df.columns for col in ['AIR_TEMP', 'TRACK_TEMP']):
            result_df['TEMP_DIFFERENCE'] = result_df['TRACK_TEMP'] - result_df['AIR_TEMP']
        
        if all(col in result_df.columns for col in ['AIR_TEMP', 'PRESSURE', 'HUMIDITY']):
            result_df['AIR_DENSITY'] = result_df.apply(
                lambda row: self._calculate_air_density(row['AIR_TEMP'], row['PRESSURE'], row['HUMIDITY']), 
                axis=1
            )
        
        return result_df

    def _enhance_telemetry_data(self, df: pd.DataFrame, track_name: str) -> pd.DataFrame:
        """Enhance telemetry data with mapping and derived metrics"""
        result_df = df.copy()
        
        # Apply telemetry mapping (case-insensitive)
        for col in list(result_df.columns):
            col_lower = col.lower()
            if col_lower in self.TELEMETRY_MAP:
                canonical_name = self.TELEMETRY_MAP[col_lower]
                if canonical_name not in result_df.columns:
                    result_df[canonical_name] = pd.to_numeric(result_df[col], errors='coerce')
        
        # Calculate derived telemetry metrics
        if all(col in result_df.columns for col in ['BRAKE_PRESSURE_FRONT', 'BRAKE_PRESSURE_REAR']):
            result_df['TOTAL_BRAKE_PRESSURE'] = (
                result_df['BRAKE_PRESSURE_FRONT'] + result_df['BRAKE_PRESSURE_REAR']
            ) / 2.0
        
        if all(col in result_df.columns for col in ['LONGITUDINAL_ACCEL', 'LATERAL_ACCEL']):
            result_df['TOTAL_ACCEL'] = np.sqrt(
                result_df['LONGITUDINAL_ACCEL']**2 + result_df['LATERAL_ACCEL']**2
            )
        
        return result_df

    # --------------------
    # UTILITY METHODS
    # --------------------
    
    def _clean_column_name(self, col: str) -> str:
        """Normalize column names consistently"""
        if not isinstance(col, str):
            return str(col)
        
        # Remove BOM and whitespace
        col = col.lstrip('\ufeff').strip()
        
        # Replace special characters with underscores
        col = re.sub(r'[^0-9A-Za-z]+', '_', col)
        col = re.sub(r'__+', '_', col)
        
        return col.upper()

    def _ensure_unique_columns(self, columns: List[str]) -> List[str]:
        """Ensure all column names are unique"""
        seen = {}
        unique_columns = []
        
        for col in columns:
            if col in seen:
                seen[col] += 1
                new_name = f"{col}_{seen[col]}"
                unique_columns.append(new_name)
            else:
                seen[col] = 0
                unique_columns.append(col)
        
        return unique_columns

    def _safe_column_conversion(self, series: pd.Series) -> pd.Series:
        """Safely convert object series to appropriate types"""
        try:
            # Try numeric conversion first
            numeric_series = pd.to_numeric(series, errors='coerce')
            if numeric_series.notna().sum() > len(series) * 0.8:  # Mostly numeric
                return numeric_series
            
            # Try datetime conversion
            datetime_series = self._safe_datetime_parse(series)
            if datetime_series.notna().sum() > len(series) * 0.8:  # Mostly datetime
                return datetime_series
            
            # Return cleaned string series
            return series.astype(str).str.strip().replace({'None': None, 'nan': None, 'NaN': None})
            
        except Exception:
            return series

    def _safe_datetime_parse(self, series: pd.Series) -> pd.Series:
        """Robust datetime parsing with single fallback strategy"""
        if series.empty:
            return pd.Series([], dtype='datetime64[ns]')
        
        # Try direct pandas parsing first (most robust)
        parsed = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
        
        # If mostly successful, return result
        if parsed.notna().sum() > len(parsed) * 0.7:
            return parsed
        
        # Fallback: generate sequential timestamps
        self._log("⚠️ Using fallback timestamp generation")
        base_time = pd.Timestamp('2025-01-01 14:00:00')
        return pd.Series([base_time + pd.Timedelta(seconds=i*90) for i in range(len(series))])

    def _calculate_air_density(self, air_temp: float, pressure: float, humidity: float) -> float:
        """Calculate air density from weather parameters"""
        try:
            R = 287.05
            temp_kelvin = air_temp + 273.15
            vapor_pressure = 0.611 * np.exp(17.27 * air_temp / (air_temp + 237.3)) * (humidity / 100)
            dry_air_pressure = pressure - vapor_pressure
            return (dry_air_pressure * 100) / (R * temp_kelvin)
        except:
            return 1.225  # Standard sea level density

    # --------------------
    # BATCH PROCESSING
    # --------------------
    
    def preprocess_track_data(self, raw_data: Dict[str, pd.DataFrame], track_name: str) -> Dict[str, pd.DataFrame]:
        """Preprocess all data types for a track with consistent schema"""
        processed = {}
        
        processors = {
            'lap_data': self.preprocess_lap_data,
            'race_data': self.preprocess_race_data, 
            'weather_data': self.preprocess_weather_data,
            'telemetry_data': self.preprocess_telemetry_data
        }
        
        for data_type, processor in processors.items():
            input_df = raw_data.get(data_type)
            processed[data_type] = processor(input_df, track_name)
        
        return processed

    def get_schema_compliance_report(self) -> Dict[str, Any]:
        """Generate report on schema compliance across processed datasets"""
        report = {
            'learned_schemas': list(self._first_valid_datasets.keys()),
            'schema_validation': {}
        }
        
        for data_type, schema in [
            ('lap_data', SchemaRegistry.LAP_DATA_SCHEMA),
            ('race_data', SchemaRegistry.RACE_DATA_SCHEMA),
            ('weather_data', SchemaRegistry.WEATHER_DATA_SCHEMA),
            ('telemetry_data', SchemaRegistry.TELEMETRY_DATA_SCHEMA)
        ]:
            report['schema_validation'][data_type] = {
                'required_columns': list(schema.required_columns),
                'optional_columns': list(schema.optional_columns),
                'has_validated_data': data_type in self._validated_schemas
            }
        
        return report


# Backward compatibility functions
def preprocess_lap_data(lap_df: Optional[pd.DataFrame], debug: bool = False) -> pd.DataFrame:
    return DataPreprocessor(debug).preprocess_lap_data(lap_df)

def preprocess_race_data(race_df: Optional[pd.DataFrame], debug: bool = False) -> pd.DataFrame:
    return DataPreprocessor(debug).preprocess_race_data(race_df)

def preprocess_weather_data(weather_df: Optional[pd.DataFrame], debug: bool = False) -> pd.DataFrame:
    return DataPreprocessor(debug).preprocess_weather_data(weather_df)

def preprocess_telemetry_data(telemetry_df: Optional[pd.DataFrame], debug: bool = False) -> pd.DataFrame:
    return DataPreprocessor(debug).preprocess_telemetry_data(telemetry_df)

def merge_session_data(processed_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Backward compatibility wrapper"""
    return {
        'lap_data': processed_data.get('lap_data', pd.DataFrame()),
        'race_data': processed_data.get('race_data', pd.DataFrame()),
        'weather_data': processed_data.get('weather_data', pd.DataFrame()),
        'telemetry_data': processed_data.get('telemetry_data', pd.DataFrame())
    }

























# import re
# import pandas as pd
# import numpy as np
# from typing import Dict, Any, Optional, Iterable, List


# def _clean_col(col: str) -> str:
#     """Normalize column names: strip BOM/whitespace, uppercase, replace separators with _"""
#     if not isinstance(col, str):
#         return col
#     # remove BOM if present
#     col = col.lstrip('\ufeff').strip()
#     # replace multiple non-alphanum with underscore
#     col = re.sub(r'[^0-9A-Za-z]+', '_', col)
#     col = re.sub(r'__+', '_', col)
#     return col.upper()


# class DataPreprocessor:
#     """
#     Robust preprocessor for lap, race, telemetry and weather datasets.
#     - Use DataPreprocessor.preprocess_* entrypoints
#     - Set debug=True to print diagnostics
#     """

#     # canonical telemetry name mappings
#     TELEMETRY_MAP = {
#         'ACCX_CAN': 'LONGITUDINAL_ACCEL',
#         'ACCY_CAN': 'LATERAL_ACCEL',
#         'APS': 'THROTTLE_POSITION',
#         'PBRACE_F': 'BRAKE_PRESSURE_FRONT',  # some datasets have typos, keep mapping flexible
#         'PBRK_F': 'BRAKE_PRESSURE_FRONT',
#         'PBRK_R': 'BRAKE_PRESSURE_REAR',
#         'PBRK_RR': 'BRAKE_PRESSURE_REAR',
#         'PBRK_R_': 'BRAKE_PRESSURE_REAR',
#         'PBRKE_R': 'BRAKE_PRESSURE_REAR',
#         'PBRK_RR': 'BRAKE_PRESSURE_REAR',
#         'PBRACE_R': 'BRAKE_PRESSURE_REAR',
#         'PBRK': 'BRAKE_PRESSURE',
#         'PBRK_FRONT': 'BRAKE_PRESSURE_FRONT',
#         'PBRK_REAR': 'BRAKE_PRESSURE_REAR',
#         'GEAR': 'GEAR',
#         'STEERING_ANGLE': 'STEERING_ANGLE',
#         'VBOX_LONG_MINUTES': 'VBOX_LONG_MIN',
#         'VBOX_LAT_MIN': 'VBOX_LAT_MIN',
#         # Add more if your telemetry keys vary
#     }

#     @staticmethod
#     def _debug_print(debug: bool, *args, **kwargs):
#         if debug:
#             print(*args, **kwargs)

#     # --------------------
#     # Public preprocessors
#     # --------------------
#     @staticmethod
#     def preprocess_lap_data(lap_df: Optional[pd.DataFrame], debug: bool = False) -> pd.DataFrame:
#         if lap_df is None:
#             DataPreprocessor._debug_print(debug, "⚠️ Lap data is None, returning empty DataFrame")
#             return pd.DataFrame()
#         if isinstance(lap_df, pd.Series):
#             lap_df = lap_df.to_frame().T
#         if lap_df.empty:
#             DataPreprocessor._debug_print(debug, "⚠️ Lap data is empty, returning empty DataFrame")
#             return pd.DataFrame()

#         df = lap_df.copy()
#         df = DataPreprocessor._normalize_dataframe(df, debug=debug)

#         # common time/sector fields to convert to seconds (if present)
#         time_columns = [
#             'LAP_TIME', 'TIME', 'FL_TIME', 'S1', 'S2', 'S3',
#             'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS', 'LAP_TIME_SECONDS'
#         ]
#         for c in time_columns:
#             if c in df.columns:
#                 df[f"{c}_SECONDS"] = df[c].apply(DataPreprocessor.time_to_seconds)

#         # gap columns
#         for col in ['GAP_FIRST', 'GAP_PREVIOUS', 'GAP']:
#             if col in df.columns:
#                 df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.gap_to_seconds)

#         # lap number detection
#         lap_num_candidates = ['LAP_NUMBER', 'LAP', 'LAPNUM', 'FL_LAPNUM']
#         for cand in lap_num_candidates:
#             if cand in df.columns and not df[cand].empty:
#                 df['LAP_NUMBER'] = pd.to_numeric(df[cand], errors='coerce')
#                 DataPreprocessor._debug_print(debug, f"✅ Found lap number column: {cand}")
#                 break

#         # driver number detection
#         num_candidates = ['NUMBER', 'DRIVER_NUMBER', 'DRIVERNO', 'DRIVER_NO']
#         for cand in num_candidates:
#             if cand in df.columns:
#                 df['NUMBER'] = pd.to_numeric(df[cand], errors='coerce')
#                 DataPreprocessor._debug_print(debug, f"✅ Found driver number column: {cand}")
#                 break

#         # KPH
#         if 'KPH' in df.columns:
#             df['KPH'] = pd.to_numeric(df['KPH'], errors='coerce')

#         # create summary metrics if lap_time present
#         if 'LAP_TIME_SECONDS' in df.columns:
#             df['PERFORMANCE_DROP'] = df.groupby('NUMBER')['LAP_TIME_SECONDS'].transform(lambda x: x - x.min())
#             df['CONSISTENCY'] = df.groupby('NUMBER')['LAP_TIME_SECONDS'].transform('std')

#         # improvements (sometimes are strings)
#         for inc in ['S1_IMPROVEMENT', 'S2_IMPROVEMENT', 'S3_IMPROVEMENT', 'LAP_IMPROVEMENT']:
#             if inc in df.columns:
#                 df[f"{inc}_SECONDS"] = df[inc].apply(DataPreprocessor.time_to_seconds)

#         # numeric cleanups
#         for c in ['POSITION', 'POS', 'PIC', 'NUMBER', 'LAPS', 'TOTAL_LAPS']:
#             if c in df.columns:
#                 df[c] = pd.to_numeric(df[c], errors='coerce')

#         DataPreprocessor._debug_print(debug, f"✅ Processed lap data with {len(df)} rows and columns: {list(df.columns)}")
#         return df

#     @staticmethod
#     def preprocess_race_data(race_df: Optional[pd.DataFrame], debug: bool = False) -> pd.DataFrame:
#         if race_df is None:
#             DataPreprocessor._debug_print(debug, "⚠️ Race data is None, returning empty DataFrame")
#             return pd.DataFrame()
#         if isinstance(race_df, pd.Series):
#             race_df = race_df.to_frame().T
#         if race_df.empty:
#             DataPreprocessor._debug_print(debug, "⚠️ Race data is empty, returning empty DataFrame")
#             return pd.DataFrame()

#         df = race_df.copy()
#         df = DataPreprocessor._normalize_dataframe(df, debug=debug)

#         # Position detection with safe checks to avoid ambiguous Series truth values
#         pos_candidates = ['POSITION', 'POS', 'PIC']
#         position_found = False
#         for cand in pos_candidates:
#             if cand in df.columns:
#                 col = df[cand]
#                 # explicit emptiness and 1-d checks
#                 if not col.empty and (isinstance(col, pd.Series) or hasattr(col, '__len__')):
#                     try:
#                         df['POSITION'] = pd.to_numeric(col, errors='coerce')
#                         position_found = True
#                         DataPreprocessor._debug_print(debug, f"✅ Using position column: {cand}")
#                         break
#                     except Exception as e:
#                         DataPreprocessor._debug_print(debug, f"❌ Error processing position column {cand}: {e}")
#                         continue

#         if not position_found:
#             DataPreprocessor._debug_print(debug, "⚠️ No valid position column found, creating default positions")
#             if len(df) > 0:
#                 df['POSITION'] = list(range(1, len(df) + 1))
#             else:
#                 df['POSITION'] = pd.Series(dtype='float64')

#         # total time -> seconds
#         if 'TOTAL_TIME' in df.columns:
#             df['TOTAL_TIME_SECONDS'] = df['TOTAL_TIME'].apply(DataPreprocessor.race_time_to_seconds)

#         # gap columns safe parse
#         for col in ['GAP_FIRST', 'GAP_PREVIOUS', 'GAP']:
#             if col in df.columns:
#                 try:
#                     df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.gap_to_seconds)
#                 except Exception as e:
#                     DataPreprocessor._debug_print(debug, f"❌ Error processing gap column {col}: {e}")
#                     df[f"{col}_SECONDS"] = np.nan

#         # best lap / fastest lap falls back FL_TIME
#         if 'BEST_LAP_TIME' in df.columns:
#             df['BEST_LAP_SECONDS'] = df['BEST_LAP_TIME'].apply(DataPreprocessor.time_to_seconds)
#         elif 'FL_TIME' in df.columns:
#             df['BEST_LAP_SECONDS'] = df['FL_TIME'].apply(DataPreprocessor.time_to_seconds)

#         # numeric conversions with safe fallback
#         for c in ['NUMBER', 'LAPS', 'FL_LAPNUM', 'BEST_LAP_NUM', 'PIC']:
#             if c in df.columns:
#                 try:
#                     df[c] = pd.to_numeric(df[c], errors='coerce')
#                 except Exception:
#                     df[c] = np.nan

#         DataPreprocessor._debug_print(debug, f"✅ Processed race data with {len(df)} rows and columns: {list(df.columns)}")
#         return df

#     @staticmethod
#     def preprocess_weather_data(weather_df: Optional[pd.DataFrame], debug: bool = False) -> pd.DataFrame:
#         if weather_df is None:
#             DataPreprocessor._debug_print(debug, "⚠️ Weather data is None, returning empty DataFrame")
#             return pd.DataFrame()
#         if isinstance(weather_df, pd.Series):
#             weather_df = weather_df.to_frame().T
#         if weather_df.empty:
#             DataPreprocessor._debug_print(debug, "⚠️ Weather data is empty, returning empty DataFrame")
#             return pd.DataFrame()

#         df = weather_df.copy()
#         df = DataPreprocessor._normalize_dataframe(df, debug=debug)

#         # timestamp columns
#         ts_candidates = ['TIME_UTC_SECONDS', 'TIME_UTC_STR', 'TIMESTAMP', 'TIME']
#         parsed = False
#         for cand in ts_candidates:
#             if cand in df.columns:
#                 if cand == 'TIME_UTC_SECONDS':
#                     try:
#                         df['timestamp'] = pd.to_datetime(pd.to_numeric(df[cand], errors='coerce'), unit='s', errors='coerce')
#                         parsed = True
#                     except Exception:
#                         parsed = False
#                 else:
#                     df['timestamp'] = DataPreprocessor._safe_datetime_parse(df[cand])
#                     parsed = True
#                 if parsed:
#                     DataPreprocessor._debug_print(debug, f"✅ Parsed weather timestamp from {cand}")
#                     break

#         # numeric weather columns
#         for c in ['AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 'PRESSURE', 'WIND_SPEED', 'WIND_DIRECTION', 'RAIN']:
#             if c in df.columns:
#                 df[c] = pd.to_numeric(df[c], errors='coerce')

#         DataPreprocessor._debug_print(debug, f"✅ Processed weather data with {len(df)} rows and columns: {list(df.columns)}")
#         return df

#     @staticmethod
#     def preprocess_telemetry_data(telemetry_df: Optional[pd.DataFrame], debug: bool = False) -> pd.DataFrame:
#         if telemetry_df is None:
#             DataPreprocessor._debug_print(debug, "⚠️ Telemetry data is None, returning empty DataFrame")
#             return pd.DataFrame()
#         if isinstance(telemetry_df, pd.Series):
#             telemetry_df = telemetry_df.to_frame().T
#         if telemetry_df.empty:
#             DataPreprocessor._debug_print(debug, "⚠️ Telemetry data is empty, returning empty DataFrame")
#             return pd.DataFrame()

#         df = telemetry_df.copy()
#         df = DataPreprocessor._normalize_dataframe(df, debug=debug)

#         # Some datasets are long format: columns telemetry_name, telemetry_value
#         # Support both long and wide formats
#         long_name_cols = [c for c in df.columns if 'TELEMETRY_NAME' in c or 'TELEMETRY' == c]
#         if {'TELEMETRY_NAME', 'TELEMETRY_VALUE'}.issubset(set(df.columns)):
#             # pivot long -> wide
#             try:
#                 pivoted = df.pivot_table(index=[c for c in df.columns if c not in ['TELEMETRY_NAME', 'TELEMETRY_VALUE']],
#                                          columns='TELEMETRY_NAME', values='TELEMETRY_VALUE', aggfunc='first').reset_index()
#                 df = DataPreprocessor._normalize_dataframe(pivoted, debug=debug)
#                 DataPreprocessor._debug_print(debug, "✅ Telemetry long->wide pivoted")
#             except Exception as e:
#                 DataPreprocessor._debug_print(debug, f"⚠️ Could not pivot telemetry long format: {e}")

#         # Rename known telemetry keys to canonical names and coerce numeric
#         for col in list(df.columns):
#             if col in DataPreprocessor.TELEMETRY_MAP:
#                 canonical = DataPreprocessor.TELEMETRY_MAP[col]
#                 df[canonical] = pd.to_numeric(df[col], errors='coerce')

#         # If telemetry keys appear in lower-case or camelCase, check that too
#         for col in list(df.columns):
#             up = col.upper()
#             if up in DataPreprocessor.TELEMETRY_MAP and up != col:
#                 canonical = DataPreprocessor.TELEMETRY_MAP[up]
#                 df[canonical] = pd.to_numeric(df[col], errors='coerce')

#         # Derived telemetry
#         if 'BRAKE_PRESSURE_FRONT' in df.columns and 'BRAKE_PRESSURE_REAR' in df.columns:
#             df['TOTAL_BRAKE_PRESSURE'] = (pd.to_numeric(df['BRAKE_PRESSURE_FRONT'], errors='coerce') +
#                                           pd.to_numeric(df['BRAKE_PRESSURE_REAR'], errors='coerce')) / 2.0

#         if 'LONGITUDINAL_ACCEL' in df.columns and 'LATERAL_ACCEL' in df.columns:
#             df['TOTAL_ACCEL'] = np.sqrt(
#                 pd.to_numeric(df['LONGITUDINAL_ACCEL'], errors='coerce')**2 +
#                 pd.to_numeric(df['LATERAL_ACCEL'], errors='coerce')**2
#             )

#         # safe timestamp parse if present
#         for cand in ['TIMESTAMP', 'TIME', 'TIME_UTC_STR', 'TIME_UTC_SECONDS']:
#             if cand in df.columns:
#                 if cand == 'TIME_UTC_SECONDS':
#                     df['timestamp'] = pd.to_datetime(pd.to_numeric(df[cand], errors='coerce'), unit='s', errors='coerce')
#                 else:
#                     df['timestamp'] = DataPreprocessor._safe_datetime_parse(df[cand])
#                 break

#         DataPreprocessor._debug_print(debug, f"✅ Processed telemetry data with {len(df)} rows and columns: {list(df.columns)}")
#         return df

#     # --------------------
#     # Helpers
#     # --------------------
#     @staticmethod
#     def _normalize_dataframe(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
#         """Clean column names and strip BOM/whitespace from string values"""
#         # Clean column names
#         new_cols = {}
#         for c in df.columns:
#             cleaned = _clean_col(str(c))
#             new_cols[c] = cleaned
#         df = df.rename(columns=new_cols)

#         # If any column names are duplicates after normalization, make them unique
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

#         # Strip strings in object columns
#         for c in df.select_dtypes(include=['object']).columns:
#             try:
#                 df[c] = df[c].astype(str).str.strip().replace({'None': None, 'nan': None, 'NaN': None})
#             except Exception:
#                 # keep original if conversion fails
#                 pass

#         DataPreprocessor._debug_print(debug, f"Normalized columns -> {list(df.columns)}")
#         return df

#     @staticmethod
#     def _safe_datetime_parse(date_series: pd.Series) -> pd.Series:
#         """Try multiple formats; fallback to pandas auto-parse"""
#         if date_series is None:
#             return pd.Series(dtype='datetime64[ns]')
#         if isinstance(date_series, (pd.DatetimeIndex, pd.Series)) and pd.api.types.is_datetime64_any_dtype(date_series):
#             return pd.to_datetime(date_series, errors='coerce')

#         # If numeric epoch seconds as strings or numbers
#         try:
#             numeric = pd.to_numeric(date_series, errors='coerce')
#             if numeric.notna().sum() > 0 and (numeric.max() > 1e9 or numeric.min() > 1):
#                 # likely unix seconds or ms (very noisy heuristic)
#                 # try seconds first then ms
#                 parsed = pd.to_datetime(numeric, unit='s', errors='coerce')
#                 if parsed.notna().sum() / len(parsed) > 0.8:
#                     return parsed
#                 parsed = pd.to_datetime(numeric, unit='ms', errors='coerce')
#                 if parsed.notna().sum() / len(parsed) > 0.8:
#                     return parsed
#         except Exception:
#             pass

#         formats_to_try = [
#             '%Y-%m-%d %H:%M:%S',
#             '%Y-%m-%dT%H:%M:%S',
#             '%m/%d/%Y %I:%M:%S %p',
#             '%m/%d/%Y %H:%M:%S',
#             '%d/%m/%Y %H:%M:%S',
#             '%Y-%m-%d %H:%M:%S.%f',
#             '%Y-%m-%dT%H:%M:%S.%f',
#             '%m/%d/%Y %H:%M:%S.%f',
#         ]
#         for fmt in formats_to_try:
#             try:
#                 parsed = pd.to_datetime(date_series, format=fmt, errors='coerce')
#                 if parsed.notna().sum() / max(1, len(parsed)) > 0.8:
#                     return parsed
#             except Exception:
#                 continue

#         # Last resort: pandas auto-parse (may warn but will generally work)
#         return pd.to_datetime(date_series, errors='coerce')

#     @staticmethod
#     def time_to_seconds(time_str: Any) -> float:
#         """Convert time strings like '1:39.496', '46:41.553', '1:54.168', 'MM:SS' or numeric seconds."""
#         if pd.isna(time_str):
#             return np.nan
#         s = str(time_str).strip()
#         if s == '' or s.upper() in {'-', 'NULL', 'NONE'}:
#             return np.nan

#         # if already numeric
#         try:
#             return float(s)
#         except Exception:
#             pass

#         # Remove possible + sign used in gaps
#         s = s.lstrip('+')

#         # Patterns:
#         # H:MM:SS.sss, MM:SS.sss, M:SS.sss, MM:SS
#         parts = s.split(':')
#         try:
#             if len(parts) == 1:
#                 # decimal seconds
#                 return float(parts[0])
#             if len(parts) == 2:
#                 # MM:SS.sss or MM:SS
#                 minutes = float(parts[0])
#                 seconds = float(parts[1])
#                 return minutes * 60.0 + seconds
#             if len(parts) == 3:
#                 hours = float(parts[0])
#                 minutes = float(parts[1])
#                 seconds = float(parts[2])
#                 return hours * 3600.0 + minutes * 60.0 + seconds
#         except Exception:
#             return np.nan

#         return np.nan

#     @staticmethod
#     def gap_to_seconds(gap_str: Any) -> float:
#         """Convert gap strings like '+0.234', '+0:00.234', '0.234' -> seconds"""
#         if pd.isna(gap_str):
#             return np.nan
#         s = str(gap_str).strip()
#         if s == '' or s.upper() in {'-', 'NULL', 'NONE'}:
#             return np.nan
#         # remove leading +
#         s = s.lstrip('+')
#         # if it's a time-like string with colon
#         if ':' in s:
#             return DataPreprocessor.time_to_seconds(s)
#         # try numeric directly
#         try:
#             return float(s)
#         except Exception:
#             return np.nan

#     @staticmethod
#     def race_time_to_seconds(time_str: Any) -> float:
#         """Total race time format often '46:41.553' -> 46*60 + 41.553"""
#         if pd.isna(time_str):
#             return np.nan
#         s = str(time_str).strip()
#         if s == '' or s.upper() in {'-', 'NULL', 'NONE'}:
#             return np.nan
#         return DataPreprocessor.time_to_seconds(s)

#     @staticmethod
#     def merge_session_data(processed_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
#         """
#         Return dictionary with keys: lap_data, race_data, weather_data, telemetry_data.
#         Missing entries will be empty DataFrames.
#         """
#         out = {}
#         out['lap_data'] = processed_data.get('lap_data', pd.DataFrame()).copy()
#         out['race_data'] = processed_data.get('race_data', pd.DataFrame()).copy()
#         out['weather_data'] = processed_data.get('weather_data', pd.DataFrame()).copy()
#         out['telemetry_data'] = processed_data.get('telemetry_data', pd.DataFrame()).copy()
#         return out





























# import pandas as pd
# import numpy as np
# from typing import Dict, Any
# import re

# class DataPreprocessor:
#     @staticmethod
#     def preprocess_lap_data(lap_df: pd.DataFrame) -> pd.DataFrame:
#         """Clean and prepare lap telemetry data for multiple data formats"""
#         if lap_df is None or lap_df.empty:
#             print("⚠️ Lap data is None or empty, returning empty DataFrame")
#             return pd.DataFrame()
            
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
#         if race_df is None:
#             print("⚠️ Race data is None, returning empty DataFrame")
#             return pd.DataFrame()
            
#         if race_df.empty:
#             print("⚠️ Race data is empty, returning empty DataFrame")
#             return pd.DataFrame()
            
#         df = race_df.copy()
        
#         # Standardize column names
#         df = DataPreprocessor._standardize_column_names(df)
        
#         # Handle position columns with proper validation
#         pos_cols = ['POSITION', 'POS', 'PIC']
#         position_found = False
        
#         for col in pos_cols:
#             if col in df.columns and df[col] is not None:
#                 try:
#                     if not df[col].empty:
#                         df['POSITION'] = pd.to_numeric(df[col], errors='coerce')
#                         position_found = True
#                         print(f"✅ Using position column: {col}")
#                         break
#                 except Exception as e:
#                     print(f"❌ Error processing position column {col}: {e}")
#                     continue
        
#         # Create default positions if no valid position column found
#         if not position_found:
#             print("⚠️ No valid position column found, creating default positions")
#             if len(df) > 0:
#                 df['POSITION'] = range(1, len(df) + 1)
#             else:
#                 df['POSITION'] = pd.Series(dtype='float64')
        
#         # Convert total race time
#         if 'TOTAL_TIME' in df.columns:
#             df['TOTAL_TIME_SECONDS'] = df['TOTAL_TIME'].apply(DataPreprocessor.race_time_to_seconds)
        
#         # Convert gap strings
#         gap_cols = ['GAP_FIRST', 'GAP_PREVIOUS']
#         for col in gap_cols:
#             if col in df.columns and df[col] is not None:
#                 try:
#                     df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.gap_to_seconds)
#                 except Exception as e:
#                     print(f"❌ Error processing gap column {col}: {e}")
#                     df[f"{col}_SECONDS"] = 0.0
        
#         # Handle best lap data
#         if 'BEST_LAP_TIME' in df.columns:
#             df['BEST_LAP_SECONDS'] = df['BEST_LAP_TIME'].apply(DataPreprocessor.time_to_seconds)
#         elif 'FL_TIME' in df.columns:
#             df['BEST_LAP_SECONDS'] = df['FL_TIME'].apply(DataPreprocessor.time_to_seconds)
        
#         # Clean numeric columns with error handling
#         numeric_cols = ['NUMBER', 'LAPS', 'FL_LAPNUM', 'BEST_LAP_NUM']
#         for col in numeric_cols:
#             if col in df.columns and df[col] is not None:
#                 try:
#                     df[col] = pd.to_numeric(df[col], errors='coerce')
#                 except Exception as e:
#                     print(f"❌ Error processing numeric column {col}: {e}")
#                     df[col] = np.nan
        
#         print(f"✅ Processed race data with {len(df)} entries")
#         return df

#     @staticmethod
#     def preprocess_weather_data(weather_df: pd.DataFrame) -> pd.DataFrame:
#         """Clean and prepare weather data"""
#         if weather_df is None or weather_df.empty:
#             print("⚠️ Weather data is None or empty, returning empty DataFrame")
#             return pd.DataFrame()
            
#         df = weather_df.copy()
        
#         # Standardize column names
#         df = DataPreprocessor._standardize_column_names(df)
        
#         # Convert timestamps with explicit format handling
#         timestamp_cols = ['TIME_UTC_STR', 'TIME_UTC_SECONDS', 'timestamp']
#         for col in timestamp_cols:
#             if col in df.columns:
#                 if col == 'TIME_UTC_SECONDS':
#                     df['timestamp'] = pd.to_datetime(df[col], unit='s', errors='coerce')
#                 else:
#                     # Try common datetime formats to avoid warnings
#                     df['timestamp'] = DataPreprocessor._safe_datetime_parse(df[col])
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
#         if telemetry_df is None or telemetry_df.empty:
#             print("⚠️ Telemetry data is None or empty, returning empty DataFrame")
#             return pd.DataFrame()
            
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
        
#         # Convert timestamps with safe parsing
#         if 'timestamp' in df.columns:
#             df['timestamp'] = DataPreprocessor._safe_datetime_parse(df['timestamp'])
        
#         return df

#     @staticmethod
#     def _safe_datetime_parse(date_series: pd.Series) -> pd.Series:
#         """Safely parse datetime series with multiple format attempts to avoid warnings"""
#         if date_series.empty:
#             return date_series
            
#         # Try common datetime formats in order of likelihood
#         formats_to_try = [
#             '%Y-%m-%d %H:%M:%S',     # 2023-01-15 14:30:25
#             '%Y-%m-%dT%H:%M:%S',     # 2023-01-15T14:30:25
#             '%m/%d/%Y %H:%M:%S',     # 01/15/2023 14:30:25
#             '%d/%m/%Y %H:%M:%S',     # 15/01/2023 14:30:25
#             '%Y-%m-%d %H:%M:%S.%f',  # 2023-01-15 14:30:25.123
#             '%Y-%m-%dT%H:%M:%S.%f',  # 2023-01-15T14:30:25.123
#         ]
        
#         for fmt in formats_to_try:
#             try:
#                 parsed = pd.to_datetime(date_series, format=fmt, errors='coerce')
#                 # Check if we successfully parsed most values
#                 if parsed.notna().sum() > len(parsed) * 0.8:  # 80% success rate
#                     return parsed
#             except:
#                 continue
        
#         # Fallback to pandas automatic detection (will show warning but works)
#         return pd.to_datetime(date_series, errors='coerce')

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
# from typing import Dict, Any
# import re

# class DataPreprocessor:
#     @staticmethod
#     def preprocess_lap_data(lap_df: pd.DataFrame) -> pd.DataFrame:
#         """Clean and prepare lap telemetry data for multiple data formats"""
#         if lap_df is None or lap_df.empty:
#             print("⚠️ Lap data is None or empty, returning empty DataFrame")
#             return pd.DataFrame()
            
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
#         # FIXED: Added proper None and empty DataFrame handling
#         if race_df is None:
#             print("⚠️ Race data is None, returning empty DataFrame")
#             return pd.DataFrame()
            
#         if race_df.empty:
#             print("⚠️ Race data is empty, returning empty DataFrame")
#             return pd.DataFrame()
            
#         df = race_df.copy()
        
#         # Standardize column names
#         df = DataPreprocessor._standardize_column_names(df)
        
#         # FIXED: Handle position columns with proper validation
#         pos_cols = ['POSITION', 'POS', 'PIC']
#         position_found = False
        
#         for col in pos_cols:
#             if col in df.columns and df[col] is not None:
#                 try:
#                     # FIXED: Check if the column has valid data before processing
#                     if not df[col].empty:
#                         df['POSITION'] = pd.to_numeric(df[col], errors='coerce')
#                         position_found = True
#                         print(f"✅ Using position column: {col}")
#                         break
#                 except Exception as e:
#                     print(f"❌ Error processing position column {col}: {e}")
#                     continue
        
#         # FIXED: Create default positions if no valid position column found
#         if not position_found:
#             print("⚠️ No valid position column found, creating default positions")
#             if len(df) > 0:
#                 df['POSITION'] = range(1, len(df) + 1)
#             else:
#                 df['POSITION'] = pd.Series(dtype='float64')
        
#         # Convert total race time
#         if 'TOTAL_TIME' in df.columns:
#             df['TOTAL_TIME_SECONDS'] = df['TOTAL_TIME'].apply(DataPreprocessor.race_time_to_seconds)
        
#         # Convert gap strings
#         gap_cols = ['GAP_FIRST', 'GAP_PREVIOUS']
#         for col in gap_cols:
#             if col in df.columns and df[col] is not None:
#                 try:
#                     df[f"{col}_SECONDS"] = df[col].apply(DataPreprocessor.gap_to_seconds)
#                 except Exception as e:
#                     print(f"❌ Error processing gap column {col}: {e}")
#                     df[f"{col}_SECONDS"] = 0.0
        
#         # Handle best lap data
#         if 'BEST_LAP_TIME' in df.columns:
#             df['BEST_LAP_SECONDS'] = df['BEST_LAP_TIME'].apply(DataPreprocessor.time_to_seconds)
#         elif 'FL_TIME' in df.columns:
#             df['BEST_LAP_SECONDS'] = df['FL_TIME'].apply(DataPreprocessor.time_to_seconds)
        
#         # Clean numeric columns with error handling
#         numeric_cols = ['NUMBER', 'LAPS', 'FL_LAPNUM', 'BEST_LAP_NUM']
#         for col in numeric_cols:
#             if col in df.columns and df[col] is not None:
#                 try:
#                     df[col] = pd.to_numeric(df[col], errors='coerce')
#                 except Exception as e:
#                     print(f"❌ Error processing numeric column {col}: {e}")
#                     df[col] = np.nan
        
#         print(f"✅ Processed race data with {len(df)} entries")
#         return df

#     @staticmethod
#     def preprocess_weather_data(weather_df: pd.DataFrame) -> pd.DataFrame:
#         """Clean and prepare weather data"""
#         if weather_df is None or weather_df.empty:
#             print("⚠️ Weather data is None or empty, returning empty DataFrame")
#             return pd.DataFrame()
            
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
#         if telemetry_df is None or telemetry_df.empty:
#             print("⚠️ Telemetry data is None or empty, returning empty DataFrame")
#             return pd.DataFrame()
            
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