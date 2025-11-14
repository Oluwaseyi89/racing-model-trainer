import pandas as pd
import numpy as np
from typing import Dict
from scipy import stats


class FeatureEngineer:
    """
    Feature engineering for racing analytics.
    Safe for inconsistent telemetry formats, missing laps,
    partial weather data, and incomplete race results.
    """

    # ------------------------------------------------------------
    # TIRE FEATURES
    # ------------------------------------------------------------
    @staticmethod
    def engineer_tire_features(lap_data: pd.DataFrame,
                               telemetry_data: pd.DataFrame) -> pd.DataFrame:

        if lap_data.empty:
            return lap_data.copy()

        df = lap_data.copy()

        # Ensure core fields exist
        if "NUMBER" not in df.columns:
            return df

        if "LAP_NUMBER" not in df.columns:
            df["LAP_NUMBER"] = df.groupby("NUMBER").cumcount() + 1

        # -----------------------------
        # Per-car tire degradation
        # -----------------------------
        try:
            for car_number in df["NUMBER"].dropna().unique():
                car_mask = df["NUMBER"] == car_number
                car_laps = df[car_mask].sort_values("LAP_NUMBER")

                if len(car_laps) < 5:
                    continue

                # LAP TIME DEGRADATION RATE
                if "LAP_TIME_SECONDS" in car_laps.columns:
                    lap_times = car_laps["LAP_TIME_SECONDS"].values
                    lap_numbers = car_laps["LAP_NUMBER"].values

                    if len(lap_times) >= 8:
                        mask = (lap_numbers >= 5) & (lap_numbers <= 15)
                        if mask.sum() >= 5:
                            slope, _, r_value, _, _ = stats.linregress(
                                lap_numbers[mask], lap_times[mask]
                            )
                            df.loc[car_mask, "TIRE_DEGRADATION_RATE"] = (
                                slope if r_value**2 > 0.5 else 0.0
                            )

                # SECTOR DEGRADATION
                for sector in ["S1_SECONDS", "S2_SECONDS", "S3_SECONDS"]:
                    if sector in car_laps.columns:
                        sec_vals = car_laps[sector].values
                        if len(sec_vals) >= 5 and not np.all(np.isnan(sec_vals)):
                            try:
                                slope = np.polyfit(
                                    car_laps["LAP_NUMBER"].values[:len(sec_vals)],
                                    sec_vals,
                                    1
                                )[0]
                                df.loc[car_mask, f"{sector}_DEGRADATION"] = slope
                            except:
                                df.loc[car_mask, f"{sector}_DEGRADATION"] = 0.0

                # CONSISTENCY
                if "LAP_TIME_SECONDS" in car_laps.columns:
                    df.loc[car_mask, "PERFORMANCE_CONSISTENCY"] = \
                        np.nanstd(car_laps["LAP_TIME_SECONDS"])

                # NON-LINEAR TIRE AGE
                df.loc[car_mask, "TIRE_AGE_NONLINEAR"] = \
                    np.log1p(car_laps["LAP_NUMBER"]) * 0.5

        except Exception as e:
            print(f"⚠️ Tire feature engineering failed: {e}")

        # Add telemetry-based features
        if not telemetry_data.empty:
            df = FeatureEngineer._add_telemetry_tire_features(df, telemetry_data)

        return df

    @staticmethod
    def _add_telemetry_tire_features(lap_df: pd.DataFrame,
                                     telemetry_df: pd.DataFrame) -> pd.DataFrame:

        df = lap_df.copy()

        # Required telemetry structure
        if not {"vehicle_number", "lap"}.issubset(telemetry_df.columns):
            return df

        telemetry_features = []

        try:
            for (car_number, lap), lap_tel in telemetry_df.groupby(
                    ["vehicle_number", "lap"]):

                if len(lap_tel) < 10:
                    continue

                telemetry_features.append({
                    "NUMBER": car_number,
                    "LAP_NUMBER": lap,
                    "LATERAL_G_MEAN":
                        lap_tel.get("LATERAL_ACCEL", pd.Series([0])).abs().mean(),
                    "LATERAL_G_VARIANCE":
                        lap_tel.get("LATERAL_ACCEL", pd.Series([0])).abs().std(),
                    "BRAKE_INTENSITY":
                        lap_tel.get("TOTAL_BRAKE_PRESSURE", pd.Series([0])).mean(),
                    "STEERING_ACTIVITY":
                        lap_tel.get("STEERING_ANGLE", pd.Series([0])).diff().abs().sum()
                })

            if telemetry_features:
                tdf = pd.DataFrame(telemetry_features)
                df = df.merge(tdf, on=["NUMBER", "LAP_NUMBER"], how="left")

        except Exception as e:
            print(f"⚠️ Telemetry tire feature merge failed: {e}")

        return df

    # ------------------------------------------------------------
    # FUEL FEATURES
    # ------------------------------------------------------------
    @staticmethod
    def engineer_fuel_features(lap_data: pd.DataFrame,
                               telemetry_data: pd.DataFrame) -> pd.DataFrame:

        if lap_data.empty:
            return lap_data.copy()

        df = lap_data.copy()

        try:
            if "LAP_TIME_SECONDS" in df.columns:
                df["FUEL_EFFICIENCY_EST"] = 1 / (df["LAP_TIME_SECONDS"] + 0.1)

            # Telemetry-based fuel usage
            if not telemetry_data.empty and "THROTTLE_POSITION" in telemetry_data.columns:
                throttle_stats = telemetry_data.groupby(
                    ["vehicle_number", "lap"]
                )["THROTTLE_POSITION"].agg(["mean", "std"]).reset_index()

                throttle_stats.columns = [
                    "NUMBER", "LAP_NUMBER",
                    "THROTTLE_MEAN", "THROTTLE_STD"
                ]

                df = df.merge(throttle_stats, on=["NUMBER", "LAP_NUMBER"], how="left")

        except Exception as e:
            print(f"⚠️ Fuel engineering failed: {e}")

        return df

    # ------------------------------------------------------------
    # TRACK FEATURES
    # ------------------------------------------------------------
    @staticmethod
    def engineer_track_features(track_name: str,
                                lap_data: pd.DataFrame) -> pd.DataFrame:

        if lap_data.empty:
            return lap_data.copy()

        df = lap_data.copy()

        wear_map = {
            "sebring": 0.9, "barber": 0.85, "sonoma": 0.8,
            "road-america": 0.7, "vir": 0.75, "cota": 0.6,
            "indianapolis": 0.5
        }

        try:
            df["TRACK_WEAR_FACTOR"] = wear_map.get(track_name.lower(), 0.7)

            if "KPH" in df.columns:
                mean_speed = df["KPH"].mean()
                if mean_speed > 0:
                    df["OVERTAKING_POTENTIAL"] = min(
                        1.0, (df["KPH"].var() / mean_speed) * 10
                    )
                else:
                    df["OVERTAKING_POTENTIAL"] = 0.1

        except Exception as e:
            print(f"⚠️ Track feature engineering failed: {e}")

        return df

    # ------------------------------------------------------------
    # WEATHER FEATURES
    # ------------------------------------------------------------
    @staticmethod
    def engineer_weather_features(weather_data: pd.DataFrame,
                                  lap_data: pd.DataFrame) -> pd.DataFrame:

        if lap_data.empty:
            return lap_data.copy()

        df = lap_data.copy()

        try:
            if not weather_data.empty:

                if "AIR_TEMP" in weather_data.columns:
                    df["TEMP_IMPACT"] = (weather_data["AIR_TEMP"].mean() - 25.0) * 0.03

                if "TRACK_TEMP" in weather_data.columns:
                    df["TRACK_TEMP_IMPACT"] = \
                        (weather_data["TRACK_TEMP"].mean() - 35.0) * 0.02

                if "RAIN" in weather_data.columns:
                    df["RAIN_IMPACT"] = weather_data["RAIN"].max() * 1.5

        except Exception as e:
            print(f"⚠️ Weather feature engineering failed: {e}")

        return df

    # ------------------------------------------------------------
    # STRATEGY FEATURES
    # ------------------------------------------------------------
    @staticmethod
    def engineer_strategy_features(race_data: pd.DataFrame,
                                   lap_data: pd.DataFrame) -> pd.DataFrame:

        if lap_data.empty or race_data.empty:
            return pd.DataFrame()

        strategy_rows = []

        try:
            for car_number in race_data["NUMBER"].dropna().unique():

                car_race = race_data[race_data["NUMBER"] == car_number]
                if car_race.empty:
                    continue

                car_laps = lap_data[lap_data["NUMBER"] == car_number]
                if len(car_laps) < 3:
                    continue

                pos = car_race.iloc[0].get("POSITION", np.nan)

                strategy_rows.append({
                    "car_number": car_number,
                    "position": pos,
                    "total_laps": len(car_laps),
                    "needs_strategy_change": 1 if pos and pos > 10 else 0
                })

        except Exception as e:
            print(f"⚠️ Strategy engineering failed: {e}")

        return pd.DataFrame(strategy_rows)

    # ------------------------------------------------------------
    # MASTER COMPOSITE FEATURE ENGINEERING
    # ------------------------------------------------------------
    @staticmethod
    def create_composite_features(processed_data: Dict) -> Dict:

        enhanced = {}

        for track_name, data in processed_data.items():

            try:
                lap = data.get("lap_data", pd.DataFrame())
                race = data.get("race_data", pd.DataFrame())
                weather = data.get("weather_data", pd.DataFrame())
                telemetry = data.get("telemetry_data", pd.DataFrame())

                if lap.empty:
                    enhanced[track_name] = data
                    continue

                lap = FeatureEngineer.engineer_tire_features(lap, telemetry)
                lap = FeatureEngineer.engineer_fuel_features(lap, telemetry)
                lap = FeatureEngineer.engineer_track_features(track_name, lap)
                lap = FeatureEngineer.engineer_weather_features(weather, lap)

                strategy = FeatureEngineer.engineer_strategy_features(race, lap)

                enhanced[track_name] = {
                    "lap_data": lap,
                    "race_data": race,
                    "weather_data": weather,
                    "telemetry_data": telemetry,
                    "strategy_features": strategy
                }

            except Exception as e:
                print(f"⚠️ Feature creation failed for {track_name}: {e}")
                enhanced[track_name] = data

        return enhanced

























# import pandas as pd
# import numpy as np
# from typing import Dict, List, Tuple
# from scipy import stats

# class FeatureEngineer:
#     """Engineer advanced features from processed racing data including telemetry"""
    
#     @staticmethod
#     def engineer_tire_features(lap_data: pd.DataFrame, telemetry_data: pd.DataFrame) -> pd.DataFrame:
#         """Create advanced tire degradation features with safe column handling"""
#         if lap_data.empty:
#             return lap_data
            
#         df = lap_data.copy()
        
#         # Safe column checks
#         if 'NUMBER' not in df.columns:
#             return df
            
#         # Create LAP_NUMBER if missing
#         if 'LAP_NUMBER' not in df.columns:
#             df['LAP_NUMBER'] = df.groupby('NUMBER').cumcount() + 1
        
#         try:
#             for car_number in df['NUMBER'].unique():
#                 car_mask = df['NUMBER'] == car_number
#                 car_laps = df[car_mask].sort_values('LAP_NUMBER')
                
#                 if len(car_laps) < 5:
#                     continue
                    
#                 # Rolling performance degradation
#                 if 'LAP_TIME_SECONDS' in car_laps.columns:
#                     lap_times = car_laps['LAP_TIME_SECONDS'].values
#                     lap_numbers = car_laps['LAP_NUMBER'].values
                    
#                     # Calculate degradation rate using linear regression
#                     if len(lap_times) >= 8:
#                         try:
#                             mask = (lap_numbers >= 5) & (lap_numbers <= 15)
#                             if mask.sum() >= 5:
#                                 deg_laps = lap_numbers[mask]
#                                 deg_times = lap_times[mask]
#                                 slope, _, r_value, _, _ = stats.linregress(deg_laps, deg_times)
#                                 df.loc[car_mask, 'TIRE_DEGRADATION_RATE'] = slope if r_value**2 > 0.5 else 0.0
#                         except:
#                             df.loc[car_mask, 'TIRE_DEGRADATION_RATE'] = 0.0
                
#                 # Sector-specific degradation patterns
#                 for sector in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']:
#                     if sector in car_laps.columns:
#                         sector_times = car_laps[sector].values
#                         if len(sector_times) >= 5 and not np.all(np.isnan(sector_times)):
#                             try:
#                                 sector_slope = np.polyfit(lap_numbers[:len(sector_times)], 
#                                                         sector_times, 1)[0]
#                                 df.loc[car_mask, f'{sector}_DEGRADATION'] = sector_slope
#                             except:
#                                 df.loc[car_mask, f'{sector}_DEGRADATION'] = 0.0
                
#                 # Performance consistency
#                 if 'LAP_TIME_SECONDS' in car_laps.columns and len(lap_times) >= 10:
#                     df.loc[car_mask, 'PERFORMANCE_CONSISTENCY'] = np.std(lap_times)
                
#                 # Tire age with non-linear effects
#                 df.loc[car_mask, 'TIRE_AGE_NONLINEAR'] = np.log1p(car_laps['LAP_NUMBER']) * 0.5
            
#         except Exception as e:
#             print(f"⚠️ Tire feature engineering failed: {e}")
        
#         # Add telemetry-based tire features if available
#         if not telemetry_data.empty:
#             df = FeatureEngineer._add_telemetry_tire_features(df, telemetry_data)
        
#         return df
    
#     @staticmethod
#     def _add_telemetry_tire_features(lap_data: pd.DataFrame, telemetry_data: pd.DataFrame) -> pd.DataFrame:
#         """Add tire-related features from telemetry data with safe column handling"""
#         df = lap_data.copy()
        
#         # Check required telemetry columns
#         required_telemetry_cols = ['vehicle_number', 'lap']
#         if not all(col in telemetry_data.columns for col in required_telemetry_cols):
#             return df
            
#         telemetry_features = []
#         try:
#             for (car_number, lap_num), lap_telemetry in telemetry_data.groupby(['vehicle_number', 'lap']):
#                 if len(lap_telemetry) < 10:
#                     continue
                    
#                 # Calculate driving style metrics with safe column checks
#                 lateral_g_mean = lap_telemetry['LATERAL_ACCEL'].abs().mean() if 'LATERAL_ACCEL' in lap_telemetry.columns else 0
#                 lateral_g_std = lap_telemetry['LATERAL_ACCEL'].abs().std() if 'LATERAL_ACCEL' in lap_telemetry.columns else 0
#                 brake_pressure_mean = lap_telemetry['TOTAL_BRAKE_PRESSURE'].mean() if 'TOTAL_BRAKE_PRESSURE' in lap_telemetry.columns else 0
#                 steering_activity = lap_telemetry['STEERING_ANGLE'].diff().abs().sum() if 'STEERING_ANGLE' in lap_telemetry.columns else 0
                
#                 telemetry_features.append({
#                     'NUMBER': car_number,
#                     'LAP_NUMBER': lap_num,
#                     'LATERAL_G_MEAN': lateral_g_mean,
#                     'LATERAL_G_VARIANCE': lateral_g_std,
#                     'BRAKE_INTENSITY': brake_pressure_mean,
#                     'STEERING_ACTIVITY': steering_activity
#                 })
            
#             if telemetry_features:
#                 telemetry_df = pd.DataFrame(telemetry_features)
#                 df = df.merge(telemetry_df, on=['NUMBER', 'LAP_NUMBER'], how='left')
                
#         except Exception as e:
#             print(f"⚠️ Telemetry tire features failed: {e}")
        
#         return df

#     @staticmethod
#     def engineer_fuel_features(lap_data: pd.DataFrame, telemetry_data: pd.DataFrame) -> pd.DataFrame:
#         """Create fuel consumption features with safe column handling"""
#         if lap_data.empty:
#             return lap_data
            
#         df = lap_data.copy()
        
#         try:
#             # Basic fuel estimation if we have lap data
#             if 'LAP_TIME_SECONDS' in df.columns:
#                 df['FUEL_EFFICIENCY_EST'] = 1 / (df['LAP_TIME_SECONDS'] + 0.1)
                
#             # Add telemetry-based fuel features if available
#             if not telemetry_data.empty:
#                 df = FeatureEngineer._add_telemetry_fuel_features(df, telemetry_data)
                
#         except Exception as e:
#             print(f"⚠️ Fuel feature engineering failed: {e}")
        
#         return df

#     @staticmethod
#     def _add_telemetry_fuel_features(lap_data: pd.DataFrame, telemetry_data: pd.DataFrame) -> pd.DataFrame:
#         """Add fuel-related features from telemetry"""
#         df = lap_data.copy()
        
#         try:
#             # Simple throttle-based fuel estimation
#             if 'THROTTLE_POSITION' in telemetry_data.columns:
#                 throttle_stats = telemetry_data.groupby(['vehicle_number', 'lap'])['THROTTLE_POSITION'].agg(['mean', 'std']).reset_index()
#                 throttle_stats.columns = ['NUMBER', 'LAP_NUMBER', 'THROTTLE_MEAN', 'THROTTLE_STD']
#                 df = df.merge(throttle_stats, on=['NUMBER', 'LAP_NUMBER'], how='left')
#         except Exception as e:
#             print(f"⚠️ Telemetry fuel features failed: {e}")
        
#         return df

#     @staticmethod
#     def engineer_track_features(track_name: str, lap_data: pd.DataFrame) -> pd.DataFrame:
#         """Create track-specific features with safe column handling"""
#         if lap_data.empty:
#             return lap_data
            
#         df = lap_data.copy()
        
#         try:
#             # Track-specific wear factors
#             track_wear_map = {
#                 'sebring': 0.9, 'barber': 0.85, 'sonoma': 0.8, 
#                 'road-america': 0.7, 'vir': 0.75, 'cota': 0.6, 
#                 'indianapolis': 0.5
#             }
            
#             df['TRACK_WEAR_FACTOR'] = track_wear_map.get(track_name.lower(), 0.7)
            
#             # Overtaking potential
#             if 'KPH' in df.columns:
#                 speed_variance = df['KPH'].var() / df['KPH'].mean() if df['KPH'].mean() > 0 else 0.1
#                 df['OVERTAKING_POTENTIAL'] = min(1.0, speed_variance * 10)
                
#         except Exception as e:
#             print(f"⚠️ Track feature engineering failed: {e}")
        
#         return df

#     @staticmethod
#     def engineer_weather_features(weather_data: pd.DataFrame, lap_data: pd.DataFrame) -> pd.DataFrame:
#         """Create weather impact features with safe column handling"""
#         if weather_data.empty or lap_data.empty:
#             return lap_data.copy()
            
#         df = lap_data.copy()
        
#         try:
#             # Simple weather impact estimation
#             if 'AIR_TEMP' in weather_data.columns:
#                 temp_avg = weather_data['AIR_TEMP'].mean()
#                 optimal_temp = 25.0
#                 df['TEMP_IMPACT'] = (temp_avg - optimal_temp) * 0.03
                
#             if 'TRACK_TEMP' in weather_data.columns:
#                 track_temp_avg = weather_data['TRACK_TEMP'].mean()
#                 optimal_track_temp = 35.0
#                 df['TRACK_TEMP_IMPACT'] = (track_temp_avg - optimal_track_temp) * 0.02
                
#             if 'RAIN' in weather_data.columns:
#                 rain_max = weather_data['RAIN'].max()
#                 df['RAIN_IMPACT'] = rain_max * 1.5
                
#         except Exception as e:
#             print(f"⚠️ Weather feature engineering failed: {e}")
        
#         return df

#     @staticmethod
#     def engineer_strategy_features(race_data: pd.DataFrame, lap_data: pd.DataFrame) -> pd.DataFrame:
#         """Create strategy features with safe column handling"""
#         strategy_features = []
        
#         if race_data.empty or lap_data.empty:
#             return pd.DataFrame()
        
#         try:
#             for car_number in race_data['NUMBER'].unique():
#                 car_race = race_data[race_data['NUMBER'] == car_number]
#                 if car_race.empty:
#                     continue
                    
#                 car_race = car_race.iloc[0]
#                 car_laps = lap_data[lap_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
                
#                 if len(car_laps) < 5:
#                     continue
                
#                 # Basic strategy metrics
#                 position = car_race.get('POSITION', 1)
                
#                 strategy_features.append({
#                     'car_number': car_number,
#                     'position': position,
#                     'total_laps': len(car_laps),
#                     'needs_strategy_change': 1 if position > 10 else 0
#                 })
            
#         except Exception as e:
#             print(f"⚠️ Strategy feature engineering failed: {e}")
        
#         return pd.DataFrame(strategy_features) if strategy_features else pd.DataFrame()

#     @staticmethod
#     def create_composite_features(processed_data: Dict) -> Dict:
#         """Create all composite features with comprehensive error handling"""
#         enhanced_data = {}
        
#         for track_name, data in processed_data.items():
#             try:
#                 lap_data = data.get('lap_data', pd.DataFrame())
#                 race_data = data.get('race_data', pd.DataFrame())
#                 weather_data = data.get('weather_data', pd.DataFrame())
#                 telemetry_data = data.get('telemetry_data', pd.DataFrame())
                
#                 # Skip if no lap data
#                 if lap_data.empty:
#                     enhanced_data[track_name] = data
#                     continue
                
#                 # Apply all feature engineering with error handling
#                 lap_data = FeatureEngineer.engineer_tire_features(lap_data, telemetry_data)
#                 lap_data = FeatureEngineer.engineer_fuel_features(lap_data, telemetry_data)
#                 lap_data = FeatureEngineer.engineer_track_features(track_name, lap_data)
#                 lap_data = FeatureEngineer.engineer_weather_features(weather_data, lap_data)
                
#                 strategy_features = FeatureEngineer.engineer_strategy_features(race_data, lap_data)
                
#                 enhanced_data[track_name] = {
#                     'lap_data': lap_data,
#                     'race_data': race_data,
#                     'weather_data': weather_data,
#                     'telemetry_data': telemetry_data,
#                     'strategy_features': strategy_features
#                 }
                
#             except Exception as e:
#                 print(f"⚠️ Feature engineering failed for {track_name}: {e}")
#                 enhanced_data[track_name] = data  # Return original data on failure
        
#         return enhanced_data

#     @staticmethod
#     def _parse_gap(gap_str: str) -> float:
#         """Parse gap string to seconds"""
#         if pd.isna(gap_str) or gap_str in ['-', '']:
#             return 0.0
#         try:
#             gap_str = str(gap_str).replace('+', '').strip()
#             return float(gap_str)
#         except:
#             return 0.0


















# import pandas as pd
# import numpy as np
# from typing import Dict, List, Tuple
# from scipy import stats

# class FeatureEngineer:
#     """Engineer advanced features from processed racing data including telemetry"""
    
#     @staticmethod
#     def engineer_tire_features(lap_data: pd.DataFrame, telemetry_data: pd.DataFrame) -> pd.DataFrame:
#         """Create advanced tire degradation features with safe column handling"""
#         if lap_data.empty:
#             return lap_data
            
#         df = lap_data.copy()
        
#         # Safe column checks
#         if 'NUMBER' not in df.columns:
#             return df
            
#         # Create LAP_NUMBER if missing
#         if 'LAP_NUMBER' not in df.columns:
#             df['LAP_NUMBER'] = df.groupby('NUMBER').cumcount() + 1
        
#         try:
#             for car_number in df['NUMBER'].unique():
#                 car_mask = df['NUMBER'] == car_number
#                 car_laps = df[car_mask].sort_values('LAP_NUMBER')
                
#                 if len(car_laps) < 5:
#                     continue
                    
#                 # Rolling performance degradation
#                 if 'LAP_TIME_SECONDS' in car_laps.columns:
#                     lap_times = car_laps['LAP_TIME_SECONDS'].values
#                     lap_numbers = car_laps['LAP_NUMBER'].values
                    
#                     # Calculate degradation rate using linear regression
#                     if len(lap_times) >= 8:
#                         try:
#                             mask = (lap_numbers >= 5) & (lap_numbers <= 15)
#                             if mask.sum() >= 5:
#                                 deg_laps = lap_numbers[mask]
#                                 deg_times = lap_times[mask]
#                                 slope, _, r_value, _, _ = stats.linregress(deg_laps, deg_times)
#                                 df.loc[car_mask, 'TIRE_DEGRADATION_RATE'] = slope if r_value**2 > 0.5 else 0.0
#                         except:
#                             df.loc[car_mask, 'TIRE_DEGRADATION_RATE'] = 0.0
                
#                 # Sector-specific degradation patterns
#                 for sector in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']:
#                     if sector in car_laps.columns:
#                         sector_times = car_laps[sector].values
#                         if len(sector_times) >= 5 and not np.all(np.isnan(sector_times)):
#                             try:
#                                 sector_slope = np.polyfit(lap_numbers[:len(sector_times)], 
#                                                         sector_times, 1)[0]
#                                 df.loc[car_mask, f'{sector}_DEGRADATION'] = sector_slope
#                             except:
#                                 df.loc[car_mask, f'{sector}_DEGRADATION'] = 0.0
                
#                 # Performance consistency
#                 if 'LAP_TIME_SECONDS' in car_laps.columns and len(lap_times) >= 10:
#                     df.loc[car_mask, 'PERFORMANCE_CONSISTENCY'] = np.std(lap_times)
                
#                 # Tire age with non-linear effects
#                 df.loc[car_mask, 'TIRE_AGE_NONLINEAR'] = np.log1p(car_laps['LAP_NUMBER']) * 0.5
            
#         except Exception as e:
#             print(f"⚠️ Tire feature engineering failed: {e}")
        
#         # Add telemetry-based tire features if available
#         if not telemetry_data.empty:
#             df = FeatureEngineer._add_telemetry_tire_features(df, telemetry_data)
        
#         return df
    
#     @staticmethod
#     def _add_telemetry_tire_features(lap_data: pd.DataFrame, telemetry_data: pd.DataFrame) -> pd.DataFrame:
#         """Add tire-related features from telemetry data with safe column handling"""
#         df = lap_data.copy()
        
#         # Check required telemetry columns
#         required_telemetry_cols = ['vehicle_number', 'lap']
#         if not all(col in telemetry_data.columns for col in required_telemetry_cols):
#             return df
            
#         telemetry_features = []
#         try:
#             for (car_number, lap_num), lap_telemetry in telemetry_data.groupby(['vehicle_number', 'lap']):
#                 if len(lap_telemetry) < 10:
#                     continue
                    
#                 # Calculate driving style metrics with safe column checks
#                 lateral_g_mean = lap_telemetry['LATERAL_ACCEL'].abs().mean() if 'LATERAL_ACCEL' in lap_telemetry.columns else 0
#                 lateral_g_std = lap_telemetry['LATERAL_ACCEL'].abs().std() if 'LATERAL_ACCEL' in lap_telemetry.columns else 0
#                 brake_pressure_mean = lap_telemetry['TOTAL_BRAKE_PRESSURE'].mean() if 'TOTAL_BRAKE_PRESSURE' in lap_telemetry.columns else 0
#                 steering_activity = lap_telemetry['STEERING_ANGLE'].diff().abs().sum() if 'STEERING_ANGLE' in lap_telemetry.columns else 0
                
#                 telemetry_features.append({
#                     'NUMBER': car_number,
#                     'LAP_NUMBER': lap_num,
#                     'LATERAL_G_MEAN': lateral_g_mean,
#                     'LATERAL_G_VARIANCE': lateral_g_std,
#                     'BRAKE_INTENSITY': brake_pressure_mean,
#                     'STEERING_ACTIVITY': steering_activity
#                 })
            
#             if telemetry_features:
#                 telemetry_df = pd.DataFrame(telemetry_features)
#                 df = df.merge(telemetry_df, on=['NUMBER', 'LAP_NUMBER'], how='left')
                
#         except Exception as e:
#             print(f"⚠️ Telemetry tire features failed: {e}")
        
#         return df

#     # [Rest of the methods follow the same safe column pattern...]
    
#     @staticmethod
#     def create_composite_features(processed_data: Dict) -> Dict:
#         """Create all composite features with comprehensive error handling"""
#         enhanced_data = {}
        
#         for track_name, data in processed_data.items():
#             try:
#                 lap_data = data.get('lap_data', pd.DataFrame())
#                 race_data = data.get('race_data', pd.DataFrame())
#                 weather_data = data.get('weather_data', pd.DataFrame())
#                 telemetry_data = data.get('telemetry_data', pd.DataFrame())
                
#                 # Skip if no lap data
#                 if lap_data.empty:
#                     enhanced_data[track_name] = data
#                     continue
                
#                 # Apply all feature engineering with error handling
#                 lap_data = FeatureEngineer.engineer_tire_features(lap_data, telemetry_data)
#                 lap_data = FeatureEngineer.engineer_fuel_features(lap_data, telemetry_data)
#                 lap_data = FeatureEngineer.engineer_track_features(track_name, lap_data)
#                 lap_data = FeatureEngineer.engineer_weather_features(weather_data, lap_data)
                
#                 strategy_features = FeatureEngineer.engineer_strategy_features(race_data, lap_data)
                
#                 enhanced_data[track_name] = {
#                     'lap_data': lap_data,
#                     'race_data': race_data,
#                     'weather_data': weather_data,
#                     'telemetry_data': telemetry_data,
#                     'strategy_features': strategy_features
#                 }
                
#             except Exception as e:
#                 print(f"⚠️ Feature engineering failed for {track_name}: {e}")
#                 enhanced_data[track_name] = data  # Return original data on failure
        
#         return enhanced_data

#     @staticmethod
#     def _parse_gap(gap_str: str) -> float:
#         """Parse gap string to seconds"""
#         if pd.isna(gap_str) or gap_str in ['-', '']:
#             return 0.0
#         try:
#             gap_str = str(gap_str).replace('+', '').strip()
#             return float(gap_str)
#         except:
#             return 0.0



















# import pandas as pd
# import numpy as np
# from typing import Dict, List, Tuple
# from scipy import stats

# class FeatureEngineer:
#     """Engineer advanced features from processed racing data including telemetry"""
    
#     @staticmethod
#     def engineer_tire_features(lap_data: pd.DataFrame, telemetry_data: pd.DataFrame) -> pd.DataFrame:
#         """Create advanced tire degradation features using lap data and telemetry"""
#         if lap_data.empty:
#             return lap_data
            
#         df = lap_data.copy()
        
#         # Group by car to calculate individual tire metrics
#         for car_number in df['NUMBER'].unique():
#             car_mask = df['NUMBER'] == car_number
#             car_laps = df[car_mask].sort_values('LAP_NUMBER')
            
#             if len(car_laps) < 5:
#                 continue
                
#             # Rolling performance degradation
#             lap_times = car_laps['LAP_TIME_SECONDS'].values
#             lap_numbers = car_laps['LAP_NUMBER'].values
            
#             # Calculate degradation rate using linear regression
#             if len(lap_times) >= 8:
#                 try:
#                     # Use laps 5-15 for stable degradation analysis
#                     mask = (lap_numbers >= 5) & (lap_numbers <= 15)
#                     if mask.sum() >= 5:
#                         deg_laps = lap_numbers[mask]
#                         deg_times = lap_times[mask]
#                         slope, _, r_value, _, _ = stats.linregress(deg_laps, deg_times)
#                         df.loc[car_mask, 'TIRE_DEGRADATION_RATE'] = slope if r_value**2 > 0.5 else 0.0
#                 except:
#                     df.loc[car_mask, 'TIRE_DEGRADATION_RATE'] = 0.0
            
#             # Sector-specific degradation patterns
#             for sector in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']:
#                 if sector in car_laps.columns:
#                     sector_times = car_laps[sector].values
#                     if len(sector_times) >= 5 and not np.all(np.isnan(sector_times)):
#                         try:
#                             sector_slope = np.polyfit(lap_numbers[:len(sector_times)], 
#                                                     sector_times, 1)[0]
#                             df.loc[car_mask, f'{sector}_DEGRADATION'] = sector_slope
#                         except:
#                             df.loc[car_mask, f'{sector}_DEGRADATION'] = 0.0
            
#             # Performance consistency (lower = better tire management)
#             if len(lap_times) >= 10:
#                 df.loc[car_mask, 'PERFORMANCE_CONSISTENCY'] = np.std(lap_times)
            
#             # Tire age with non-linear effects
#             df.loc[car_mask, 'TIRE_AGE_NONLINEAR'] = np.log1p(car_laps['LAP_NUMBER']) * 0.5
        
#         # Add telemetry-based tire features if available
#         if not telemetry_data.empty:
#             df = FeatureEngineer._add_telemetry_tire_features(df, telemetry_data)
        
#         return df
    
#     @staticmethod
#     def _add_telemetry_tire_features(lap_data: pd.DataFrame, telemetry_data: pd.DataFrame) -> pd.DataFrame:
#         """Add tire-related features from telemetry data"""
#         df = lap_data.copy()
        
#         # Group telemetry by car and lap
#         telemetry_features = []
#         for (car_number, lap_num), lap_telemetry in telemetry_data.groupby(['vehicle_number', 'lap']):
#             if len(lap_telemetry) < 10:  # Minimum telemetry points
#                 continue
                
#             # Calculate driving style metrics that affect tires
#             lateral_g_mean = lap_telemetry['LATERAL_ACCEL'].abs().mean()
#             lateral_g_std = lap_telemetry['LATERAL_ACCEL'].abs().std()
#             brake_pressure_mean = lap_telemetry['TOTAL_BRAKE_PRESSURE'].mean()
            
#             # Steering activity (indicator of tire scrubbing)
#             steering_activity = lap_telemetry['STEERING_ANGLE'].diff().abs().sum()
            
#             telemetry_features.append({
#                 'NUMBER': car_number,
#                 'LAP_NUMBER': lap_num,
#                 'LATERAL_G_MEAN': lateral_g_mean,
#                 'LATERAL_G_VARIANCE': lateral_g_std,
#                 'BRAKE_INTENSITY': brake_pressure_mean,
#                 'STEERING_ACTIVITY': steering_activity
#             })
        
#         if telemetry_features:
#             telemetry_df = pd.DataFrame(telemetry_features)
#             df = df.merge(telemetry_df, on=['NUMBER', 'LAP_NUMBER'], how='left')
        
#         return df
    
#     @staticmethod
#     def engineer_fuel_features(lap_data: pd.DataFrame, telemetry_data: pd.DataFrame) -> pd.DataFrame:
#         """Create realistic fuel consumption features using telemetry data"""
#         if lap_data.empty:
#             return lap_data
            
#         df = lap_data.copy()
        
#         for car_number in df['NUMBER'].unique():
#             car_mask = df['NUMBER'] == car_number
#             car_laps = df[car_mask].sort_values('LAP_NUMBER')
            
#             if len(car_laps) < 5:
#                 continue
            
#             # Fuel load estimation (non-linear due to weight effect)
#             total_race_laps = car_laps['LAP_NUMBER'].max()
#             current_laps = car_laps['LAP_NUMBER'].values
            
#             # Weight reduction effect (faster at lower fuel)
#             base_fuel = 80.0  # liters for GR86
#             fuel_remaining = np.maximum(0, base_fuel * (1 - current_laps / total_race_laps))
#             df.loc[car_mask, 'ESTIMATED_FUEL_LOAD_KG'] = fuel_remaining * 0.75  # Convert to kg
            
#             # Fuel effect on lap time (empirical model)
#             weight_penalty = fuel_remaining * 0.03  # 0.03s per kg
#             df.loc[car_mask, 'FUEL_PENALTY_ESTIMATE'] = weight_penalty
            
#             # Fuel burn rate estimation
#             if len(current_laps) > 1:
#                 fuel_burn_rate = np.diff(fuel_remaining) / np.diff(current_laps)
#                 df.loc[car_mask, 'FUEL_BURN_RATE'] = np.concatenate([[fuel_burn_rate[0]], fuel_burn_rate])
        
#         # Add throttle-based fuel consumption if telemetry available
#         if not telemetry_data.empty:
#             df = FeatureEngineer._add_telemetry_fuel_features(df, telemetry_data)
        
#         return df
    
#     @staticmethod
#     def _add_telemetry_fuel_features(lap_data: pd.DataFrame, telemetry_data: pd.DataFrame) -> pd.DataFrame:
#         """Add fuel-related features from telemetry data"""
#         df = lap_data.copy()
        
#         throttle_features = []
#         for (car_number, lap_num), lap_telemetry in telemetry_data.groupby(['vehicle_number', 'lap']):
#             if len(lap_telemetry) < 10:
#                 continue
            
#             # Throttle usage patterns
#             throttle_mean = lap_telemetry['THROTTLE_POSITION'].mean()
#             throttle_std = lap_telemetry['THROTTLE_POSITION'].std()
            
#             # High-load throttle usage (indicative of fuel consumption)
#             high_throttle_pct = (lap_telemetry['THROTTLE_POSITION'] > 80).mean() * 100
            
#             # Engine load approximation
#             avg_speed = lap_telemetry.get('KPH', 0).mean()
#             engine_load = (throttle_mean * avg_speed) / 10000  # Simplified engine load
            
#             throttle_features.append({
#                 'NUMBER': car_number,
#                 'LAP_NUMBER': lap_num,
#                 'THROTTLE_MEAN': throttle_mean,
#                 'THROTTLE_VARIANCE': throttle_std,
#                 'HIGH_THROTTLE_PCT': high_throttle_pct,
#                 'ENGINE_LOAD_ESTIMATE': engine_load
#             })
        
#         if throttle_features:
#             throttle_df = pd.DataFrame(throttle_features)
#             df = df.merge(throttle_df, on=['NUMBER', 'LAP_NUMBER'], how='left')
            
#             # Fuel consumption estimate based on throttle usage
#             if 'THROTTLE_MEAN' in df.columns and 'ENGINE_LOAD_ESTIMATE' in df.columns:
#                 base_consumption = 2.8  # liters per lap
#                 throttle_factor = df['THROTTLE_MEAN'] / 100 * 0.8
#                 load_factor = df['ENGINE_LOAD_ESTIMATE'] * 1.2
#                 df['ESTIMATED_FUEL_CONSUMPTION'] = base_consumption * (1 + throttle_factor + load_factor)
        
#         return df
    
#     @staticmethod
#     def engineer_strategy_features(race_data: pd.DataFrame, lap_data: pd.DataFrame) -> pd.DataFrame:
#         """Create realistic race strategy features"""
#         strategy_features = []
        
#         if race_data.empty or lap_data.empty:
#             return pd.DataFrame()
        
#         for car_number in race_data['NUMBER'].unique():
#             car_race = race_data[race_data['NUMBER'] == car_number]
#             if car_race.empty:
#                 continue
                
#             car_race = car_race.iloc[0]
#             car_laps = lap_data[lap_data['NUMBER'] == car_number].sort_values('LAP_NUMBER')
            
#             if len(car_laps) < 5:
#                 continue
            
#             # Position and gap analysis
#             position = car_race.get('POSITION', 1)
#             gap_to_leader = FeatureEngineer._parse_gap(car_race.get('GAP_FIRST', '0'))
#             gap_to_next = FeatureEngineer._parse_gap(car_race.get('GAP_PREVIOUS', '0'))
            
#             # Performance analysis
#             lap_times = car_laps['LAP_TIME_SECONDS'].values
#             best_lap_time = np.min(lap_times)
#             avg_lap_time = np.mean(lap_times)
#             consistency = np.std(lap_times)
            
#             # Race situation analysis
#             total_laps = car_race.get('LAPS', len(car_laps))
#             best_lap_num = car_laps.loc[car_laps['LAP_TIME_SECONDS'].idxmin(), 'LAP_NUMBER']
#             race_progress = best_lap_num / total_laps if total_laps > 0 else 0
            
#             # Competitive pressure metrics
#             position_pressure = 1.0 / max(1, position)
#             gap_pressure = 1.0 / max(1, gap_to_next) if gap_to_next > 0 else 1.0
            
#             # Pace analysis
#             pace_deficit = (avg_lap_time - best_lap_time) / best_lap_time
#             has_winning_pace = 1 if pace_deficit < 0.02 and position <= 5 else 0  # Within 2% of best pace
            
#             strategy_features.append({
#                 'car_number': car_number,
#                 'position': position,
#                 'gap_to_leader': gap_to_leader,
#                 'gap_to_next': gap_to_next,
#                 'best_lap_time': best_lap_time,
#                 'avg_lap_time': avg_lap_time,
#                 'performance_consistency': consistency,
#                 'pace_deficit': pace_deficit,
#                 'optimal_lap_timing': best_lap_num,
#                 'race_progress': race_progress,
#                 'position_pressure': position_pressure,
#                 'gap_pressure': gap_pressure,
#                 'total_laps': total_laps,
#                 'has_winning_pace': has_winning_pace,
#                 'needs_strategy_change': 1 if (position_pressure > 0.3 or pace_deficit > 0.03) else 0
#             })
        
#         return pd.DataFrame(strategy_features) if strategy_features else pd.DataFrame()
    
#     @staticmethod
#     def engineer_track_features(track_name: str, lap_data: pd.DataFrame) -> pd.DataFrame:
#         """Create track-specific features based on actual performance data"""
#         if lap_data.empty:
#             return lap_data
            
#         df = lap_data.copy()
        
#         # Calculate actual track characteristics from sector data
#         if all(col in df.columns for col in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']):
#             sector_means = df[['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']].mean()
#             total_time = sector_means.sum()
            
#             # Track layout characteristics
#             df['S1_PERCENTAGE'] = sector_means['S1_SECONDS'] / total_time
#             df['S2_PERCENTAGE'] = sector_means['S2_SECONDS'] / total_time  
#             df['S3_PERCENTAGE'] = sector_means['S3_SECONDS'] / total_time
            
#             # Track technicality (higher variance = more technical)
#             sector_variance = df[['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']].var().mean()
#             df['TRACK_TECHNICALITY'] = min(1.0, sector_variance / 2.0)
        
#         # Track-specific wear factors based on real characteristics
#         track_wear_map = {
#             'sebring': 0.9,      # Bumpy surface, high wear
#             'barber': 0.85,      # Technical, elevation changes
#             'sonoma': 0.8,       # Hilly, abrasive
#             'road-america': 0.7, # Long straights, medium wear
#             'vir': 0.75,         # Balanced
#             'cota': 0.6,         # Modern, smooth
#             'indianapolis': 0.5  # Oval, low wear
#         }
        
#         df['TRACK_WEAR_FACTOR'] = track_wear_map.get(track_name.lower(), 0.7)
        
#         # Overtaking potential (based on speed variance)
#         if 'KPH' in df.columns:
#             speed_variance = df['KPH'].var() / df['KPH'].mean() if df['KPH'].mean() > 0 else 0.1
#             df['OVERTAKING_POTENTIAL'] = min(1.0, speed_variance * 10)
        
#         return df
    
#     @staticmethod
#     def engineer_weather_features(weather_data: pd.DataFrame, lap_data: pd.DataFrame) -> pd.DataFrame:
#         """Create realistic weather impact features"""
#         if weather_data.empty or lap_data.empty:
#             return lap_data.copy()
            
#         df = lap_data.copy()
        
#         # Calculate weather conditions during the session
#         weather_avg = weather_data.agg({
#             'AIR_TEMP': 'mean',
#             'TRACK_TEMP': 'mean', 
#             'HUMIDITY': 'mean',
#             'PRESSURE': 'mean',
#             'WIND_SPEED': 'mean',
#             'RAIN': 'max'
#         })
        
#         # Temperature impact on performance
#         optimal_temp = 25.0  # Optimal air temperature
#         temp_diff = weather_avg['AIR_TEMP'] - optimal_temp
#         df['TEMP_IMPACT'] = temp_diff * 0.03  # 0.03s per degree from optimal
        
#         # Track temperature effect on tires
#         optimal_track_temp = 35.0
#         track_temp_diff = weather_avg['TRACK_TEMP'] - optimal_track_temp
#         df['TRACK_TEMP_IMPACT'] = track_temp_diff * 0.02  # 0.02s per degree
        
#         # Humidity effect (engine performance)
#         optimal_humidity = 50.0
#         humidity_diff = weather_avg['HUMIDITY'] - optimal_humidity
#         df['HUMIDITY_IMPACT'] = humidity_diff * 0.001  # Small effect per percent
        
#         # Air density effect (engine power)
#         air_density = FeatureEngineer._calculate_air_density(
#             weather_avg['AIR_TEMP'], 
#             weather_avg['PRESSURE'],
#             weather_avg['HUMIDITY']
#         )
#         std_air_density = 1.225  # kg/m³ at sea level, 15°C
#         density_ratio = air_density / std_air_density
#         df['AIR_DENSITY_IMPACT'] = (1 - density_ratio) * 2.0  # 2s effect at extreme conditions
        
#         # Rain impact (major effect)
#         rain_effect = weather_avg['RAIN'] * 1.5  # 1.5s per mm of rain
#         df['RAIN_IMPACT'] = rain_effect
        
#         # Combined weather effect
#         df['TOTAL_WEATHER_IMPACT'] = (
#             df['TEMP_IMPACT'] + df['TRACK_TEMP_IMPACT'] + 
#             df['HUMIDITY_IMPACT'] + df['AIR_DENSITY_IMPACT'] + 
#             df['RAIN_IMPACT']
#         )
        
#         return df
    
#     @staticmethod
#     def _calculate_air_density(air_temp: float, pressure: float, humidity: float) -> float:
#         """Calculate air density for engine performance impact"""
#         # Simplified air density calculation
#         R = 287.05  # J/kg·K
#         temp_k = air_temp + 273.15
        
#         # Vapor pressure calculation
#         vapor_pressure = 0.611 * np.exp(17.27 * air_temp / (air_temp + 237.3)) * (humidity / 100)
        
#         # Dry air pressure (simplified)
#         dry_pressure = pressure - vapor_pressure
        
#         # Air density in kg/m³
#         return (dry_pressure * 100) / (R * temp_k)  # Convert pressure to Pa
    
#     @staticmethod
#     def create_composite_features(processed_data: Dict) -> Dict:
#         """Create all composite features for model training"""
#         enhanced_data = {}
        
#         for track_name, data in processed_data.items():
#             lap_data = data['lap_data']
#             race_data = data['race_data']
#             weather_data = data['weather_data']
#             telemetry_data = data.get('telemetry_data', pd.DataFrame())
            
#             # Skip if no lap data
#             if lap_data.empty:
#                 enhanced_data[track_name] = data
#                 continue
            
#             # Apply all feature engineering steps with telemetry integration
#             lap_data = FeatureEngineer.engineer_tire_features(lap_data, telemetry_data)
#             lap_data = FeatureEngineer.engineer_fuel_features(lap_data, telemetry_data)
#             lap_data = FeatureEngineer.engineer_track_features(track_name, lap_data)
#             lap_data = FeatureEngineer.engineer_weather_features(weather_data, lap_data)
            
#             strategy_features = FeatureEngineer.engineer_strategy_features(race_data, lap_data)
            
#             enhanced_data[track_name] = {
#                 'lap_data': lap_data,
#                 'race_data': race_data,
#                 'weather_data': weather_data,
#                 'telemetry_data': telemetry_data,
#                 'strategy_features': strategy_features
#             }
        
#         return enhanced_data
    
#     @staticmethod
#     def _parse_gap(gap_str: str) -> float:
#         """Parse gap string to seconds"""
#         if pd.isna(gap_str) or gap_str in ['-', '']:
#             return 0.0
#         try:
#             gap_str = str(gap_str).replace('+', '').strip()
#             return float(gap_str)
#         except:
#             return 0.0



















# import pandas as pd
# import numpy as np
# from typing import Dict, List

# class FeatureEngineer:
#     """Engineer advanced features from raw racing data"""
    
#     @staticmethod
#     def engineer_tire_features(lap_data: pd.DataFrame) -> pd.DataFrame:
#         """Create advanced tire degradation features"""
#         df = lap_data.copy()
        
#         # Rolling performance metrics
#         df['ROLLING_5_LAP_AVG'] = df['LAP_TIME_SECONDS'].rolling(window=5, min_periods=1).mean()
#         df['PERFORMANCE_TREND'] = df['LAP_TIME_SECONDS'].diff().rolling(window=3).mean()
        
#         # Sector consistency (tire wear indicator)
#         df['SECTOR_VARIANCE'] = df[['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']].std(axis=1)
#         df['SECTOR_BALANCE'] = (df['S1_SECONDS'] - df['S3_SECONDS']).abs()
        
#         # Tire age with exponential decay factor
#         df['TIRE_AGE_EXP'] = np.exp(df['LAP_NUMBER'] * 0.1) - 1
        
#         # Performance drop from personal best
#         df['PERSONAL_BEST'] = df.groupby('NUMBER')['LAP_TIME_SECONDS'].transform('min')
#         df['DROP_FROM_PB'] = df['LAP_TIME_SECONDS'] - df['PERSONAL_BEST']
        
#         # Rolling degradation rate
#         df['DEGRADATION_RATE'] = df['LAP_TIME_SECONDS'].diff().rolling(window=5).mean()
        
#         return df
    
#     @staticmethod
#     def engineer_fuel_features(lap_data: pd.DataFrame) -> pd.DataFrame:
#         """Create advanced fuel consumption features"""
#         df = lap_data.copy()
        
#         # Fuel load estimation (decreases linearly with laps)
#         total_laps = df['LAP_NUMBER'].max()
#         df['ESTIMATED_FUEL_LOAD'] = 1.0 - (df['LAP_NUMBER'] / total_laps)
        
#         # Speed efficiency (higher speed = more fuel consumption)
#         df['SPEED_EFFICIENCY'] = df['KPH'] / df['LAP_TIME_SECONDS']
        
#         # Throttle usage approximation
#         df['THROTTLE_ESTIMATE'] = (df['KPH'] / df['KPH'].max()) * 100
        
#         # Fuel burn rate trend
#         df['FUEL_BURN_TREND'] = df['LAP_TIME_SECONDS'].rolling(window=3).std()
        
#         # Lap time improvement (fuel burn effect)
#         df['LAP_IMPROVEMENT'] = df['LAP_TIME_SECONDS'].diff().rolling(window=5).mean() * -1
        
#         return df
    
#     @staticmethod
#     def engineer_strategy_features(race_data: pd.DataFrame, lap_data: pd.DataFrame) -> pd.DataFrame:
#         """Create race strategy features"""
#         strategy_features = []
        
#         for car_number in race_data['NUMBER'].unique():
#             car_race = race_data[race_data['NUMBER'] == car_number].iloc[0]
#             car_laps = lap_data[lap_data['NUMBER'] == car_number]
            
#             if len(car_laps) < 5:
#                 continue
            
#             # Position-based features
#             position = car_race.get('POSITION', 1)
#             gap_to_leader = FeatureEngineer._parse_gap(car_race.get('GAP_FIRST', '0'))
#             gap_to_next = FeatureEngineer._parse_gap(car_race.get('GAP_PREVIOUS', '0'))
            
#             # Performance consistency
#             lap_std = car_laps['LAP_TIME_SECONDS'].std()
#             best_lap_num = car_race.get('BEST_LAP_NUM', car_laps['LAP_NUMBER'].iloc[car_laps['LAP_TIME_SECONDS'].argmin()])
            
#             # Race phase analysis
#             total_laps = car_race.get('LAPS', len(car_laps))
#             race_progress = best_lap_num / total_laps
            
#             # Competitor pressure
#             position_pressure = 1.0 / position if position > 0 else 1.0
            
#             strategy_features.append(pd.DataFrame([{
#                 'car_number': car_number,
#                 'position': position,
#                 'gap_to_leader': gap_to_leader,
#                 'gap_to_next': gap_to_next,
#                 'performance_consistency': lap_std,
#                 'optimal_lap_timing': best_lap_num,
#                 'race_progress': race_progress,
#                 'position_pressure': position_pressure,
#                 'total_laps': total_laps,
#                 'has_winning_pace': 1 if position <= 3 else 0
#             }]))
        
#         if strategy_features:
#             return pd.concat(strategy_features, ignore_index=True)
#         return pd.DataFrame()
    
#     @staticmethod
#     def engineer_track_features(track_name: str, lap_data: pd.DataFrame) -> pd.DataFrame:
#         """Create track-specific features"""
#         df = lap_data.copy()
        
#         # Track characteristics based on sector times
#         sector_ratios = df[['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']].mean()
#         total_sector_time = sector_ratios.sum()
        
#         df['S1_RATIO'] = sector_ratios['S1_SECONDS'] / total_sector_time
#         df['S2_RATIO'] = sector_ratios['S2_SECONDS'] / total_sector_time
#         df['S3_RATIO'] = sector_ratios['S3_SECONDS'] / total_sector_time
        
#         # Track wear classification
#         track_wear_factors = {
#             'barber-motorsports-park': 'high',
#             'circuit-of-the-americas': 'medium',
#             'indianapolis': 'low', 
#             'road-america': 'medium',
#             'sebring': 'high',
#             'sonoma': 'medium',
#             'virginia-international-raceway': 'medium'
#         }
        
#         wear_level = track_wear_factors.get(track_name, 'medium')
#         df['TRACK_WEAR_FACTOR'] = 0.9 if wear_level == 'high' else 0.7 if wear_level == 'medium' else 0.5
        
#         # Overtaking difficulty (based on top speed variance)
#         overtaking_difficulty = df['TOP_SPEED'].std() / df['TOP_SPEED'].mean() if 'TOP_SPEED' in df.columns else 0.1
#         df['OVERTAKING_DIFFICULTY'] = overtaking_difficulty
        
#         return df
    
#     @staticmethod
#     def engineer_weather_features(weather_data: pd.DataFrame, lap_data: pd.DataFrame) -> pd.DataFrame:
#         """Create weather impact features"""
#         if weather_data.empty:
#             return lap_data.copy()
        
#         df = lap_data.copy()
#         weather_avg = weather_data.mean(numeric_only=True)
        
#         # Temperature impact
#         df['TEMP_IMPACT'] = (weather_avg.get('AIR_TEMP', 25) - 25) * 0.1  # 0.1s per degree
        
#         # Humidity impact  
#         df['HUMIDITY_IMPACT'] = (weather_avg.get('HUMIDITY', 50) - 50) * 0.05  # 0.05s per 10% humidity
        
#         # Wind impact (simplified)
#         wind_speed = weather_avg.get('WIND_SPEED', 0)
#         df['WIND_IMPACT'] = wind_speed * 0.02  # 0.02s per km/h
        
#         # Rain impact
#         rain = weather_avg.get('RAIN', 0)
#         df['RAIN_IMPACT'] = rain * 0.5  # 0.5s per mm of rain
        
#         # Combined weather effect
#         df['TOTAL_WEATHER_IMPACT'] = (df['TEMP_IMPACT'] + df['HUMIDITY_IMPACT'] + 
#                                     df['WIND_IMPACT'] + df['RAIN_IMPACT'])
        
#         return df
    
#     @staticmethod
#     def create_composite_features(processed_data: Dict) -> Dict:
#         """Create all composite features for model training"""
#         enhanced_data = {}
        
#         for track_name, data in processed_data.items():
#             lap_data = data['lap_data']
#             race_data = data['race_data']
#             weather_data = data['weather_data']
            
#             # Apply all feature engineering steps
#             lap_data = FeatureEngineer.engineer_tire_features(lap_data)
#             lap_data = FeatureEngineer.engineer_fuel_features(lap_data)
#             lap_data = FeatureEngineer.engineer_track_features(track_name, lap_data)
#             lap_data = FeatureEngineer.engineer_weather_features(weather_data, lap_data)
            
#             strategy_features = FeatureEngineer.engineer_strategy_features(race_data, lap_data)
            
#             enhanced_data[track_name] = {
#                 'lap_data': lap_data,
#                 'race_data': race_data,
#                 'weather_data': weather_data,
#                 'strategy_features': strategy_features
#             }
        
#         return enhanced_data
    
#     @staticmethod
#     def _parse_gap(gap_str: str) -> float:
#         """Parse gap string to seconds"""
#         if pd.isna(gap_str) or gap_str in ['-', '']:
#             return 0
#         try:
#             gap_str = str(gap_str).replace('+', '').strip()
#             return float(gap_str)
#         except:
#             return 0