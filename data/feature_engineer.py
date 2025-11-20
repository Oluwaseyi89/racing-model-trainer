import pandas as pd
import numpy as np
from typing import Dict, Iterable, Optional, Tuple, Any
from scipy import stats


class FeatureEngineer:
    """
    Feature engineering for racing analytics - Consistent with FirebaseDataLoader schemas.
    Uses EXACT column names from the data structures provided.
    """

    # ----------------------
    # Helper / normalization - Updated for FirebaseDataLoader consistency
    # ----------------------
    @staticmethod
    def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _ensure_number_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Guarantee that dataframe has a numeric 'NUMBER' column using EXACT FirebaseDataLoader names.
        """
        df = df.copy()
        candidates = ["NUMBER", "DRIVER_NUMBER"]  # EXACT column names from your data
        col = FeatureEngineer._first_existing_column(df, candidates)
        
        if col:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            if col != "NUMBER":
                df = df.rename(columns={col: "NUMBER"})
                return df, "NUMBER"
            return df, col
        return df, None

    @staticmethod
    def _ensure_lap_number(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure LAP_NUMBER column exists using EXACT FirebaseDataLoader names."""
        df = df.copy()
        candidates = ["LAP_NUMBER", "lap"]  # EXACT column names
        
        col = FeatureEngineer._first_existing_column(df, candidates)
        if col and col != "LAP_NUMBER":
            df = df.rename(columns={col: "LAP_NUMBER"})
            
        if "LAP_NUMBER" not in df.columns:
            if "NUMBER" in df.columns:
                df["LAP_NUMBER"] = df.groupby("NUMBER").cumcount() + 1
            else:
                df["LAP_NUMBER"] = np.arange(len(df)) + 1
                
        df["LAP_NUMBER"] = pd.to_numeric(df["LAP_NUMBER"], errors="coerce").fillna(0)
        return df

    @staticmethod
    def _safe_regression(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Run a robust linear regression if inputs are sane.
        Returns (slope, r_squared) or None on failure/safety checks.
        """
        try:
            mask = (~np.isnan(x)) & (~np.isnan(y))
            if mask.sum() < 5:
                return None
            xv = x[mask].astype(float)
            yv = y[mask].astype(float)
            if xv.size < 5:
                return None
            slope, intercept, r_value, p_value, std_err = stats.linregress(xv, yv)
            r2 = float(r_value ** 2)
            return float(slope), r2
        except Exception:
            return None

    @staticmethod
    def _safe_polyfit_slope(x: np.ndarray, y: np.ndarray) -> Optional[float]:
        """Try to get first-degree polynomial slope with safety checks."""
        try:
            mask = (~np.isnan(x)) & (~np.isnan(y))
            if mask.sum() < 5:
                return None
            xv = x[mask].astype(float)
            yv = y[mask].astype(float)
            if xv.size < 5:
                return None
            coeffs = np.polyfit(xv, yv, 1)
            slope = float(coeffs[0])
            return slope
        except Exception:
            return None

    # ------------------------------------------------------------
    # TIRE FEATURES - Updated for FirebaseDataLoader consistency
    # ------------------------------------------------------------
    @staticmethod
    def engineer_tire_features(pit_data: pd.DataFrame,
                               telemetry_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer tire features using EXACT column names from pit_data and telemetry_data."""
        if pit_data is None or pit_data.empty:
            return pd.DataFrame()

        pit_df = pit_data.copy()
        pit_df, id_col = FeatureEngineer._ensure_number_column(pit_df)
        pit_df = FeatureEngineer._ensure_lap_number(pit_df)

        if id_col is None or "NUMBER" not in pit_df.columns:
            return pit_df

        # Initialize tire feature columns
        pit_df["TIRE_DEGRADATION_RATE"] = np.nan
        pit_df["PERFORMANCE_CONSISTENCY"] = np.nan
        pit_df["TIRE_AGE_NONLINEAR"] = np.nan

        try:
            for car_number in pd.unique(pit_df["NUMBER"].dropna()):
                car_mask = pit_df["NUMBER"] == car_number
                car_laps = pit_df.loc[car_mask].sort_values("LAP_NUMBER")

                if car_laps.shape[0] < 5:
                    continue

                # Use LAP_TIME_SECONDS or convert LAP_TIME using EXACT column names
                if "LAP_TIME_SECONDS" in car_laps.columns:
                    lap_times = pd.to_numeric(car_laps["LAP_TIME_SECONDS"], errors="coerce").values
                    lap_numbers = pd.to_numeric(car_laps["LAP_NUMBER"], errors="coerce").values
                elif "LAP_TIME" in car_laps.columns:
                    lap_times = car_laps["LAP_TIME"].apply(FeatureEngineer._time_to_seconds).values
                    lap_numbers = pd.to_numeric(car_laps["LAP_NUMBER"], errors="coerce").values
                else:
                    lap_times = np.array([])
                    lap_numbers = np.array([])

                # Lap time degradation analysis
                if lap_times.size >= 8:
                    mask_range = (lap_numbers >= 5) & (lap_numbers <= 15)
                    if mask_range.sum() >= 5:
                        res = FeatureEngineer._safe_regression(lap_numbers, lap_times)
                        if res is not None:
                            slope, r2 = res
                            if r2 > 0.4:
                                pit_df.loc[car_mask, "TIRE_DEGRADATION_RATE"] = slope
                            else:
                                pit_df.loc[car_mask, "TIRE_DEGRADATION_RATE"] = 0.0

                # Sector degradation using EXACT column names
                for sector in ["S1_SECONDS", "S2_SECONDS", "S3_SECONDS"]:
                    if sector in car_laps.columns:
                        sector_times = pd.to_numeric(car_laps[sector], errors="coerce").fillna(0).values
                        slope = FeatureEngineer._safe_polyfit_slope(car_laps["LAP_NUMBER"].values, sector_times)
                        if slope is not None:
                            pit_df.loc[car_mask, f"{sector}_DEGRADATION"] = slope

                # Performance consistency
                if "LAP_TIME_SECONDS" in car_laps.columns:
                    try:
                        pit_df.loc[car_mask, "PERFORMANCE_CONSISTENCY"] = \
                            float(np.nanstd(pd.to_numeric(car_laps["LAP_TIME_SECONDS"], errors="coerce")))
                    except Exception:
                        pit_df.loc[car_mask, "PERFORMANCE_CONSISTENCY"] = np.nan

                # Non-linear tire age effect
                try:
                    pit_df.loc[car_mask, "TIRE_AGE_NONLINEAR"] = np.log1p(
                        pd.to_numeric(car_laps["LAP_NUMBER"], errors="coerce")
                    ).fillna(0).values * 0.5
                except Exception:
                    pit_df.loc[car_mask, "TIRE_AGE_NONLINEAR"] = np.nan

        except Exception as e:
            print(f"⚠️ Tire feature engineering failed: {e}")

        # Add telemetry-based tire features if available
        if telemetry_data is not None and not telemetry_data.empty:
            try:
                pit_df = FeatureEngineer._add_telemetry_tire_features(pit_df, telemetry_data)
            except Exception as e:
                print(f"⚠️ Telemetry tire merging failed: {e}")

        return pit_df

    @staticmethod
    def _add_telemetry_tire_features(pit_df: pd.DataFrame,
                                     telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """Add telemetry-based tire features using EXACT column names."""
        df = pit_df.copy()
        telemetry = telemetry_df.copy()

        # Use EXACT column names from telemetry_data
        if "vehicle_id" not in telemetry.columns or "lap" not in telemetry.columns:
            return df

        # Ensure numeric types
        telemetry["vehicle_id"] = telemetry["vehicle_id"].astype(str)
        telemetry["lap"] = pd.to_numeric(telemetry["lap"], errors="coerce").fillna(0)

        # Extract vehicle number from vehicle_id (e.g., "GR86-002-000" -> 2)
        telemetry["NUMBER"] = telemetry["vehicle_id"].apply(
            lambda x: int(x.split('-')[1]) if '-' in x and x.split('-')[1].isdigit() else 0
        )

        # Aggregate telemetry features using EXACT column names
        agg_cols = {}
        if "accy_can" in telemetry.columns:  # Lateral acceleration
            agg_cols["accy_can"] = ["mean", "std"]
        if "accx_can" in telemetry.columns:  # Longitudinal acceleration (braking)
            agg_cols["accx_can"] = ["mean"]

        if not agg_cols:
            return df

        try:
            # Group by NUMBER and lap (LAP_NUMBER in pit_data)
            telemetry_agg = telemetry.groupby(["NUMBER", "lap"]).agg(agg_cols)
            telemetry_agg.columns = ["_".join(col).strip() for col in telemetry_agg.columns.values]
            telemetry_agg = telemetry_agg.reset_index()

            # Rename to meaningful feature names
            rename_map = {}
            if "accy_can_mean" in telemetry_agg.columns:
                rename_map["accy_can_mean"] = "LATERAL_G_MEAN"
            if "accy_can_std" in telemetry_agg.columns:
                rename_map["accy_can_std"] = "LATERAL_G_STD"
            if "accx_can_mean" in telemetry_agg.columns:
                rename_map["accx_can_mean"] = "LONGITUDINAL_G_MEAN"

            telemetry_agg = telemetry_agg.rename(columns=rename_map)
            telemetry_agg = telemetry_agg.rename(columns={"lap": "LAP_NUMBER"})

            # Ensure proper numeric types
            telemetry_agg["NUMBER"] = pd.to_numeric(telemetry_agg["NUMBER"], errors="coerce").fillna(0)
            telemetry_agg["LAP_NUMBER"] = pd.to_numeric(telemetry_agg["LAP_NUMBER"], errors="coerce").fillna(0)

            # Merge with pit data
            df = df.merge(telemetry_agg, on=["NUMBER", "LAP_NUMBER"], how="left")

        except Exception as e:
            print(f"⚠️ Telemetry tire feature aggregation failed: {e}")

        return df

    # ------------------------------------------------------------
    # FUEL FEATURES - Updated for FirebaseDataLoader consistency
    # ------------------------------------------------------------
    @staticmethod
    def engineer_fuel_features(pit_data: pd.DataFrame,
                               telemetry_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer fuel features using EXACT column names."""
        if pit_data is None or pit_data.empty:
            return pd.DataFrame()

        df = pit_data.copy()
        df, _ = FeatureEngineer._ensure_number_column(df)
        df = FeatureEngineer._ensure_lap_number(df)

        try:
            # Fuel efficiency estimation using EXACT column names
            if "LAP_TIME_SECONDS" in df.columns:
                df["FUEL_EFFICIENCY_EST"] = 1.0 / (
                    pd.to_numeric(df["LAP_TIME_SECONDS"], errors="coerce").fillna(1.0) + 0.1
                )
            else:
                df["FUEL_EFFICIENCY_EST"] = np.nan

            # Speed-based fuel consumption
            if "KPH" in df.columns:
                kph = pd.to_numeric(df["KPH"], errors="coerce").fillna(0)
                df["FUEL_CONSUMPTION_RATE"] = kph * 0.02  # Simplified model

            # Add telemetry-based fuel features
            if telemetry_data is not None and not telemetry_data.empty:
                df = FeatureEngineer._add_telemetry_fuel_features(df, telemetry_data)

        except Exception as e:
            print(f"⚠️ Fuel feature engineering failed: {e}")

        return df

    @staticmethod
    def _add_telemetry_fuel_features(pit_df: pd.DataFrame,
                                     telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """Add telemetry-based fuel features using EXACT column names."""
        df = pit_df.copy()
        telemetry = telemetry_df.copy()

        if "vehicle_id" not in telemetry.columns or "lap" not in telemetry.columns:
            return df

        # Extract vehicle number and ensure numeric types
        telemetry["NUMBER"] = telemetry["vehicle_id"].apply(
            lambda x: int(x.split('-')[1]) if '-' in x and x.split('-')[1].isdigit() else 0
        )
        telemetry["lap"] = pd.to_numeric(telemetry["lap"], errors="coerce").fillna(0)

        # Calculate throttle usage and speed patterns
        agg_cols = {}
        if "speed" in telemetry.columns:
            agg_cols["speed"] = ["mean", "std"]
        if "gear" in telemetry.columns:
            agg_cols["gear"] = ["mean"]

        if agg_cols:
            try:
                telemetry_agg = telemetry.groupby(["NUMBER", "lap"]).agg(agg_cols)
                telemetry_agg.columns = ["_".join(col).strip() for col in telemetry_agg.columns.values]
                telemetry_agg = telemetry_agg.reset_index()

                # Rename columns
                rename_map = {}
                if "speed_mean" in telemetry_agg.columns:
                    rename_map["speed_mean"] = "TELEMETRY_SPEED_MEAN"
                if "speed_std" in telemetry_agg.columns:
                    rename_map["speed_std"] = "TELEMETRY_SPEED_STD"
                if "gear_mean" in telemetry_agg.columns:
                    rename_map["gear_mean"] = "AVG_GEAR"

                telemetry_agg = telemetry_agg.rename(columns=rename_map)
                telemetry_agg = telemetry_agg.rename(columns={"lap": "LAP_NUMBER"})

                # Ensure numeric types
                telemetry_agg["NUMBER"] = pd.to_numeric(telemetry_agg["NUMBER"], errors="coerce").fillna(0)
                telemetry_agg["LAP_NUMBER"] = pd.to_numeric(telemetry_agg["LAP_NUMBER"], errors="coerce").fillna(0)

                # Merge with pit data
                df = df.merge(telemetry_agg, on=["NUMBER", "LAP_NUMBER"], how="left")

            except Exception as e:
                print(f"⚠️ Telemetry fuel feature aggregation failed: {e}")

        return df

    # ------------------------------------------------------------
    # TRACK FEATURES - Updated for actual track names
    # ------------------------------------------------------------
    @staticmethod
    def engineer_track_features(track_name: str,
                                pit_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer track-specific features using EXACT track names."""
        if pit_data is None or pit_data.empty:
            return pd.DataFrame()

        df = pit_data.copy()

        # Track characteristics based on actual track names
        track_wear_map = {
            "barber": 0.8, "cota": 0.6, "indianapolis": 0.5,
            "road_america": 0.7, "sebring": 0.9, "sonoma": 0.8, "virginia": 0.75
        }

        track_overtaking_map = {
            "barber": 0.6, "cota": 0.8, "indianapolis": 0.9,
            "road_america": 0.7, "sebring": 0.5, "sonoma": 0.4, "virginia": 0.6
        }

        try:
            if track_name:
                df["TRACK_WEAR_FACTOR"] = track_wear_map.get(str(track_name).lower(), 0.7)
                df["OVERTAKING_POTENTIAL"] = track_overtaking_map.get(str(track_name).lower(), 0.5)
            else:
                df["TRACK_WEAR_FACTOR"] = 0.7
                df["OVERTAKING_POTENTIAL"] = 0.5

            # Speed-based overtaking potential refinement
            if "KPH" in df.columns:
                try:
                    kph = pd.to_numeric(df["KPH"], errors="coerce").fillna(0)
                    mean_speed = float(kph.mean()) if not kph.empty else 0.0
                    var_speed = float(kph.var()) if not kph.empty else 0.0
                    speed_factor = min(1.0, (var_speed / (mean_speed + 1e-6)) * 10) if mean_speed > 0 else 0.1
                    df["OVERTAKING_POTENTIAL"] = df["OVERTAKING_POTENTIAL"] * 0.7 + speed_factor * 0.3
                except Exception:
                    pass

        except Exception as e:
            print(f"⚠️ Track feature engineering failed: {e}")

        return df

    # ------------------------------------------------------------
    # WEATHER FEATURES - Updated for FirebaseDataLoader consistency
    # ------------------------------------------------------------
    @staticmethod
    def engineer_weather_features(weather_data: pd.DataFrame,
                                  pit_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer weather features using EXACT column names."""
        if pit_data is None or pit_data.empty:
            return pd.DataFrame()

        df = pit_data.copy()

        try:
            if weather_data is None or weather_data.empty:
                # Set default weather impacts
                df["TEMP_IMPACT"] = 0.0
                df["TRACK_TEMP_IMPACT"] = 0.0
                df["RAIN_IMPACT"] = 0.0
                return df

            weather = weather_data.copy()

            # Ensure numeric types using EXACT column names
            for col in ["AIR_TEMP", "TRACK_TEMP", "HUMIDITY", "PRESSURE", "WIND_SPEED", "RAIN"]:
                if col in weather.columns:
                    weather[col] = pd.to_numeric(weather[col], errors="coerce").fillna(0)

            # Calculate weather impacts
            if "AIR_TEMP" in weather.columns:
                temp_impact = (float(weather["AIR_TEMP"].mean()) - 25.0) * 0.03
                df["TEMP_IMPACT"] = temp_impact

            if "TRACK_TEMP" in weather.columns:
                track_temp_impact = (float(weather["TRACK_TEMP"].mean()) - 35.0) * 0.02
                df["TRACK_TEMP_IMPACT"] = track_temp_impact

            if "RAIN" in weather.columns:
                rain_impact = float(weather["RAIN"].max()) * 1.5
                df["RAIN_IMPACT"] = rain_impact

            # Combined weather effect
            df["TOTAL_WEATHER_IMPACT"] = (
                df.get("TEMP_IMPACT", 0) + 
                df.get("TRACK_TEMP_IMPACT", 0) + 
                df.get("RAIN_IMPACT", 0)
            )

        except Exception as e:
            print(f"⚠️ Weather feature engineering failed: {e}")

        return df

    # ------------------------------------------------------------
    # STRATEGY FEATURES - Updated for FirebaseDataLoader consistency
    # ------------------------------------------------------------
    @staticmethod
    def engineer_strategy_features(race_data: pd.DataFrame,
                                   pit_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer strategy features using EXACT column names."""
        if race_data is None or race_data.empty or pit_data is None or pit_data.empty:
            return pd.DataFrame()

        race_df = race_data.copy()
        pit_df = pit_data.copy()

        race_df, _ = FeatureEngineer._ensure_number_column(race_df)
        pit_df, _ = FeatureEngineer._ensure_number_column(pit_df)
        pit_df = FeatureEngineer._ensure_lap_number(pit_df)

        if "NUMBER" not in race_df.columns or "NUMBER" not in pit_df.columns:
            return pd.DataFrame()

        rows = []
        try:
            unique_numbers = pd.unique(race_df["NUMBER"].dropna())
            for car_number in unique_numbers:
                try:
                    car_race = race_df[race_df["NUMBER"] == car_number]
                    if car_race.empty:
                        continue
                    car_row = car_race.iloc[0]
                    car_laps = pit_df[pit_df["NUMBER"] == car_number].sort_values("LAP_NUMBER")
                    
                    if car_laps.shape[0] < 3:
                        continue

                    # Extract race position using EXACT column names
                    position = car_row.get("POSITION", np.nan)
                    try:
                        position = int(position) if not pd.isna(position) else np.nan
                    except Exception:
                        position = np.nan

                    total_laps = int(car_laps["LAP_NUMBER"].max()) if "LAP_NUMBER" in car_laps.columns else car_laps.shape[0]

                    # Strategy heuristics
                    needs_strategy_change = 1 if (not pd.isna(position) and position > 10) else 0
                    is_leading = 1 if (not pd.isna(position) and position <= 3) else 0

                    # Performance metrics
                    avg_lap_time = float(
                        pd.to_numeric(car_laps.get("LAP_TIME_SECONDS", pd.Series([])), errors="coerce"
                    ).mean(skipna=True)) if "LAP_TIME_SECONDS" in car_laps.columns else np.nan

                    tire_degradation = float(
                        pd.to_numeric(car_laps.get("TIRE_DEGRADATION_RATE", pd.Series([np.nan])), errors="coerce"
                    ).mean(skipna=True)) if "TIRE_DEGRADATION_RATE" in car_laps.columns else np.nan

                    rows.append({
                        "NUMBER": car_number,
                        "POSITION": position,
                        "TOTAL_LAPS": total_laps,
                        "AVG_LAP_TIME": avg_lap_time,
                        "TIRE_DEGRADATION": tire_degradation,
                        "NEEDS_STRATEGY_CHANGE": needs_strategy_change,
                        "IS_LEADING": is_leading,
                        "STRATEGY_RISK": 0.3 if needs_strategy_change else 0.1
                    })
                except Exception:
                    continue

        except Exception as e:
            print(f"⚠️ Strategy feature engineering failed: {e}")

        return pd.DataFrame(rows)

    # ------------------------------------------------------------
    # MASTER COMPOSITE FEATURE ENGINEERING - Updated for FirebaseDataLoader
    # ------------------------------------------------------------
    @staticmethod
    def create_composite_features(track_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Create composite features for all tracks using EXACT FirebaseDataLoader structure.
        
        Args:
            track_data: Dict[track_name] -> {
                'pit_data': pd.DataFrame,
                'race_data': pd.DataFrame, 
                'weather_data': pd.DataFrame,
                'telemetry_data': pd.DataFrame
            }
            
        Returns:
            Enhanced data with additional feature columns
        """
        enhanced_data = {}

        for track_name, data_dict in track_data.items():
            try:
                pit_data = data_dict.get("pit_data", pd.DataFrame())
                race_data = data_dict.get("race_data", pd.DataFrame())
                weather_data = data_dict.get("weather_data", pd.DataFrame())
                telemetry_data = data_dict.get("telemetry_data", pd.DataFrame())

                if pit_data.empty:
                    enhanced_data[track_name] = data_dict
                    continue

                # Apply all feature engineering steps
                pit_enhanced = FeatureEngineer.engineer_tire_features(pit_data, telemetry_data)
                pit_enhanced = FeatureEngineer.engineer_fuel_features(pit_enhanced, telemetry_data)
                pit_enhanced = FeatureEngineer.engineer_track_features(track_name, pit_enhanced)
                pit_enhanced = FeatureEngineer.engineer_weather_features(weather_data, pit_enhanced)

                strategy_features = FeatureEngineer.engineer_strategy_features(race_data, pit_enhanced)

                enhanced_data[track_name] = {
                    "pit_data": pit_enhanced,
                    "race_data": race_data,
                    "weather_data": weather_data, 
                    "telemetry_data": telemetry_data,
                    "strategy_features": strategy_features
                }

                print(f"✅ Enhanced features for {track_name}: {len(pit_enhanced)} pit records, "
                      f"{len(strategy_features)} strategy features")

            except Exception as e:
                print(f"⚠️ Feature creation failed for {track_name}: {e}")
                enhanced_data[track_name] = data_dict

        return enhanced_data

    # ----------------------
    # Utility methods - Consistent with DataPreprocessor
    # ----------------------
    @staticmethod
    def _time_to_seconds(time_str: Any) -> float:
        """
        Convert time strings to seconds - Consistent with DataPreprocessor implementation.
        Handles formats like: '1:54.168', '2:13.691'
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
    def validate_feature_engineering(enhanced_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Validate that feature engineering produced expected results.
        """
        validation_results = {}
        
        for track_name, data_dict in enhanced_data.items():
            track_validation = {}
            
            for data_type, df in data_dict.items():
                if df.empty:
                    track_validation[data_type] = {'status': 'empty', 'rows': 0, 'columns': 0}
                else:
                    # Check for engineered features
                    engineered_features = []
                    if data_type == 'pit_data':
                        engineered_features = ['TIRE_DEGRADATION_RATE', 'FUEL_EFFICIENCY_EST', 'TRACK_WEAR_FACTOR']
                    elif data_type == 'strategy_features':
                        engineered_features = ['NEEDS_STRATEGY_CHANGE', 'STRATEGY_RISK']
                    
                    present_features = [feat for feat in engineered_features if feat in df.columns]
                    
                    track_validation[data_type] = {
                        'status': 'enhanced' if present_features else 'basic',
                        'rows': len(df),
                        'columns': len(df.columns),
                        'engineered_features': present_features,
                        'engineered_count': len(present_features)
                    }
            
            validation_results[track_name] = track_validation
        
        # Print validation summary
        print("\n" + "="*60)
        print("FEATURE ENGINEERING VALIDATION SUMMARY")
        print("="*60)
        
        total_enhanced = 0
        for track_name, track_validation in validation_results.items():
            print(f"\n🏁 {track_name.upper()}:")
            for data_type, validation in track_validation.items():
                status_icon = "✅" if validation['status'] == 'enhanced' else "📝"
                print(f"  {status_icon} {data_type}: {validation['rows']} rows, "
                      f"{validation['engineered_count']} engineered features")
                if validation['engineered_features']:
                    print(f"     Features: {validation['engineered_features']}")
            
            if any(v['status'] == 'enhanced' for v in track_validation.values()):
                total_enhanced += 1
        
        print(f"\n📊 Overall: {total_enhanced}/{len(validation_results)} tracks successfully enhanced")
        
        return validation_results