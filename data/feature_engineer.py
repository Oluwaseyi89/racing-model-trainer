import pandas as pd
import numpy as np
from typing import Dict, Iterable, Optional, Tuple, List
from scipy import stats


class FeatureEngineer:
    """
    Robust, modular feature engineering for racing analytics.
    Handles inconsistent telemetry/lap/weather formats safely.

    Usage:
        fe = FeatureEngineer(debug=True)
        lap_features = fe.engineer_tire_features(lap_df, telemetry_df)
        enhanced = fe.create_composite_features(track_data_dict)
    """

    def __init__(self, debug: bool = False):
        self.debug = debug

    # ----------------------
    # Logging helper
    # ----------------------
    def _log(self, msg: str):
        if self.debug:
            print(msg)

    # ----------------------
    # Column normalization
    # ----------------------
    @staticmethod
    def _normalize_column(df: pd.DataFrame, candidates: Iterable[str], new_name: str) -> pd.DataFrame:
        for c in candidates:
            if c in df.columns:
                if c != new_name:
                    df = df.rename(columns={c: new_name})
                return df
        return df

    def _ensure_number_column(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
        df = self._normalize_column(df.copy(),
                                    ["NUMBER", "DRIVER_NUMBER", "vehicle_number", "VEHICLE", "VEHICLE_NUMBER"],
                                    "NUMBER")
        if "NUMBER" in df.columns:
            df["NUMBER"] = pd.to_numeric(df["NUMBER"], errors="coerce")
            df = df.dropna(subset=["NUMBER"])
            return df, "NUMBER"
        return df, None

    def _ensure_lap_number(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._normalize_column(df.copy(), ["LAP_NUMBER", "LAPNUM", "LAP", "lap", "Lap"], "LAP_NUMBER")
        if "LAP_NUMBER" not in df.columns:
            if "NUMBER" in df.columns:
                df["LAP_NUMBER"] = df.groupby("NUMBER").cumcount() + 1
            else:
                df["LAP_NUMBER"] = np.arange(len(df)) + 1
        df["LAP_NUMBER"] = pd.to_numeric(df["LAP_NUMBER"], errors="coerce")
        return df

    # ----------------------
    # Safe math / regression
    # ----------------------
    @staticmethod
    def _safe_regression(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if mask.sum() < 5:
            return None
        x, y = x[mask].astype(float), y[mask].astype(float)
        try:
            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            return float(slope), float(r_value ** 2)
        except Exception:
            return None

    @staticmethod
    def _safe_polyfit_slope(x: np.ndarray, y: np.ndarray) -> Optional[float]:
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if mask.sum() < 5:
            return None
        try:
            return float(np.polyfit(x[mask], y[mask], 1)[0])
        except Exception:
            return None

    @staticmethod
    def _parse_time_to_seconds(val) -> float:
        if pd.isna(val):
            return np.nan
        try:
            return float(val)
        except Exception:
            s = str(val).strip()
            parts = [float(p) for p in s.split(":")]
            if len(parts) == 2:
                return parts[0] * 60 + parts[1]
            elif len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            return np.nan

    # ----------------------
    # Tire features
    # ----------------------
    def engineer_tire_features(self, lap_df: pd.DataFrame, telemetry_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if lap_df is None:
            return pd.DataFrame()
        df, id_col = self._ensure_number_column(lap_df)
        df = self._ensure_lap_number(df)
        if id_col is None:
            self._log("No NUMBER column found; skipping tire features.")
            return df

        # initialize columns
        for col in ["TIRE_DEGRADATION_RATE", "PERFORMANCE_CONSISTENCY", "TIRE_AGE_NONLINEAR"]:
            df[col] = df.get(col, np.nan)

        for car_number in pd.unique(df["NUMBER"].dropna()):
            car_mask = df["NUMBER"] == car_number
            car_laps = df.loc[car_mask].sort_values("LAP_NUMBER")
            if len(car_laps) < 5:
                continue

            # LAP times degradation
            lap_times = None
            if "LAP_TIME_SECONDS" in car_laps:
                lap_times = pd.to_numeric(car_laps["LAP_TIME_SECONDS"], errors="coerce").values
            elif "LAP_TIME" in car_laps:
                lap_times = car_laps["LAP_TIME"].apply(self._parse_time_to_seconds).values
            lap_numbers = pd.to_numeric(car_laps["LAP_NUMBER"], errors="coerce").values

            if lap_times is not None and len(lap_times) >= 8:
                mask_range = (lap_numbers >= 5) & (lap_numbers <= 15)
                if mask_range.sum() >= 5:
                    res = self._safe_regression(lap_numbers, lap_times)
                    if res:
                        slope, r2 = res
                        df.loc[car_mask, "TIRE_DEGRADATION_RATE"] = slope if r2 > 0.4 else 0.0

            # sector degradation
            for sector in ["S1", "S2", "S3", "S1_SECONDS", "S2_SECONDS", "S3_SECONDS"]:
                if sector in car_laps:
                    col_vals = pd.to_numeric(car_laps[sector], errors="coerce")
                    if col_vals.isna().all():
                        try:
                            col_vals = car_laps[sector].apply(self._parse_time_to_seconds)
                        except Exception:
                            continue
                    slope = self._safe_polyfit_slope(car_laps["LAP_NUMBER"].values, col_vals.values)
                    if slope is not None:
                        out_col = sector if sector.endswith("_DEGRADATION") else f"{sector}_DEGRADATION"
                        df.loc[car_mask, out_col] = slope

            # performance consistency
            if "LAP_TIME_SECONDS" in car_laps:
                df.loc[car_mask, "PERFORMANCE_CONSISTENCY"] = float(np.nanstd(pd.to_numeric(car_laps["LAP_TIME_SECONDS"], errors="coerce")))
            # non-linear tire age
            df.loc[car_mask, "TIRE_AGE_NONLINEAR"] = np.log1p(pd.to_numeric(car_laps["LAP_NUMBER"], errors="coerce")).fillna(0).values * 0.5

        # Merge telemetry tire features
        if telemetry_df is not None and not telemetry_df.empty:
            df = self._add_telemetry_features(df, telemetry_df)

        return df

    # ----------------------
    # Telemetry generic aggregation
    # ----------------------
    def _add_telemetry_features(self, lap_df: pd.DataFrame, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        df = lap_df.copy()
        t = telemetry_df.copy()

        # Normalize keys
        t = self._normalize_column(t, ["vehicle_number", "vehicle_id", "NUMBER", "VEHICLE"], "NUMBER")
        t = self._normalize_column(t, ["lap", "lap_number", "LAP", "LAP_NUMBER"], "LAP_NUMBER")
        t["NUMBER"] = pd.to_numeric(t["NUMBER"], errors="coerce")
        t["LAP_NUMBER"] = pd.to_numeric(t["LAP_NUMBER"], errors="coerce")

        # example feature map: col -> aggregation
        feature_map = {
            "LATERAL_ACCEL": ["mean", "std"],
            "BRAKE_PRESSURE_FRONT": ["mean"],
            "BRAKE_PRESSURE_REAR": ["mean"],
            "STEERING_ANGLE": ["sum"]
        }

        # find first matching column for each
        agg_cols = {col: aggs for col, aggs in feature_map.items() if any(c in t.columns for c in [col, col.lower()])}
        if not agg_cols:
            return df

        # aggregate
        telemetry_agg = t.groupby(["NUMBER", "LAP_NUMBER"]).agg(agg_cols)
        telemetry_agg.columns = ["_".join(map(str, c)) for c in telemetry_agg.columns]
        telemetry_agg = telemetry_agg.reset_index()
        return df.merge(telemetry_agg, on=["NUMBER", "LAP_NUMBER"], how="left")

    # ----------------------
    # Other feature modules
    # ----------------------
    def engineer_fuel_features(self, lap_df: pd.DataFrame, telemetry_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df = lap_df.copy()
        df, _ = self._ensure_number_column(df)
        df = self._ensure_lap_number(df)

        if "LAP_TIME_SECONDS" in df.columns:
            df["FUEL_EFFICIENCY_EST"] = 1.0 / (pd.to_numeric(df["LAP_TIME_SECONDS"], errors="coerce").fillna(1.0) + 0.1)
        else:
            df["FUEL_EFFICIENCY_EST"] = np.nan

        if telemetry_df is not None and not telemetry_df.empty:
            t = telemetry_df.copy()
            t = self._normalize_column(t, ["vehicle_number", "vehicle_id", "NUMBER", "VEHICLE"], "NUMBER")
            t = self._normalize_column(t, ["lap", "lap_number", "LAP", "LAP_NUMBER"], "LAP_NUMBER")
            throttle_col = self._first_existing_column(t, ["THROTTLE_POSITION", "aps", "throttle", "THROTTLE"])
            if throttle_col and "NUMBER" in t and "LAP_NUMBER" in t:
                throttle_stats = t.groupby(["NUMBER", "LAP_NUMBER"])[throttle_col].agg(["mean", "std"]).reset_index()
                throttle_stats = throttle_stats.rename(columns={"mean": "THROTTLE_MEAN", "std": "THROTTLE_STD"})
                df = df.merge(throttle_stats, on=["NUMBER", "LAP_NUMBER"], how="left")
        return df

    @staticmethod
    def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # TODO: implement track/weather/strategy features using same pattern

    # ----------------------
    # Composite master
    # ----------------------
    def create_composite_features(self, processed_data: Dict) -> Dict:
        enhanced: Dict = {}
        for track_name, data in processed_data.items():
            try:
                lap = data.get("lap_data", pd.DataFrame())
                race = data.get("race_data", pd.DataFrame())
                weather = data.get("weather_data", pd.DataFrame())
                telemetry = data.get("telemetry_data", pd.DataFrame())

                if lap.empty:
                    enhanced[track_name] = data
                    continue

                lap = self.engineer_tire_features(lap, telemetry)
                lap = self.engineer_fuel_features(lap, telemetry)
                # add track/weather/strategy features similarly
                enhanced[track_name] = {**data, "lap_data": lap}
            except Exception as e:
                self._log(f"⚠️ Feature creation failed for {track_name}: {e}")
                enhanced[track_name] = data
        return enhanced





















# import pandas as pd
# import numpy as np
# from typing import Dict, Iterable, Optional, Tuple
# from scipy import stats


# class FeatureEngineer:
#     """
#     Feature engineering for racing analytics.
#     Safe for inconsistent telemetry formats, missing laps,
#     partial weather data, and incomplete race results.

#     Public API (unchanged):
#       - engineer_tire_features(lap_data, telemetry_data) -> pd.DataFrame
#       - engineer_fuel_features(lap_data, telemetry_data) -> pd.DataFrame
#       - engineer_track_features(track_name, lap_data) -> pd.DataFrame
#       - engineer_weather_features(weather_data, lap_data) -> pd.DataFrame
#       - engineer_strategy_features(race_data, lap_data) -> pd.DataFrame
#       - create_composite_features(processed_data) -> Dict
#     """

#     # ----------------------
#     # Helper / normalization
#     # ----------------------
#     @staticmethod
#     def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
#         for c in candidates:
#             if c in df.columns:
#                 return c
#         return None

#     @staticmethod
#     def _ensure_number_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
#         """
#         Guarantee that dataframe has a numeric 'NUMBER' column (or a variant).
#         Returns (df_copy, column_name_used_or_none).
#         """
#         df = df.copy()
#         candidates = ["NUMBER", "DRIVER_NUMBER", "vehicle_number", "VEHICLE", "VEHICLE_NUMBER"]
#         col = FeatureEngineer._first_existing_column(df, candidates)
#         if col is None:
#             # No numeric id column found — try to infer from columns that look numeric
#             for c in df.columns:
#                 if c.lower().startswith("num") or c.lower().startswith("driver"):
#                     col = c
#                     break
#         if col:
#             # Convert to numeric where possible
#             df[col] = pd.to_numeric(df[col], errors="coerce")
#             df = df.dropna(subset=[col]) if df[col].dtype.kind in "fiu" else df
#             # rename to NUMBER for uniform downstream use (but keep original too)
#             if col != "NUMBER":
#                 df = df.rename(columns={col: "NUMBER"})
#                 return df, "NUMBER"
#             return df, col
#         return df, None

#     @staticmethod
#     def _ensure_lap_number(df: pd.DataFrame) -> pd.DataFrame:
#         df = df.copy()
#         candidates = ["LAP_NUMBER", "LAPNUM", "LAP", "lap", "Lap"]
#         col = FeatureEngineer._first_existing_column(df, candidates)
#         if col and col != "LAP_NUMBER":
#             df = df.rename(columns={col: "LAP_NUMBER"})
#         if "LAP_NUMBER" not in df.columns:
#             # create a lap counter per NUMBER
#             if "NUMBER" in df.columns:
#                 df["LAP_NUMBER"] = df.groupby("NUMBER").cumcount() + 1
#             else:
#                 df["LAP_NUMBER"] = np.arange(len(df)) + 1
#         df["LAP_NUMBER"] = pd.to_numeric(df["LAP_NUMBER"], errors="coerce")
#         return df

#     @staticmethod
#     def _safe_regression(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
#         """
#         Run a robust linear regression if inputs are sane.
#         Returns (slope, r_squared) or None on failure/safety checks.
#         """
#         try:
#             # drop NaNs and ensure floats
#             mask = (~np.isnan(x)) & (~np.isnan(y))
#             if mask.sum() < 5:
#                 return None
#             xv = x[mask].astype(float)
#             yv = y[mask].astype(float)
#             if xv.size < 5:
#                 return None
#             slope, intercept, r_value, p_value, std_err = stats.linregress(xv, yv)
#             r2 = float(r_value ** 2)
#             return float(slope), r2
#         except Exception:
#             return None

#     @staticmethod
#     def _safe_polyfit_slope(x: np.ndarray, y: np.ndarray) -> Optional[float]:
#         """
#         Try to get first-degree polynomial slope with safety checks.
#         """
#         try:
#             mask = (~np.isnan(x)) & (~np.isnan(y))
#             if mask.sum() < 5:
#                 return None
#             xv = x[mask].astype(float)
#             yv = y[mask].astype(float)
#             if xv.size < 5:
#                 return None
#             coeffs = np.polyfit(xv, yv, 1)
#             slope = float(coeffs[0])
#             return slope
#         except Exception:
#             return None

#     # ------------------------------------------------------------
#     # TIRE FEATURES
#     # ------------------------------------------------------------
#     @staticmethod
#     def engineer_tire_features(lap_data: pd.DataFrame,
#                                telemetry_data: pd.DataFrame) -> pd.DataFrame:

#         if lap_data is None:
#             return pd.DataFrame()

#         # work on a copy
#         lap_df = lap_data.copy()

#         # Normalize id and lap columns
#         lap_df, id_col = FeatureEngineer._ensure_number_column(lap_df)
#         lap_df = FeatureEngineer._ensure_lap_number(lap_df)

#         if id_col is None or "NUMBER" not in lap_df.columns:
#             # nothing to do, return a safe copy
#             return lap_df

#         # initialize columns so merges don't fail later
#         lap_df["TIRE_DEGRADATION_RATE"] = lap_df.get("TIRE_DEGRADATION_RATE", np.nan)
#         lap_df["PERFORMANCE_CONSISTENCY"] = lap_df.get("PERFORMANCE_CONSISTENCY", np.nan)
#         lap_df["TIRE_AGE_NONLINEAR"] = lap_df.get("TIRE_AGE_NONLINEAR", np.nan)

#         try:
#             for car_number in pd.unique(lap_df["NUMBER"].dropna()):
#                 car_mask = lap_df["NUMBER"] == car_number
#                 car_laps = lap_df.loc[car_mask].sort_values("LAP_NUMBER")

#                 # require some laps to compute meaningful degradation stats
#                 if car_laps.shape[0] < 5:
#                     continue

#                 # attempt to use LAP_TIME_SECONDS or fallback to LAP_TIME / FL_TIME
#                 if "LAP_TIME_SECONDS" in car_laps.columns:
#                     lap_times = pd.to_numeric(car_laps["LAP_TIME_SECONDS"], errors="coerce").values
#                     lap_numbers = pd.to_numeric(car_laps["LAP_NUMBER"], errors="coerce").values
#                 else:
#                     # fallback: check if LAP_TIME exists and can be parsed to seconds (str like 1:39.123)
#                     if "LAP_TIME" in car_laps.columns:
#                         # try parse mm:ss.xxx or hh:mm:ss
#                         lap_times = car_laps["LAP_TIME"].apply(FeatureEngineer._parse_time_to_seconds).values
#                         lap_numbers = pd.to_numeric(car_laps["LAP_NUMBER"], errors="coerce").values
#                     else:
#                         lap_times = np.array([])
#                         lap_numbers = np.array([])

#                 # LAP time degradation: consider laps 5..15 if possible
#                 if lap_times.size >= 8:
#                     mask_range = (lap_numbers >= 5) & (lap_numbers <= 15)
#                     if mask_range.sum() >= 5:
#                         res = FeatureEngineer._safe_regression(lap_numbers, lap_times)
#                         if res is not None:
#                             slope, r2 = res
#                             # only set degradation if regression is reasonably predictive
#                             if r2 > 0.4:
#                                 lap_df.loc[car_mask, "TIRE_DEGRADATION_RATE"] = slope
#                             else:
#                                 lap_df.loc[car_mask, "TIRE_DEGRADATION_RATE"] = 0.0

#                 # Sector degradation
#                 for sector in ["S1_SECONDS", "S2_SECONDS", "S3_SECONDS", "S1", "S2", "S3"]:
#                     if sector in car_laps.columns:
#                         # try numeric conversion; if values contain mm:ss, fallback to parse_time
#                         col_vals = pd.to_numeric(car_laps[sector], errors="coerce")
#                         if col_vals.isna().all():
#                             # fallback parse strings if necessary
#                             try:
#                                 col_vals = car_laps[sector].apply(FeatureEngineer._parse_time_to_seconds)
#                             except Exception:
#                                 continue
#                         slope = FeatureEngineer._safe_polyfit_slope(car_laps["LAP_NUMBER"].values, col_vals.values)
#                         if slope is not None:
#                             out_col = sector if sector.endswith("_DEGRADATION") else f"{sector}_DEGRADATION"
#                             lap_df.loc[car_mask, out_col] = slope

#                 # consistency
#                 if "LAP_TIME_SECONDS" in car_laps.columns:
#                     try:
#                         lap_df.loc[car_mask, "PERFORMANCE_CONSISTENCY"] = \
#                             float(np.nanstd(pd.to_numeric(car_laps["LAP_TIME_SECONDS"], errors="coerce")))
#                     except Exception:
#                         lap_df.loc[car_mask, "PERFORMANCE_CONSISTENCY"] = np.nan

#                 # non-linear tire age
#                 try:
#                     lap_df.loc[car_mask, "TIRE_AGE_NONLINEAR"] = np.log1p(pd.to_numeric(car_laps["LAP_NUMBER"], errors="coerce")).fillna(0).values * 0.5
#                 except Exception:
#                     lap_df.loc[car_mask, "TIRE_AGE_NONLINEAR"] = np.nan

#         except Exception as e:
#             # never let one failure stop the pipeline
#             print(f"⚠️ Tire feature engineering failed: {e}")

#         # Merge telemetry tire features if telemetry provided
#         if telemetry_data is not None and not telemetry_data.empty:
#             try:
#                 lap_df = FeatureEngineer._add_telemetry_tire_features(lap_df, telemetry_data)
#             except Exception as e:
#                 print(f"⚠️ Telemetry tire merging failed at top-level: {e}")

#         return lap_df

#     @staticmethod
#     def _add_telemetry_tire_features(lap_df: pd.DataFrame,
#                                      telemetry_df: pd.DataFrame) -> pd.DataFrame:

#         df = lap_df.copy()

#         # Accept multiple naming conventions in telemetry (vehicle_number, NUMBER, VEHICLE)
#         telemetry = telemetry_df.copy()
#         telemetry_cols = telemetry.columns.str.lower().tolist()

#         # Normalize telemetry column names we will use
#         # find vehicle id column in telemetry
#         vehicle_col = FeatureEngineer._first_existing_column(telemetry, ["vehicle_number", "vehicle_id", "NUMBER", "number", "VEHICLE"])
#         lap_col = FeatureEngineer._first_existing_column(telemetry, ["lap", "lap_number", "LAP", "LAP_NUMBER"])

#         if vehicle_col is None or lap_col is None:
#             return df

#         # rename normalized cols
#         telemetry = telemetry.rename(columns={vehicle_col: "vehicle_number", lap_col: "lap"})

#         # Try to cast useful telemetry columns to numeric if present
#         telemetry["vehicle_number"] = pd.to_numeric(telemetry["vehicle_number"], errors="coerce")
#         telemetry["lap"] = pd.to_numeric(telemetry["lap"], errors="coerce")

#         # prefer the mapped names from preprocessor, otherwise attempt given ones
#         lat_col = FeatureEngineer._first_existing_column(telemetry, ["LATERAL_ACCEL", "accy_can", "accy", "lat_acc"])
#         brake_f = FeatureEngineer._first_existing_column(telemetry, ["BRAKE_PRESSURE_FRONT", "pbrake_f", "brake_f"])
#         brake_r = FeatureEngineer._first_existing_column(telemetry, ["BRAKE_PRESSURE_REAR", "pbrake_r", "brake_r"])
#         steer_col = FeatureEngineer._first_existing_column(telemetry, ["STEERING_ANGLE", "steering_angle", "Steering_Angle"])

#         # Build aggregated telemetry per (vehicle_number, lap)
#         agg_cols = {}
#         if lat_col:
#             agg_cols[lat_col] = ["mean", "std"]
#         if brake_f:
#             agg_cols[brake_f] = ["mean"]
#         if brake_r:
#             agg_cols[brake_r] = ["mean"]
#         if steer_col:
#             agg_cols[steer_col] = ["sum"]

#         if not agg_cols:
#             # nothing useful to aggregate
#             return df

#         # perform aggregation
#         try:
#             telemetry_agg = telemetry.groupby(["vehicle_number", "lap"]).agg(agg_cols)
#             # flatten columns
#             telemetry_agg.columns = ["_".join(map(str, c)).strip() for c in telemetry_agg.columns.values]
#             telemetry_agg = telemetry_agg.reset_index()

#             # rename aggregated columns to friendly names
#             rename_map = {}
#             if lat_col:
#                 # find mean column name created
#                 lat_mean_col = f"{lat_col}_mean"
#                 rename_map[lat_mean_col] = "LATERAL_G_MEAN"
#                 # std
#                 lat_std_col = f"{lat_col}_std"
#                 rename_map[lat_std_col] = "LATERAL_G_STD"
#             if brake_f:
#                 rename_map[f"{brake_f}_mean"] = "BRAKE_PRESSURE_FRONT_MEAN"
#             if brake_r:
#                 rename_map[f"{brake_r}_mean"] = "BRAKE_PRESSURE_REAR_MEAN"
#             if steer_col:
#                 rename_map[f"{steer_col}_sum"] = "STEERING_ACTIVITY_SUM"

#             telemetry_agg = telemetry_agg.rename(columns=rename_map)

#             # ensure merge keys match lap_df convention
#             # lap_df uses 'NUMBER' and 'LAP_NUMBER'
#             if "NUMBER" not in df.columns:
#                 df, _ = FeatureEngineer._ensure_number_column(df)
#             if "LAP_NUMBER" not in df.columns:
#                 df = FeatureEngineer._ensure_lap_number(df)

#             telemetry_agg = telemetry_agg.rename(columns={"vehicle_number": "NUMBER", "lap": "LAP_NUMBER"})
#             telemetry_agg["NUMBER"] = pd.to_numeric(telemetry_agg["NUMBER"], errors="coerce")
#             telemetry_agg["LAP_NUMBER"] = pd.to_numeric(telemetry_agg["LAP_NUMBER"], errors="coerce")

#             # merge
#             df = df.merge(telemetry_agg, on=["NUMBER", "LAP_NUMBER"], how="left")

#         except Exception as e:
#             print(f"⚠️ _add_telemetry_tire_features aggregation failed: {e}")

#         return df

#     # ------------------------------------------------------------
#     # FUEL FEATURES
#     # ------------------------------------------------------------
#     @staticmethod
#     def engineer_fuel_features(lap_data: pd.DataFrame,
#                                telemetry_data: pd.DataFrame) -> pd.DataFrame:

#         if lap_data is None:
#             return pd.DataFrame()

#         df = lap_data.copy()
#         df, _ = FeatureEngineer._ensure_number_column(df)
#         df = FeatureEngineer._ensure_lap_number(df)

#         try:
#             if "LAP_TIME_SECONDS" in df.columns:
#                 # avoid division by zero / inf
#                 df["FUEL_EFFICIENCY_EST"] = 1.0 / (pd.to_numeric(df["LAP_TIME_SECONDS"], errors="coerce").fillna(1.0) + 0.1)
#             else:
#                 df["FUEL_EFFICIENCY_EST"] = np.nan

#             # telemetry throttle features
#             if telemetry_data is not None and not telemetry_data.empty:
#                 t = telemetry_data.copy()
#                 # normalize throttle column names and keys
#                 throttle_col = FeatureEngineer._first_existing_column(t, ["THROTTLE_POSITION", "aps", "throttle", "THROTTLE"])
#                 vehicle_col = FeatureEngineer._first_existing_column(t, ["vehicle_number", "vehicle_id", "NUMBER", "VEHICLE"])
#                 lap_col = FeatureEngineer._first_existing_column(t, ["lap", "LAP", "lap_number", "LAP_NUMBER"])
#                 if throttle_col and vehicle_col and lap_col:
#                     t = t.rename(columns={vehicle_col: "vehicle_number", lap_col: "lap", throttle_col: "THROTTLE_POSITION"})
#                     t["vehicle_number"] = pd.to_numeric(t["vehicle_number"], errors="coerce")
#                     t["lap"] = pd.to_numeric(t["lap"], errors="coerce")
#                     throttle_stats = t.groupby(["vehicle_number", "lap"])["THROTTLE_POSITION"].agg(["mean", "std"]).reset_index()
#                     throttle_stats.columns = ["NUMBER", "LAP_NUMBER", "THROTTLE_MEAN", "THROTTLE_STD"]
#                     throttle_stats["NUMBER"] = pd.to_numeric(throttle_stats["NUMBER"], errors="coerce")
#                     throttle_stats["LAP_NUMBER"] = pd.to_numeric(throttle_stats["LAP_NUMBER"], errors="coerce")
#                     # ensure df has NUMBER and LAP_NUMBER
#                     if "NUMBER" not in df.columns:
#                         df, _ = FeatureEngineer._ensure_number_column(df)
#                     if "LAP_NUMBER" not in df.columns:
#                         df = FeatureEngineer._ensure_lap_number(df)
#                     df = df.merge(throttle_stats, on=["NUMBER", "LAP_NUMBER"], how="left")

#         except Exception as e:
#             print(f"⚠️ Fuel engineering failed: {e}")

#         return df

#     # ------------------------------------------------------------
#     # TRACK FEATURES
#     # ------------------------------------------------------------
#     @staticmethod
#     def engineer_track_features(track_name: str,
#                                 lap_data: pd.DataFrame) -> pd.DataFrame:

#         if lap_data is None:
#             return pd.DataFrame()

#         df = lap_data.copy()

#         wear_map = {
#             "sebring": 0.9, "barber": 0.85, "sonoma": 0.8,
#             "road-america": 0.7, "vir": 0.75, "cota": 0.6,
#             "indianapolis": 0.5
#         }

#         try:
#             if track_name:
#                 df["TRACK_WEAR_FACTOR"] = wear_map.get(str(track_name).lower(), 0.7)
#             else:
#                 df["TRACK_WEAR_FACTOR"] = 0.7

#             # overtaking potential computed robustly
#             if "KPH" in df.columns:
#                 try:
#                     kph = pd.to_numeric(df["KPH"], errors="coerce")
#                     mean_speed = float(kph.mean()) if not kph.dropna().empty else 0.0
#                     var_speed = float(kph.var()) if not kph.dropna().empty else 0.0
#                     df["OVERTAKING_POTENTIAL"] = min(1.0, (var_speed / (mean_speed + 1e-6)) * 10) if mean_speed > 0 else 0.1
#                 except Exception:
#                     df["OVERTAKING_POTENTIAL"] = 0.1
#             else:
#                 df["OVERTAKING_POTENTIAL"] = 0.1

#         except Exception as e:
#             print(f"⚠️ Track feature engineering failed: {e}")

#         return df

#     # ------------------------------------------------------------
#     # WEATHER FEATURES
#     # ------------------------------------------------------------
#     @staticmethod
#     def engineer_weather_features(weather_data: pd.DataFrame,
#                                   lap_data: pd.DataFrame) -> pd.DataFrame:

#         if lap_data is None:
#             return pd.DataFrame()

#         df = lap_data.copy()

#         try:
#             if weather_data is None or weather_data.empty:
#                 return df

#             # coerce common columns
#             weather = weather_data.copy()
#             # Accept TIME_UTC_SECONDS or TIME_UTC_STR etc; not modifying timestamp here, just use numeric cols
#             for col in ["AIR_TEMP", "TRACK_TEMP", "HUMIDITY", "PRESSURE", "WIND_SPEED", "RAIN"]:
#                 if col in weather.columns:
#                     weather[col] = pd.to_numeric(weather[col], errors="coerce")

#             if "AIR_TEMP" in weather.columns:
#                 df["TEMP_IMPACT"] = (float(weather["AIR_TEMP"].mean(skipna=True)) - 25.0) * 0.03

#             if "TRACK_TEMP" in weather.columns:
#                 df["TRACK_TEMP_IMPACT"] = (float(weather["TRACK_TEMP"].mean(skipna=True)) - 35.0) * 0.02

#             if "RAIN" in weather.columns:
#                 df["RAIN_IMPACT"] = float(weather["RAIN"].max(skipna=True)) * 1.5

#         except Exception as e:
#             print(f"⚠️ Weather feature engineering failed: {e}")

#         return df

#     # ------------------------------------------------------------
#     # STRATEGY FEATURES
#     # ------------------------------------------------------------
#     @staticmethod
#     def engineer_strategy_features(race_data: pd.DataFrame,
#                                    lap_data: pd.DataFrame) -> pd.DataFrame:
#         """
#         Returns a small dataframe with per-car strategy features.
#         Keeps method signature and returns pd.DataFrame.
#         """

#         # validate inputs
#         if race_data is None or lap_data is None:
#             return pd.DataFrame()

#         # Normalize id columns
#         race_df = race_data.copy()
#         lap_df = lap_data.copy()

#         race_df, _ = FeatureEngineer._ensure_number_column(race_df)
#         lap_df, _ = FeatureEngineer._ensure_number_column(lap_df)
#         lap_df = FeatureEngineer._ensure_lap_number(lap_df)

#         if "NUMBER" not in race_df.columns or "NUMBER" not in lap_df.columns:
#             # not enough identifiers to build strategy features
#             print("⚠️ Strategy engineering failed: missing 'NUMBER' after normalization")
#             return pd.DataFrame()

#         rows = []
#         try:
#             unique_numbers = pd.unique(race_df["NUMBER"].dropna())
#             for car_number in unique_numbers:
#                 try:
#                     car_race = race_df[race_df["NUMBER"] == car_number]
#                     if car_race.empty:
#                         continue
#                     car_row = car_race.iloc[0]
#                     car_laps = lap_df[lap_df["NUMBER"] == car_number].sort_values("LAP_NUMBER")
#                     if car_laps.shape[0] < 3:
#                         # not enough lap history to infer strategy
#                         continue

#                     # position safely
#                     pos = car_row.get("POSITION", car_row.get("POS", np.nan))
#                     try:
#                         pos = int(pos) if not pd.isna(pos) else np.nan
#                     except Exception:
#                         pos = np.nan

#                     total_laps = int(car_laps["LAP_NUMBER"].max()) if "LAP_NUMBER" in car_laps.columns else car_laps.shape[0]

#                     # simple heuristics: needs_strategy_change if position > 10
#                     needs_strategy = 1 if (not pd.isna(pos) and pos > 10) else 0

#                     # add fuel/tire related rollups if available
#                     avg_lap = float(pd.to_numeric(car_laps.get("LAP_TIME_SECONDS", pd.Series([])), errors="coerce").mean(skipna=True)) if "LAP_TIME_SECONDS" in car_laps.columns else np.nan
#                     tire_deg = float(pd.to_numeric(car_laps.get("TIRE_DEGRADATION_RATE", pd.Series([np.nan])), errors="coerce").mean(skipna=True)) if "TIRE_DEGRADATION_RATE" in car_laps.columns else np.nan

#                     rows.append({
#                         "car_number": car_number,
#                         "position": pos,
#                         "total_laps": total_laps,
#                         "avg_lap_time": avg_lap,
#                         "mean_tire_deg": tire_deg,
#                         "needs_strategy_change": needs_strategy
#                     })
#                 except Exception:
#                     # per-car failure should not stop others
#                     continue

#         except Exception as e:
#             print(f"⚠️ Strategy engineering failed: {e}")

#         return pd.DataFrame(rows)

#     # ------------------------------------------------------------
#     # MASTER COMPOSITE FEATURE ENGINEERING
#     # ------------------------------------------------------------
#     @staticmethod
#     def create_composite_features(processed_data: Dict) -> Dict:
#         """
#         processed_data: dict[track_name] -> {
#             'lap_data': pd.DataFrame,
#             'race_data': pd.DataFrame,
#             'weather_data': pd.DataFrame,
#             'telemetry_data': pd.DataFrame
#         }
#         """
#         enhanced: Dict = {}

#         for track_name, data in processed_data.items():
#             try:
#                 lap = data.get("lap_data", pd.DataFrame()) if isinstance(data, dict) else pd.DataFrame()
#                 race = data.get("race_data", pd.DataFrame()) if isinstance(data, dict) else pd.DataFrame()
#                 weather = data.get("weather_data", pd.DataFrame()) if isinstance(data, dict) else pd.DataFrame()
#                 telemetry = data.get("telemetry_data", pd.DataFrame()) if isinstance(data, dict) else pd.DataFrame()

#                 if lap is None or getattr(lap, "empty", True):
#                     # keep original if nothing to enhance
#                     enhanced[track_name] = data
#                     continue

#                 lap = FeatureEngineer.engineer_tire_features(lap, telemetry)
#                 lap = FeatureEngineer.engineer_fuel_features(lap, telemetry)
#                 lap = FeatureEngineer.engineer_track_features(track_name, lap)
#                 lap = FeatureEngineer.engineer_weather_features(weather, lap)

#                 strategy = FeatureEngineer.engineer_strategy_features(race, lap)

#                 enhanced[track_name] = {
#                     "lap_data": lap,
#                     "race_data": race,
#                     "weather_data": weather,
#                     "telemetry_data": telemetry,
#                     "strategy_features": strategy
#                 }

#             except Exception as e:
#                 print(f"⚠️ Feature creation failed for {track_name}: {e}")
#                 enhanced[track_name] = data

#         return enhanced

#     # ----------------------
#     # Utility parsing helpers
#     # ----------------------
#     @staticmethod
#     def _parse_time_to_seconds(val) -> float:
#         """
#         Robust parse: accepts numeric seconds, mm:ss.sss, hh:mm:ss, or returns NaN.
#         """
#         if pd.isna(val):
#             return np.nan
#         # if already numeric
#         try:
#             return float(val)
#         except Exception:
#             pass

#         s = str(val).strip()
#         # formats: mm:ss.sss or hh:mm:ss(.sss)
#         if ":" in s:
#             parts = s.split(":")
#             try:
#                 parts = [p.strip() for p in parts]
#                 parts = [float(p) for p in parts]
#                 if len(parts) == 2:
#                     return parts[0] * 60.0 + parts[1]
#                 elif len(parts) == 3:
#                     return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
#             except Exception:
#                 return np.nan
#         # fallback
#         try:
#             return float(s)
#         except Exception:
#             return np.nan

























# import pandas as pd
# import numpy as np
# from typing import Dict
# from scipy import stats


# class FeatureEngineer:
#     """
#     Feature engineering for racing analytics.
#     Safe for inconsistent telemetry formats, missing laps,
#     partial weather data, and incomplete race results.
#     """

#     # ------------------------------------------------------------
#     # TIRE FEATURES
#     # ------------------------------------------------------------
#     @staticmethod
#     def engineer_tire_features(lap_data: pd.DataFrame,
#                                telemetry_data: pd.DataFrame) -> pd.DataFrame:

#         if lap_data.empty:
#             return lap_data.copy()

#         df = lap_data.copy()

#         # Ensure core fields exist
#         if "NUMBER" not in df.columns:
#             return df

#         if "LAP_NUMBER" not in df.columns:
#             df["LAP_NUMBER"] = df.groupby("NUMBER").cumcount() + 1

#         # -----------------------------
#         # Per-car tire degradation
#         # -----------------------------
#         try:
#             for car_number in df["NUMBER"].dropna().unique():
#                 car_mask = df["NUMBER"] == car_number
#                 car_laps = df[car_mask].sort_values("LAP_NUMBER")

#                 if len(car_laps) < 5:
#                     continue

#                 # LAP TIME DEGRADATION RATE
#                 if "LAP_TIME_SECONDS" in car_laps.columns:
#                     lap_times = car_laps["LAP_TIME_SECONDS"].values
#                     lap_numbers = car_laps["LAP_NUMBER"].values

#                     if len(lap_times) >= 8:
#                         mask = (lap_numbers >= 5) & (lap_numbers <= 15)
#                         if mask.sum() >= 5:
#                             slope, _, r_value, _, _ = stats.linregress(
#                                 lap_numbers[mask], lap_times[mask]
#                             )
#                             df.loc[car_mask, "TIRE_DEGRADATION_RATE"] = (
#                                 slope if r_value**2 > 0.5 else 0.0
#                             )

#                 # SECTOR DEGRADATION
#                 for sector in ["S1_SECONDS", "S2_SECONDS", "S3_SECONDS"]:
#                     if sector in car_laps.columns:
#                         sec_vals = car_laps[sector].values
#                         if len(sec_vals) >= 5 and not np.all(np.isnan(sec_vals)):
#                             try:
#                                 slope = np.polyfit(
#                                     car_laps["LAP_NUMBER"].values[:len(sec_vals)],
#                                     sec_vals,
#                                     1
#                                 )[0]
#                                 df.loc[car_mask, f"{sector}_DEGRADATION"] = slope
#                             except:
#                                 df.loc[car_mask, f"{sector}_DEGRADATION"] = 0.0

#                 # CONSISTENCY
#                 if "LAP_TIME_SECONDS" in car_laps.columns:
#                     df.loc[car_mask, "PERFORMANCE_CONSISTENCY"] = \
#                         np.nanstd(car_laps["LAP_TIME_SECONDS"])

#                 # NON-LINEAR TIRE AGE
#                 df.loc[car_mask, "TIRE_AGE_NONLINEAR"] = \
#                     np.log1p(car_laps["LAP_NUMBER"]) * 0.5

#         except Exception as e:
#             print(f"⚠️ Tire feature engineering failed: {e}")

#         # Add telemetry-based features
#         if not telemetry_data.empty:
#             df = FeatureEngineer._add_telemetry_tire_features(df, telemetry_data)

#         return df

#     @staticmethod
#     def _add_telemetry_tire_features(lap_df: pd.DataFrame,
#                                      telemetry_df: pd.DataFrame) -> pd.DataFrame:

#         df = lap_df.copy()

#         # Required telemetry structure
#         if not {"vehicle_number", "lap"}.issubset(telemetry_df.columns):
#             return df

#         telemetry_features = []

#         try:
#             for (car_number, lap), lap_tel in telemetry_df.groupby(
#                     ["vehicle_number", "lap"]):

#                 if len(lap_tel) < 10:
#                     continue

#                 telemetry_features.append({
#                     "NUMBER": car_number,
#                     "LAP_NUMBER": lap,
#                     "LATERAL_G_MEAN":
#                         lap_tel.get("LATERAL_ACCEL", pd.Series([0])).abs().mean(),
#                     "LATERAL_G_VARIANCE":
#                         lap_tel.get("LATERAL_ACCEL", pd.Series([0])).abs().std(),
#                     "BRAKE_INTENSITY":
#                         lap_tel.get("TOTAL_BRAKE_PRESSURE", pd.Series([0])).mean(),
#                     "STEERING_ACTIVITY":
#                         lap_tel.get("STEERING_ANGLE", pd.Series([0])).diff().abs().sum()
#                 })

#             if telemetry_features:
#                 tdf = pd.DataFrame(telemetry_features)
#                 df = df.merge(tdf, on=["NUMBER", "LAP_NUMBER"], how="left")

#         except Exception as e:
#             print(f"⚠️ Telemetry tire feature merge failed: {e}")

#         return df

#     # ------------------------------------------------------------
#     # FUEL FEATURES
#     # ------------------------------------------------------------
#     @staticmethod
#     def engineer_fuel_features(lap_data: pd.DataFrame,
#                                telemetry_data: pd.DataFrame) -> pd.DataFrame:

#         if lap_data.empty:
#             return lap_data.copy()

#         df = lap_data.copy()

#         try:
#             if "LAP_TIME_SECONDS" in df.columns:
#                 df["FUEL_EFFICIENCY_EST"] = 1 / (df["LAP_TIME_SECONDS"] + 0.1)

#             # Telemetry-based fuel usage
#             if not telemetry_data.empty and "THROTTLE_POSITION" in telemetry_data.columns:
#                 throttle_stats = telemetry_data.groupby(
#                     ["vehicle_number", "lap"]
#                 )["THROTTLE_POSITION"].agg(["mean", "std"]).reset_index()

#                 throttle_stats.columns = [
#                     "NUMBER", "LAP_NUMBER",
#                     "THROTTLE_MEAN", "THROTTLE_STD"
#                 ]

#                 df = df.merge(throttle_stats, on=["NUMBER", "LAP_NUMBER"], how="left")

#         except Exception as e:
#             print(f"⚠️ Fuel engineering failed: {e}")

#         return df

#     # ------------------------------------------------------------
#     # TRACK FEATURES
#     # ------------------------------------------------------------
#     @staticmethod
#     def engineer_track_features(track_name: str,
#                                 lap_data: pd.DataFrame) -> pd.DataFrame:

#         if lap_data.empty:
#             return lap_data.copy()

#         df = lap_data.copy()

#         wear_map = {
#             "sebring": 0.9, "barber": 0.85, "sonoma": 0.8,
#             "road-america": 0.7, "vir": 0.75, "cota": 0.6,
#             "indianapolis": 0.5
#         }

#         try:
#             df["TRACK_WEAR_FACTOR"] = wear_map.get(track_name.lower(), 0.7)

#             if "KPH" in df.columns:
#                 mean_speed = df["KPH"].mean()
#                 if mean_speed > 0:
#                     df["OVERTAKING_POTENTIAL"] = min(
#                         1.0, (df["KPH"].var() / mean_speed) * 10
#                     )
#                 else:
#                     df["OVERTAKING_POTENTIAL"] = 0.1

#         except Exception as e:
#             print(f"⚠️ Track feature engineering failed: {e}")

#         return df

#     # ------------------------------------------------------------
#     # WEATHER FEATURES
#     # ------------------------------------------------------------
#     @staticmethod
#     def engineer_weather_features(weather_data: pd.DataFrame,
#                                   lap_data: pd.DataFrame) -> pd.DataFrame:

#         if lap_data.empty:
#             return lap_data.copy()

#         df = lap_data.copy()

#         try:
#             if not weather_data.empty:

#                 if "AIR_TEMP" in weather_data.columns:
#                     df["TEMP_IMPACT"] = (weather_data["AIR_TEMP"].mean() - 25.0) * 0.03

#                 if "TRACK_TEMP" in weather_data.columns:
#                     df["TRACK_TEMP_IMPACT"] = \
#                         (weather_data["TRACK_TEMP"].mean() - 35.0) * 0.02

#                 if "RAIN" in weather_data.columns:
#                     df["RAIN_IMPACT"] = weather_data["RAIN"].max() * 1.5

#         except Exception as e:
#             print(f"⚠️ Weather feature engineering failed: {e}")

#         return df

#     # ------------------------------------------------------------
#     # STRATEGY FEATURES
#     # ------------------------------------------------------------
#     @staticmethod
#     def engineer_strategy_features(race_data: pd.DataFrame,
#                                    lap_data: pd.DataFrame) -> pd.DataFrame:

#         if lap_data.empty or race_data.empty:
#             return pd.DataFrame()

#         strategy_rows = []

#         try:
#             for car_number in race_data["NUMBER"].dropna().unique():

#                 car_race = race_data[race_data["NUMBER"] == car_number]
#                 if car_race.empty:
#                     continue

#                 car_laps = lap_data[lap_data["NUMBER"] == car_number]
#                 if len(car_laps) < 3:
#                     continue

#                 pos = car_race.iloc[0].get("POSITION", np.nan)

#                 strategy_rows.append({
#                     "car_number": car_number,
#                     "position": pos,
#                     "total_laps": len(car_laps),
#                     "needs_strategy_change": 1 if pos and pos > 10 else 0
#                 })

#         except Exception as e:
#             print(f"⚠️ Strategy engineering failed: {e}")

#         return pd.DataFrame(strategy_rows)

#     # ------------------------------------------------------------
#     # MASTER COMPOSITE FEATURE ENGINEERING
#     # ------------------------------------------------------------
#     @staticmethod
#     def create_composite_features(processed_data: Dict) -> Dict:

#         enhanced = {}

#         for track_name, data in processed_data.items():

#             try:
#                 lap = data.get("lap_data", pd.DataFrame())
#                 race = data.get("race_data", pd.DataFrame())
#                 weather = data.get("weather_data", pd.DataFrame())
#                 telemetry = data.get("telemetry_data", pd.DataFrame())

#                 if lap.empty:
#                     enhanced[track_name] = data
#                     continue

#                 lap = FeatureEngineer.engineer_tire_features(lap, telemetry)
#                 lap = FeatureEngineer.engineer_fuel_features(lap, telemetry)
#                 lap = FeatureEngineer.engineer_track_features(track_name, lap)
#                 lap = FeatureEngineer.engineer_weather_features(weather, lap)

#                 strategy = FeatureEngineer.engineer_strategy_features(race, lap)

#                 enhanced[track_name] = {
#                     "lap_data": lap,
#                     "race_data": race,
#                     "weather_data": weather,
#                     "telemetry_data": telemetry,
#                     "strategy_features": strategy
#                 }

#             except Exception as e:
#                 print(f"⚠️ Feature creation failed for {track_name}: {e}")
#                 enhanced[track_name] = data

#         return enhanced

























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