import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import json
import logging
import joblib
import os

class ModelEvaluator:
    """Comprehensive model evaluation and validation for racing models - Consistent with FirebaseDataLoader structure"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def evaluate_tire_model(self, model_result: Dict, test_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Evaluate tire degradation model with realistic validation using FirebaseDataLoader structure"""
        if model_result.get('status') != 'success':
            return {'status': 'failed', 'error': model_result.get('error', 'Model training failed')}
        
        metrics = {}
        try:
            model = model_result['model']
            feature_columns = model_result.get('features', [])
            
            # Extract test data from FirebaseDataLoader structure
            test_pit_data = pd.DataFrame()
            for track_data in test_data.values():
                if 'pit_data' in track_data and not track_data['pit_data'].empty:
                    test_pit_data = pd.concat([test_pit_data, track_data['pit_data']], ignore_index=True)
            
            if not feature_columns or test_pit_data.empty:
                return {'status': 'insufficient_data', 'error': 'Missing features or test data'}
            
            # Prepare test features
            X_test = test_pit_data[feature_columns].dropna()
            if X_test.empty:
                return {'status': 'insufficient_data', 'error': 'No valid test samples after preprocessing'}
            
            predictions = []
            valid_predictions = 0
            
            # Test degradation predictions on sample data
            sample_features = X_test.iloc[:min(20, len(X_test))].to_dict('records')
            
            for features in sample_features:
                try:
                    degradation_pred = model.predict_tire_degradation(features)
                    
                    # Validate degradation rates are realistic (0.01-0.5 seconds per lap per sector)
                    valid_sectors = 0
                    for sector in ['degradation_s1', 'degradation_s2', 'degradation_s3']:
                        if sector in degradation_pred and 0.01 <= degradation_pred[sector] <= 0.5:
                            valid_sectors += 1
                    
                    if valid_sectors >= 2:  # At least 2/3 sectors realistic
                        valid_predictions += 1
                    
                    predictions.append(degradation_pred)
                except Exception as e:
                    self.logger.warning(f"Tire degradation prediction failed: {e}")
                    continue
            
            # Test stint length predictions
            stint_predictions = []
            for features in X_test.iloc[:min(10, len(X_test))].to_dict('records'):
                try:
                    stint = model.estimate_optimal_stint_length(features)
                    stint_predictions.append(stint)
                except Exception as e:
                    continue
            
            stint_validity = 0
            if stint_predictions:
                valid_stints = [5 <= stint <= 30 for stint in stint_predictions]
                stint_validity = sum(valid_stints) / len(valid_stints)
            
            metrics.update({
                'status': 'success',
                'test_samples': len(X_test),
                'feature_count': len(feature_columns),
                'training_score': model_result.get('test_score', 0),
                'degradation_prediction_rate': valid_predictions / len(sample_features) if sample_features else 0,
                'stint_length_validity': stint_validity,
                'tracks_tested': len(test_data),
                'feature_importance': model_result.get('feature_importance', {})
            })
            
        except Exception as e:
            metrics = {'status': 'evaluation_failed', 'error': str(e)}
        
        return metrics
    
    def evaluate_fuel_model(self, model_result: Dict, test_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Evaluate fuel consumption model using FirebaseDataLoader structure"""
        if model_result.get('status') != 'success':
            return {'status': 'failed', 'error': model_result.get('error', 'Model training failed')}
        
        metrics = {}
        try:
            model = model_result['model']
            feature_columns = model_result.get('features', [])
            
            # Extract test data from multiple tracks
            test_pit_data = pd.DataFrame()
            for track_data in test_data.values():
                if 'pit_data' in track_data and not track_data['pit_data'].empty:
                    test_pit_data = pd.concat([test_pit_data, track_data['pit_data']], ignore_index=True)
            
            if not feature_columns or test_pit_data.empty:
                return {'status': 'insufficient_data', 'error': 'Missing features or test data'}
            
            # Test fuel consumption predictions
            test_samples = test_pit_data.sample(min(25, len(test_pit_data)))
            predictions = []
            valid_predictions = 0
            
            for _, lap in test_samples.iterrows():
                try:
                    features = {col: lap.get(col, 0) for col in feature_columns}
                    fuel_pred = model.predict_fuel_consumption(features)
                    
                    # Validate prediction is realistic (1.5-4.0 liters per lap for GR86)
                    if 1.5 <= fuel_pred <= 4.0:
                        valid_predictions += 1
                    predictions.append(fuel_pred)
                except Exception as e:
                    continue
            
            if predictions:
                metrics.update({
                    'status': 'success',
                    'test_predictions': len(predictions),
                    'valid_prediction_rate': valid_predictions / len(predictions),
                    'mean_consumption': float(np.mean(predictions)),
                    'std_consumption': float(np.std(predictions)),
                    'min_consumption': float(min(predictions)),
                    'max_consumption': float(max(predictions)),
                    'training_score': model_result.get('test_score', 0),
                    'tracks_tested': len(test_data),
                    'feature_count': len(feature_columns)
                })
            else:
                metrics = {'status': 'no_predictions', 'error': 'Could not generate fuel predictions'}
                
        except Exception as e:
            metrics = {'status': 'evaluation_failed', 'error': str(e)}
        
        return metrics
    
    def evaluate_pit_strategy_model(self, model_result: Dict, test_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Evaluate pit strategy model using FirebaseDataLoader structure"""
        if model_result.get('status') != 'success':
            return {'status': 'failed', 'error': model_result.get('error', 'Model training failed')}
        
        metrics = {}
        try:
            model = model_result['model']
            accuracy = model_result.get('accuracy', 0)
            feature_columns = model_result.get('features', [])
            
            # Create realistic test scenarios based on actual track data
            test_scenarios = self._create_strategy_test_scenarios(test_data, feature_columns)
            valid_predictions = 0
            total_predictions = 0
            strategy_distribution = {}
            
            for scenario in test_scenarios:
                try:
                    strategy = model.predict_pit_strategy(scenario)
                    if strategy in ['early', 'middle', 'late', 'undercut', 'overcut']:
                        valid_predictions += 1
                        strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
                    
                    # Test confidence scores
                    confidence = model.get_strategy_confidence(scenario)
                    if confidence and all(0 <= score <= 1 for score in confidence.values()):
                        valid_predictions += 0.5  # Partial credit for valid confidence
                    
                    total_predictions += 1
                except Exception as e:
                    continue
            
            metrics.update({
                'status': 'success',
                'accuracy': accuracy,
                'strategy_prediction_validity': valid_predictions / total_predictions if total_predictions > 0 else 0,
                'test_scenarios': total_predictions,
                'strategy_distribution': strategy_distribution,
                'training_samples': model_result.get('training_samples', 0),
                'tracks_used': model_result.get('tracks_used', 0),
                'feature_importance': model_result.get('feature_importance', {})
            })
            
        except Exception as e:
            metrics = {'status': 'evaluation_failed', 'error': str(e)}
        
        return metrics
    
    def evaluate_weather_model(self, model_result: Dict, test_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Evaluate weather impact model using FirebaseDataLoader structure"""
        if model_result.get('status') != 'success':
            return {'status': 'failed', 'error': model_result.get('error', 'Model training failed')}
        
        metrics = {}
        try:
            model = model_result['model']
            feature_columns = model_result.get('features', [])
            
            # Extract weather data from test tracks
            test_weather_data = []
            track_names = []
            for track_name, track_data in test_data.items():
                if 'weather_data' in track_data and not track_data['weather_data'].empty:
                    test_weather_data.append(track_data['weather_data'])
                    track_names.append(track_name)
            
            if not test_weather_data:
                return {'status': 'insufficient_data', 'error': 'No weather data available for testing'}
            
            # Test weather impact predictions for various conditions
            impacts = []
            valid_impacts = 0
            
            for track_name, weather_df in zip(track_names, test_weather_data):
                for _, weather_row in weather_df.iterrows():
                    try:
                        # Create weather conditions dictionary
                        conditions = {
                            'AIR_TEMP': weather_row.get('AIR_TEMP', 25),
                            'TRACK_TEMP': weather_row.get('TRACK_TEMP', 30),
                            'HUMIDITY': weather_row.get('HUMIDITY', 50),
                            'PRESSURE': weather_row.get('PRESSURE', 1013),
                            'WIND_SPEED': weather_row.get('WIND_SPEED', 0),
                            'RAIN': weather_row.get('RAIN', 0)
                        }
                        
                        # Create lap context
                        lap_context = {
                            'lap_info': {'LAP_NUMBER': 10},
                            'telemetry': {'avg_speed': 120, 'driving_aggressiveness': 0.6}
                        }
                        
                        impact = model.predict_weather_impact(conditions, track_name, lap_context)
                        
                        # Weather impact should be within reasonable bounds (-5 to +5 seconds)
                        if -5 <= impact <= 5:
                            valid_impacts += 1
                        impacts.append(impact)
                    except Exception as e:
                        continue
            
            if impacts:
                metrics.update({
                    'status': 'success',
                    'test_conditions': len(impacts),
                    'valid_impact_rate': valid_impacts / len(impacts),
                    'mean_impact': float(np.mean(impacts)),
                    'impact_range': float(max(impacts) - min(impacts)),
                    'training_score': model_result.get('test_score', 0),
                    'tracks_tested': len(track_names),
                    'feature_count': len(feature_columns)
                })
            else:
                metrics = {'status': 'no_predictions', 'error': 'Could not generate weather impact predictions'}
                
        except Exception as e:
            metrics = {'status': 'evaluation_failed', 'error': str(e)}
        
        return metrics
    
    def _create_strategy_test_scenarios(self, test_data: Dict[str, Dict[str, pd.DataFrame]], feature_columns: List[str]) -> List[Dict]:
        """Create realistic test scenarios for pit strategy validation using actual data"""
        scenarios = []
        
        # Extract features from test data to create realistic scenarios
        for track_name, track_data in test_data.items():
            if 'pit_data' in track_data and not track_data['pit_data'].empty:
                pit_data = track_data['pit_data']
                
                # Sample different race situations from actual data
                for _, lap in pit_data.sample(min(5, len(pit_data))).iterrows():
                    scenario = {}
                    
                    # Fill scenario with available features
                    for feature in feature_columns:
                        scenario[feature] = lap.get(feature, 0)
                    
                    # Ensure required features have reasonable values
                    scenario.setdefault('tire_degradation_rate', 0.1)
                    scenario.setdefault('position_normalized', 0.5)
                    scenario.setdefault('gap_to_leader_seconds', 15.0)
                    scenario.setdefault('track_abrasiveness', 0.7)
                    scenario.setdefault('caution_flag_ratio', 0.1)
                    
                    scenarios.append(scenario)
        
        # Add some edge cases if we have few scenarios
        if len(scenarios) < 10:
            base_scenarios = [
                # Leading car with low degradation
                {'tire_degradation_rate': 0.05, 'position_normalized': 0.1, 'gap_to_leader_seconds': 5.0, 'track_abrasiveness': 0.6},
                # Mid-field with high degradation
                {'tire_degradation_rate': 0.2, 'position_normalized': 0.5, 'gap_to_leader_seconds': 25.0, 'track_abrasiveness': 0.8},
                # Back marker needing aggressive strategy
                {'tire_degradation_rate': 0.15, 'position_normalized': 0.8, 'gap_to_leader_seconds': 45.0, 'track_abrasiveness': 0.7},
                # Close battle situation
                {'tire_degradation_rate': 0.1, 'position_normalized': 0.3, 'gap_to_leader_seconds': 8.0, 'gap_to_next_seconds': 0.5}
            ]
            scenarios.extend(base_scenarios)
        
        return scenarios
    
    def cross_validate_track_consistency(self, processed_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Perform cross-validation across different tracks to check model consistency"""
        cv_results = {}
        
        for track_name, data in processed_data.items():
            if 'pit_data' not in data or data['pit_data'].empty:
                continue
                
            track_results = {}
            pit_data = data['pit_data']
            
            # Analyze lap time consistency across the track
            if 'LAP_TIME_SECONDS' in pit_data.columns:
                lap_times = pit_data['LAP_TIME_SECONDS'].dropna().values
                if len(lap_times) >= 5:
                    track_results['lap_time_mean'] = float(np.mean(lap_times))
                    track_results['lap_time_std'] = float(np.std(lap_times))
                    track_results['lap_time_cv'] = float(track_results['lap_time_std'] / track_results['lap_time_mean'])
            
            # Analyze tire degradation patterns if available
            if 'TIRE_DEGRADATION_RATE' in pit_data.columns:
                degradation = pit_data['TIRE_DEGRADATION_RATE'].dropna().values
                if len(degradation) >= 3:
                    track_results['degradation_mean'] = float(np.mean(degradation))
                    track_results['degradation_positive'] = float(np.sum(degradation > 0) / len(degradation))
            
            cv_results[track_name] = track_results
        
        # Calculate cross-track consistency metrics
        if len(cv_results) >= 2:
            lap_time_means = [r['lap_time_mean'] for r in cv_results.values() if 'lap_time_mean' in r]
            if lap_time_means:
                cv_results['cross_track_consistency'] = {
                    'lap_time_mean_std': float(np.std(lap_time_means)),
                    'lap_time_cv_mean': float(np.mean([r.get('lap_time_cv', 0) for r in cv_results.values() if 'lap_time_cv' in r]))
                }
        
        return cv_results
    
    def generate_comprehensive_report(self, model_results: Dict, evaluation_results: Dict) -> str:
        """Generate comprehensive evaluation report with FirebaseDataLoader context"""
        report = {
            'summary': {
                'total_models': len(model_results),
                'successful_models': sum(1 for r in model_results.values() if r.get('status') == 'success'),
                'successful_evaluations': sum(1 for r in evaluation_results.values() if r.get('status') == 'success'),
                'evaluation_timestamp': pd.Timestamp.now().isoformat(),
                'tracks_used': max([r.get('tracks_used', 0) for r in model_results.values() if isinstance(r, dict)] or [0])
            },
            'model_details': {},
            'performance_metrics': {},
            'recommendations': []
        }
        
        for model_name, eval_result in evaluation_results.items():
            report['model_details'][model_name] = eval_result
            
            # Model-specific recommendations
            if eval_result.get('status') == 'success':
                if model_name == 'fuel_consumption':
                    validity = eval_result.get('valid_prediction_rate', 0)
                    if validity < 0.8:
                        report['recommendations'].append(
                            f"Improve {model_name} model - {validity:.1%} realistic predictions (target: 80%)"
                        )
                
                elif model_name == 'pit_strategy':
                    accuracy = eval_result.get('accuracy', 0)
                    if accuracy < 0.7:
                        report['recommendations'].append(
                            f"Enhance {model_name} model - {accuracy:.1%} accuracy (target: 70%)"
                        )
                
                elif model_name == 'tire_degradation':
                    pred_rate = eval_result.get('degradation_prediction_rate', 0)
                    if pred_rate < 0.7:
                        report['recommendations'].append(
                            f"Optimize {model_name} model - {pred_rate:.1%} valid predictions (target: 70%)"
                        )
            
            elif eval_result.get('status') != 'success':
                report['recommendations'].append(
                    f"Fix {model_name} model - {eval_result.get('error', 'evaluation failed')}"
                )
        
        # Overall assessment
        success_rate = report['summary']['successful_evaluations'] / report['summary']['total_models']
        if success_rate >= 0.8:
            report['overall_assessment'] = "EXCELLENT - Models performing well across tracks"
        elif success_rate >= 0.6:
            report['overall_assessment'] = "GOOD - Models generally adequate for race strategy"
        else:
            report['overall_assessment'] = "NEEDS IMPROVEMENT - Significant model issues detected"
        
        return json.dumps(report, indent=2)
    
    def plot_evaluation_results(self, evaluation_results: Dict, save_path: str = None):
        """Create visualization of model evaluation results"""
        if not evaluation_results:
            self.logger.warning("No evaluation results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Evaluation Results - Racing Analytics', fontsize=16)
        
        # Plot 1: Model Status Distribution
        status_counts = {}
        for result in evaluation_results.values():
            status = result.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        if status_counts:
            axes[0, 0].pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Model Evaluation Status Distribution')
        
        # Plot 2: Performance Scores Comparison
        performance_data = {}
        for model_name, result in evaluation_results.items():
            if result.get('status') == 'success':
                score = result.get('training_score', result.get('accuracy', 0))
                performance_data[model_name] = score
        
        if performance_data:
            model_names = list(performance_data.keys())
            scores = list(performance_data.values())
            
            bars = axes[0, 1].bar(model_names, scores, color=['#2E8B57' if s >= 0.7 else '#FFA500' if s >= 0.5 else '#DC143C' for s in scores])
            axes[0, 1].set_title('Model Performance Scores')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Target (0.7)')
            axes[0, 1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Minimum (0.5)')
            axes[0, 1].legend()
            axes[0, 1].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{score:.3f}', ha='center', va='bottom')
        
        # Plot 3: Prediction Validity Rates
        validity_data = {}
        for model_name, result in evaluation_results.items():
            validity_keys = ['valid_prediction_rate', 'strategy_prediction_validity', 'degradation_prediction_rate', 'valid_impact_rate']
            for key in validity_keys:
                if key in result:
                    validity_data[model_name] = result[key]
                    break
        
        if validity_data:
            model_names = list(validity_data.keys())
            validity_rates = list(validity_data.values())
            
            bars = axes[1, 0].bar(model_names, validity_rates, 
                                color=['#2E8B57' if r >= 0.8 else '#FFA500' if r >= 0.6 else '#DC143C' for r in validity_rates])
            axes[1, 0].set_title('Prediction Validity Rates')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Target (80%)')
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].legend()
            
            # Add value labels
            for bar, rate in zip(bars, validity_rates):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{rate:.1%}', ha='center', va='bottom')
        
        # Plot 4: Training Data Coverage
        coverage_data = {}
        for model_name, result in evaluation_results.items():
            if 'training_samples' in result:
                coverage_data[model_name] = result['training_samples']
            elif 'test_samples' in result:
                coverage_data[model_name] = result['test_samples']
        
        if coverage_data:
            model_names = list(coverage_data.keys())
            samples = list(coverage_data.values())
            
            axes[1, 1].bar(model_names, samples, color='skyblue')
            axes[1, 1].set_title('Training/Test Sample Sizes')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].set_ylabel('Number of Samples')
            
            # Add value labels
            for i, v in enumerate(samples):
                axes[1, 1].text(i, v + max(samples)*0.01, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"✅ Saved evaluation plot to {save_path}")
        
        plt.show()

    def save_evaluation_report(self, report: str, filepath: str = "outputs/reports/evaluation_report.json"):
        """Save evaluation report to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            with open(filepath, 'w') as f:
                f.write(report)
            self.logger.info(f"✅ Evaluation report saved to {filepath}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save evaluation report: {e}")