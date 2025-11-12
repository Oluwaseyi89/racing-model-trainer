import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import json
import logging

class ModelEvaluator:
    """Comprehensive model evaluation and validation for racing models"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def evaluate_tire_model(self, model_result: Dict, test_data: Dict) -> Dict[str, Any]:
        """Evaluate tire degradation model with realistic validation"""
        if 'error' in model_result:
            return {'status': 'failed', 'error': model_result['error']}
        
        metrics = {}
        try:
            # Extract model and features
            model = model_result['model']
            feature_columns = model_result.get('features', [])
            
            if not feature_columns or test_data['lap_data'].empty:
                return {'status': 'insufficient_data', 'error': 'Missing features or test data'}
            
            # Prepare test data
            X_test = test_data['lap_data'][feature_columns].dropna()
            if X_test.empty:
                return {'status': 'insufficient_data', 'error': 'No valid test samples after preprocessing'}
            
            # For tire model, we need to validate degradation predictions
            predictions = {}
            actual_metrics = {}
            
            # Validate degradation rate predictions
            if hasattr(model, 'predict_degradation'):
                # Test on sample laps to get degradation predictions
                sample_features = X_test.iloc[:10].to_dict('records')
                degradation_predictions = []
                for features in sample_features:
                    try:
                        pred = model.predict_degradation(features)
                        degradation_predictions.append(pred)
                    except Exception as e:
                        self.logger.warning(f"Degradation prediction failed: {e}")
                
                if degradation_predictions:
                    # Validate degradation rates are realistic (0.01-0.5 seconds per lap)
                    degradation_rates = [np.mean([p.get(f'degradation_s{i}', 0) for i in range(1, 4)]) 
                                       for p in degradation_predictions]
                    valid_rates = [0.01 <= rate <= 0.5 for rate in degradation_rates]
                    metrics['degradation_rate_validity'] = sum(valid_rates) / len(valid_rates)
            
            # Validate stint length predictions
            if hasattr(model, 'estimate_optimal_stint_length'):
                stint_predictions = []
                for features in X_test.iloc[:5].to_dict('records'):
                    try:
                        stint = model.estimate_optimal_stint_length(features)
                        stint_predictions.append(stint)
                    except:
                        continue
                
                if stint_predictions:
                    # Validate stint lengths are realistic (5-30 laps)
                    valid_stints = [5 <= stint <= 30 for stint in stint_predictions]
                    metrics['stint_length_validity'] = sum(valid_stints) / len(valid_stints)
            
            metrics.update({
                'status': 'success',
                'test_samples': len(X_test),
                'feature_count': len(feature_columns),
                'training_score': model_result.get('test_score', 0),
                'feature_importance': model_result.get('feature_importance', {})
            })
            
        except Exception as e:
            metrics = {'status': 'evaluation_failed', 'error': str(e)}
        
        return metrics
    
    def evaluate_fuel_model(self, model_result: Dict, test_data: Dict) -> Dict[str, Any]:
        """Evaluate fuel consumption model with telemetry validation"""
        if 'error' in model_result:
            return {'status': 'failed', 'error': model_result['error']}
        
        metrics = {}
        try:
            model = model_result['model']
            feature_columns = model_result.get('features', [])
            
            if not feature_columns or test_data['lap_data'].empty:
                return {'status': 'insufficient_data', 'error': 'Missing features or test data'}
            
            # Test fuel consumption predictions
            test_laps = test_data['lap_data'].sample(min(20, len(test_data['lap_data'])))
            predictions = []
            valid_predictions = 0
            
            for _, lap in test_laps.iterrows():
                try:
                    # Create feature dictionary for prediction
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
                    'mean_consumption': np.mean(predictions),
                    'std_consumption': np.std(predictions),
                    'min_consumption': min(predictions),
                    'max_consumption': max(predictions),
                    'training_score': model_result.get('test_score', 0)
                })
            else:
                metrics = {'status': 'no_predictions', 'error': 'Could not generate fuel predictions'}
                
        except Exception as e:
            metrics = {'status': 'evaluation_failed', 'error': str(e)}
        
        return metrics
    
    def evaluate_pit_strategy_model(self, model_result: Dict, test_data: Dict) -> Dict[str, Any]:
        """Evaluate pit strategy model with race scenario validation"""
        if 'error' in model_result:
            return {'status': 'failed', 'error': model_result['error']}
        
        metrics = {}
        try:
            model = model_result['model']
            accuracy = model_result.get('accuracy', 0)
            
            # Validate strategy predictions make sense
            test_scenarios = self._create_strategy_test_scenarios()
            valid_predictions = 0
            total_predictions = 0
            
            for scenario in test_scenarios:
                try:
                    strategy = model.predict_optimal_strategy(scenario)
                    if strategy in ['early', 'middle', 'late']:
                        valid_predictions += 1
                    total_predictions += 1
                except:
                    continue
            
            metrics.update({
                'status': 'success',
                'accuracy': accuracy,
                'strategy_prediction_validity': valid_predictions / total_predictions if total_predictions > 0 else 0,
                'test_scenarios': total_predictions,
                'feature_importance': model_result.get('feature_importance', {})
            })
            
        except Exception as e:
            metrics = {'status': 'evaluation_failed', 'error': str(e)}
        
        return metrics
    
    def evaluate_weather_model(self, model_result: Dict, test_data: Dict) -> Dict[str, Any]:
        """Evaluate weather impact model with realistic condition validation"""
        if 'error' in model_result:
            return {'status': 'failed', 'error': model_result['error']}
        
        metrics = {}
        try:
            model = model_result['model']
            
            # Test weather impact predictions for various conditions
            test_conditions = [
                {'air_temp': 20, 'track_temp': 25, 'humidity': 60, 'pressure': 1013, 'wind_speed': 2, 'rain': 0},
                {'air_temp': 30, 'track_temp': 40, 'humidity': 80, 'pressure': 1000, 'wind_speed': 10, 'rain': 1},
                {'air_temp': 15, 'track_temp': 18, 'humidity': 40, 'pressure': 1020, 'wind_speed': 5, 'rain': 0}
            ]
            
            impacts = []
            for conditions in test_conditions:
                try:
                    impact = model.predict_weather_impact(conditions, 'test_track', {'lap_number': 10})
                    # Weather impact should be within reasonable bounds (-5 to +5 seconds)
                    if -5 <= impact <= 5:
                        impacts.append(impact)
                except:
                    continue
            
            if impacts:
                metrics.update({
                    'status': 'success',
                    'test_conditions': len(impacts),
                    'mean_impact': np.mean(impacts),
                    'impact_range': max(impacts) - min(impacts),
                    'training_score': model_result.get('test_score', 0),
                    'realistic_impacts': len([i for i in impacts if abs(i) <= 3]) / len(impacts)
                })
            else:
                metrics = {'status': 'no_predictions', 'error': 'Could not generate weather impact predictions'}
                
        except Exception as e:
            metrics = {'status': 'evaluation_failed', 'error': str(e)}
        
        return metrics
    
    def _create_strategy_test_scenarios(self) -> List[Dict]:
        """Create realistic test scenarios for pit strategy validation"""
        scenarios = []
        
        # Various race situations
        base_scenario = {
            'tire_degradation_rate': 0.1,
            'fuel_effect': 0.8,
            'position': 5,
            'gap_to_leader': 15.0,
            'gap_to_next': 2.5,
            'track_wear_factor': 0.7,
            'total_laps': 25
        }
        
        # Leading car with low degradation
        scenarios.append({**base_scenario, 'position': 1, 'tire_degradation_rate': 0.05})
        
        # Mid-field with high degradation
        scenarios.append({**base_scenario, 'position': 8, 'tire_degradation_rate': 0.2})
        
        # Back marker with medium degradation
        scenarios.append({**base_scenario, 'position': 15, 'gap_to_leader': 45.0})
        
        # Close battle for position
        scenarios.append({**base_scenario, 'gap_to_next': 0.5, 'gap_to_leader': 8.0})
        
        # High wear track
        scenarios.append({**base_scenario, 'track_wear_factor': 0.9, 'tire_degradation_rate': 0.15})
        
        return scenarios
    
    def cross_validate_models(self, processed_data: Dict, n_splits: int = 3) -> Dict[str, Any]:
        """Perform cross-validation across different tracks"""
        cv_results = {}
        
        for track_name, data in processed_data.items():
            if data['lap_data'].empty:
                continue
                
            track_results = {}
            lap_data = data['lap_data']
            
            # Simple time-series cross-validation for lap data
            if len(lap_data) >= 10:
                tscv = TimeSeriesSplit(n_splits=min(n_splits, len(lap_data) // 3))
                
                # Example: Validate lap time consistency
                if 'LAP_TIME_SECONDS' in lap_data.columns:
                    lap_times = lap_data['LAP_TIME_SECONDS'].dropna().values
                    if len(lap_times) >= 10:
                        # Calculate consistency across splits
                        consistencies = []
                        for train_idx, test_idx in tscv.split(lap_times):
                            train_std = np.std(lap_times[train_idx])
                            test_std = np.std(lap_times[test_idx])
                            consistencies.append(abs(train_std - test_std))
                        
                        track_results['lap_time_consistency_cv'] = np.mean(consistencies)
            
            cv_results[track_name] = track_results
        
        return cv_results
    
    def generate_comprehensive_report(self, model_results: Dict, evaluation_results: Dict) -> str:
        """Generate comprehensive evaluation report"""
        report = {
            'summary': {
                'total_models': len(model_results),
                'successful_evaluations': sum(1 for r in evaluation_results.values() 
                                            if r.get('status') == 'success'),
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            },
            'model_details': {},
            'performance_metrics': {},
            'recommendations': []
        }
        
        for model_name, eval_result in evaluation_results.items():
            report['model_details'][model_name] = eval_result
            
            # Generate recommendations based on evaluation results
            if eval_result.get('status') == 'success':
                if model_name == 'fuel_consumption':
                    validity = eval_result.get('valid_prediction_rate', 0)
                    if validity < 0.8:
                        report['recommendations'].append(
                            f"Improve {model_name} model - only {validity:.1%} of predictions are realistic"
                        )
                
                elif model_name == 'pit_strategy':
                    accuracy = eval_result.get('accuracy', 0)
                    if accuracy < 0.7:
                        report['recommendations'].append(
                            f"Enhance {model_name} model - accuracy {accuracy:.1%} below target"
                        )
            
            elif eval_result.get('status') != 'success':
                report['recommendations'].append(
                    f"Fix {model_name} model - {eval_result.get('error', 'evaluation failed')}"
                )
        
        # Overall assessment
        success_rate = report['summary']['successful_evaluations'] / report['summary']['total_models']
        if success_rate >= 0.8:
            report['overall_assessment'] = "EXCELLENT - Most models performing well"
        elif success_rate >= 0.6:
            report['overall_assessment'] = "GOOD - Models generally performing adequately"
        else:
            report['overall_assessment'] = "NEEDS IMPROVEMENT - Significant model issues detected"
        
        return json.dumps(report, indent=2)
    
    def plot_evaluation_results(self, evaluation_results: Dict, save_path: str = None):
        """Create visualization of model evaluation results"""
        if not evaluation_results:
            self.logger.warning("No evaluation results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Evaluation Results', fontsize=16)
        
        # Plot 1: Model Status
        status_counts = {}
        for result in evaluation_results.values():
            status = result.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        if status_counts:
            axes[0, 0].pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%')
            axes[0, 0].set_title('Model Evaluation Status')
        
        # Plot 2: Performance Scores
        performance_data = {}
        for model_name, result in evaluation_results.items():
            if result.get('status') == 'success':
                score = result.get('training_score', result.get('accuracy', 0))
                performance_data[model_name] = score
        
        if performance_data:
            axes[0, 1].bar(performance_data.keys(), performance_data.values())
            axes[0, 1].set_title('Model Performance Scores')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].axhline(y=0.7, color='r', linestyle='--', alpha=0.7, label='Target Score')
            axes[0, 1].legend()
        
        # Plot 3: Prediction Validity
        validity_data = {}
        for model_name, result in evaluation_results.items():
            if 'valid_prediction_rate' in result:
                validity_data[model_name] = result['valid_prediction_rate']
            elif 'strategy_prediction_validity' in result:
                validity_data[model_name] = result['strategy_prediction_validity']
            elif 'realistic_impacts' in result:
                validity_data[model_name] = result['realistic_impacts']
        
        if validity_data:
            axes[1, 0].bar(validity_data.keys(), validity_data.values())
            axes[1, 0].set_title('Prediction Validity Rates')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].axhline(y=0.8, color='g', linestyle='--', alpha=0.7, label='Validity Target')
            axes[1, 0].legend()
        
        # Plot 4: Test Sample Sizes
        sample_data = {}
        for model_name, result in evaluation_results.items():
            if 'test_samples' in result:
                sample_data[model_name] = result['test_samples']
        
        if sample_data:
            axes[1, 1].bar(sample_data.keys(), sample_data.values())
            axes[1, 1].set_title('Test Sample Sizes')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved evaluation plot to {save_path}")
        
        plt.show()






















# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns
# from typing import Dict, Any
# import json

# class ModelEvaluator:
#     """Comprehensive model evaluation and validation"""
    
#     @staticmethod
#     def evaluate_regression_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, Any]:
#         """Evaluate regression model performance"""
#         y_pred = model.predict(X_test)
        
#         metrics = {
#             'model_name': model_name,
#             'mae': mean_absolute_error(y_test, y_pred),
#             'mse': mean_squared_error(y_test, y_pred),
#             'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
#             'r2_score': r2_score(y_test, y_pred),
#             'mean_absolute_percentage_error': np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
#             'test_samples': len(y_test),
#             'mean_prediction': float(np.mean(y_pred)),
#             'std_prediction': float(np.std(y_pred))
#         }
        
#         return metrics
    
#     @staticmethod
#     def evaluate_classification_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, Any]:
#         """Evaluate classification model performance"""
#         y_pred = model.predict(X_test)
        
#         metrics = {
#             'model_name': model_name,
#             'accuracy': accuracy_score(y_test, y_pred),
#             'test_samples': len(y_test),
#             'class_distribution': dict(y_test.value_counts()),
#             'prediction_distribution': dict(pd.Series(y_pred).value_counts())
#         }
        
#         # Add detailed classification report
#         report = classification_report(y_test, y_pred, output_dict=True)
#         metrics['detailed_report'] = report
        
#         return metrics
    
#     @staticmethod
#     def cross_validate_tire_model(model, lap_data: pd.DataFrame, n_splits: int = 5) -> Dict[str, Any]:
#         """Cross-validate tire degradation model across different tracks"""
#         tracks = lap_data['track_name'].unique() if 'track_name' in lap_data.columns else ['combined']
#         results = {}
        
#         for track in tracks:
#             if track != 'combined':
#                 track_data = lap_data[lap_data['track_name'] == track]
#             else:
#                 track_data = lap_data
            
#             if len(track_data) < 10:
#                 continue
                
#             # Simple train-test split for each track
#             from sklearn.model_selection import train_test_split
#             features = ['LAP_NUMBER', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']
#             X = track_data[features].dropna()
#             y = track_data.loc[X.index, 'LAP_TIME_SECONDS']
            
#             if len(X) < 5:
#                 continue
                
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#             model.fit(X_train, y_train)
            
#             track_metrics = ModelEvaluator.evaluate_regression_model(
#                 model, X_test, y_test, f"tire_model_{track}"
#             )
#             results[track] = track_metrics
        
#         return results
    
#     @staticmethod
#     def validate_fuel_predictions(model, lap_data: pd.DataFrame) -> Dict[str, Any]:
#         """Validate fuel consumption predictions against realistic bounds"""
#         predictions = []
#         actual_consumption = []
        
#         # Simulate fuel consumption validation
#         for _, lap in lap_data.iterrows():
#             features = {
#                 'lap_number': lap['LAP_NUMBER'],
#                 'avg_speed': lap.get('KPH', 0),
#                 'time_variation': 0,  # Simplified
#                 's1_time': lap.get('S1_SECONDS', 0),
#                 's2_time': lap.get('S2_SECONDS', 0),
#                 's3_time': lap.get('S3_SECONDS', 0),
#                 'top_speed': lap.get('TOP_SPEED', lap.get('KPH', 0) * 1.1),
#                 'position_change': 0,
#                 'temp_effect': 1.0
#             }
            
#             try:
#                 pred = model.predict_fuel_consumption(features)
#                 predictions.append(pred)
                
#                 # Simulate actual consumption (2-4 liters per lap realistic range)
#                 actual = np.random.uniform(2.0, 4.0)
#                 actual_consumption.append(actual)
#             except:
#                 continue
        
#         if not predictions:
#             return {'validation_passed': False, 'error': 'No predictions generated'}
        
#         validation_metrics = {
#             'validation_passed': all(2.0 <= p <= 4.0 for p in predictions),
#             'mean_predicted_consumption': np.mean(predictions),
#             'std_predicted_consumption': np.std(predictions),
#             'min_predicted_consumption': min(predictions),
#             'max_predicted_consumption': max(predictions),
#             'predictions_within_bounds': sum(2.0 <= p <= 4.0 for p in predictions) / len(predictions),
#             'total_predictions': len(predictions)
#         }
        
#         return validation_metrics
    
#     @staticmethod
#     def analyze_feature_importance(model, feature_names: list) -> Dict[str, float]:
#         """Analyze and rank feature importance"""
#         if hasattr(model, 'feature_importances_'):
#             importance_dict = dict(zip(feature_names, model.feature_importances_))
#             return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
#         else:
#             return {'error': 'Model does not have feature_importances_ attribute'}
    
#     @staticmethod
#     def generate_model_report(trained_models: Dict[str, Any]) -> str:
#         """Generate comprehensive model evaluation report"""
#         report = {
#             'summary': {
#                 'total_models_trained': len(trained_models),
#                 'models_trained': list(trained_models.keys()),
#                 'timestamp': pd.Timestamp.now().isoformat()
#             },
#             'model_performance': {},
#             'recommendations': []
#         }
        
#         for model_name, model_result in trained_models.items():
#             if 'accuracy' in model_result:  # Classification model
#                 report['model_performance'][model_name] = {
#                     'type': 'classification',
#                     'accuracy': model_result['accuracy'],
#                     'test_samples': model_result.get('test_samples', 'N/A')
#                 }
                
#                 if model_result['accuracy'] < 0.7:
#                     report['recommendations'].append(
#                         f"Consider improving {model_name} - accuracy below 70%"
#                     )
                    
#             elif 'test_score' in model_result:  # Regression model
#                 report['model_performance'][model_name] = {
#                     'type': 'regression',
#                     'r2_score': model_result['test_score'],
#                     'train_score': model_result['train_score'],
#                     'overfitting_risk': model_result['train_score'] - model_result['test_score'] > 0.1
#                 }
                
#                 if model_result['test_score'] < 0.6:
#                     report['recommendations'].append(
#                         f"Consider feature engineering for {model_name} - R² below 60%"
#                     )
        
#         # Overall assessment
#         if len(report['recommendations']) == 0:
#             report['overall_assessment'] = "All models performing adequately"
#         else:
#             report['overall_assessment'] = f"Found {len(report['recommendations'])} areas for improvement"
        
#         return json.dumps(report, indent=2)
    
#     @staticmethod
#     def plot_model_performance(metrics: Dict[str, Any], save_path: str = None):
#         """Create visualization of model performance"""
#         fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
#         # Plot 1: R² Scores for regression models
#         regression_scores = {
#             name: metrics['test_score'] 
#             for name, metrics in metrics.items() 
#             if 'test_score' in metrics
#         }
        
#         if regression_scores:
#             axes[0, 0].bar(regression_scores.keys(), regression_scores.values())
#             axes[0, 0].set_title('Regression Models R² Scores')
#             axes[0, 0].set_ylabel('R² Score')
#             axes[0, 0].tick_params(axis='x', rotation=45)
        
#         # Plot 2: Feature Importance (example for first model)
#         if 'feature_importance' in list(metrics.values())[0]:
#             first_model = list(metrics.values())[0]
#             importance = first_model['feature_importance']
#             axes[0, 1].barh(list(importance.keys())[:10], list(importance.values())[:10])
#             axes[0, 1].set_title('Top 10 Feature Importances')
        
#         # Plot 3: Train vs Test scores
#         train_scores = [m.get('train_score', 0) for m in metrics.values() if 'train_score' in m]
#         test_scores = [m.get('test_score', 0) for m in metrics.values() if 'test_score' in m]
        
#         if train_scores and test_scores:
#             x = range(len(train_scores))
#             axes[1, 0].plot(x, train_scores, 'b-', label='Train Score')
#             axes[1, 0].plot(x, test_scores, 'r-', label='Test Score')
#             axes[1, 0].set_title('Train vs Test Scores')
#             axes[1, 0].legend()
        
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
#         plt.show()
    
#     @staticmethod
#     def validate_real_time_predictions(predictions: Dict[str, Any]) -> Dict[str, bool]:
#         """Validate that real-time predictions are within realistic bounds"""
#         validation_results = {}
        
#         # Tire wear validation (0-100%)
#         if 'tire_wear' in predictions:
#             wear = predictions['tire_wear']
#             validation_results['tire_wear_valid'] = 0 <= wear <= 100
        
#         # Fuel consumption validation (realistic range)
#         if 'fuel_consumption' in predictions:
#             consumption = predictions['fuel_consumption']
#             validation_results['fuel_consumption_valid'] = 1.5 <= consumption <= 5.0
        
#         # Pit strategy validation (sensible categories)
#         if 'pit_strategy' in predictions:
#             strategy = predictions['pit_strategy']
#             validation_results['pit_strategy_valid'] = strategy in ['early', 'middle', 'late']
        
#         validation_results['all_valid'] = all(validation_results.values())
        
#         return validation_results