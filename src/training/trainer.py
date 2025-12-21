import numpy as np
from typing import Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelTrainer:
    """Обработка логики обучения моделей"""
    
    @staticmethod
    def apply_log_transform(y_train, y_val, y_test):
        """Применение логарифмического преобразования"""
        return (
            np.log1p(y_train),
            np.log1p(y_val),
            np.log1p(y_test)
        )
    
    @staticmethod
    def inverse_log_transform(y_pred, y_test_orig):
        """Обратное преобразование из логарифмической шкалы"""
        return np.expm1(y_pred), y_test_orig
    
    @staticmethod
    def calculate_regression_metrics(y_true, y_pred):
        """Расчет метрик регрессии"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        mask = y_true > 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    @staticmethod
    def filter_attacks_only(X_train, y_train, X_val, y_val, X_test, y_test, 
                           threshold=1.0) -> Tuple:
        # Train
        train_mask = y_train > threshold
        X_train_filtered = X_train[train_mask]
        y_train_filtered = y_train[train_mask]
        
        # Val
        val_mask = y_val > threshold
        X_val_filtered = X_val[val_mask]
        y_val_filtered = y_val[val_mask]
        
        # Test
        test_mask = y_test > threshold
        X_test_filtered = X_test[test_mask]
        y_test_filtered = y_test[test_mask]
        
        return (
            X_train_filtered, y_train_filtered,
            X_val_filtered, y_val_filtered,
            X_test_filtered, y_test_filtered,
            test_mask
        )
    
    @staticmethod
    def train_classification(model_class, config, X_train, y_train, X_val, y_val, 
                            X_test, y_test, label_encoder, feature_names):
        """Обучение модели классификации"""
        model = model_class(config, task='classification')

        model.train(X_train, y_train, X_val, y_val)
        
        test_metrics = model.evaluate(X_test, y_test, label_encoder)
        y_pred = test_metrics['predictions']

        importance = {}
        if hasattr(model, 'get_feature_importance') and feature_names:
            importance = model.get_feature_importance(feature_names, top_n=15)
        
        model.save()
        
        return {
            'test_metrics': test_metrics,
            'importance': importance,
            'y_test': y_test,
            'y_pred': y_pred,
            'model_instance': model
        }
    
    @staticmethod
    def train_regression(model_class, config, X_train, y_train, X_val, y_val, 
                        X_test, y_test, feature_names, use_log_transform=True,
                        attack_only=True):

        print(f"      Режим: {'только атаки' if attack_only else 'все данные'}")
        
        if attack_only:
            # Фильтруем только атаки
            (X_train_filt, y_train_filt,
             X_val_filt, y_val_filt,
             X_test_filt, y_test_filt,
             test_mask) = ModelTrainer.filter_attacks_only(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            
            print(f"      Фильтрация: train {len(y_train_filt):,}/{len(y_train):,}, "
                  f"test {len(y_test_filt):,}/{len(y_test):,}")
            
            X_train, y_train = X_train_filt, y_train_filt
            X_val, y_val = X_val_filt, y_val_filt
            X_test_orig, y_test_orig = X_test, y_test
            X_test, y_test = X_test_filt, y_test_filt
        else:
            y_test_orig = y_test.copy()
            test_mask = np.ones(len(y_test), dtype=bool)
        
        if use_log_transform:
            y_train, y_val, y_test = ModelTrainer.apply_log_transform(
                y_train, y_val, y_test
            )
        
        model = model_class(config, task='regression')
        
        model.train(X_train, y_train, X_val, y_val)
        
        test_metrics = model.evaluate(X_test, y_test, None)
        y_pred_filt = test_metrics['predictions']
        
        # Обратное преобразование
        if use_log_transform:
            y_pred_filt = np.expm1(y_pred_filt)
            y_test_filt = np.expm1(y_test)
        else:
            y_test_filt = y_test
        
        if attack_only:
            y_pred_full = np.zeros(len(y_test_orig))
            y_pred_full[test_mask] = y_pred_filt

            test_metrics_full = ModelTrainer.calculate_regression_metrics(
                y_test_orig, y_pred_full
            )
            test_metrics_full['predictions'] = y_pred_full

            test_metrics_attacks = ModelTrainer.calculate_regression_metrics(
                y_test_filt, y_pred_filt
            )
            
            print(f"      Метрики (все данные): MAE=${test_metrics_full['mae']:,.2f}, "
                  f"R²={test_metrics_full['r2']:.4f}")
            print(f"      Метрики (только атаки): MAE=${test_metrics_attacks['mae']:,.2f}, "
                  f"R²={test_metrics_attacks['r2']:.4f}")
            
            test_metrics = test_metrics_full
            y_test_return = y_test_orig
            y_pred_return = y_pred_full
        else:
            test_metrics = ModelTrainer.calculate_regression_metrics(y_test_orig, y_pred_filt)
            test_metrics['predictions'] = y_pred_filt
            y_test_return = y_test_orig
            y_pred_return = y_pred_filt
        
        importance = {}
        if hasattr(model, 'get_feature_importance') and feature_names:
            importance = model.get_feature_importance(feature_names, top_n=15)
        
        model.save()
        
        return {
            'test_metrics': test_metrics,
            'importance': importance,
            'y_test': y_test_return,
            'y_pred': y_pred_return,
            'model_instance': model
        }
