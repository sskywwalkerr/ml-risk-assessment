import sys
sys.path.append('.')
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from src.utils.config import Config
from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer
from src.data.financial import FinancialLabeler
from src.data.preprocessor import Preprocessor
from src.models.random_forest import RandomForestModel
from src.utils.visualizer import Visualizer
from src.training.trainer import ModelTrainer

def prepare_data(config: Config):
    print("ПОДГОТОВКА ДАННЫХ")
    print("Загрузка данных")
    loader = DataLoader(config)
    data = loader.load()
    print(f"      Строк: {len(data):,}, Признаков: {len(data.columns)}")

    print("\nFeature Engineering")
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    engineer = FeatureEngineer(config)
    data = engineer.engineer(data)
    print(f"      Финальных признаков: {len(data.columns)}")

    print("\nДобавление финансовых меток")
    labeler = FinancialLabeler(config)
    data = labeler.add_financial_labels(data)
    data['total_financial_loss'] = pd.to_numeric(
        data['total_financial_loss'], errors='coerce'
    ).fillna(0.0).clip(lower=0.0)
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    data = data.dropna(subset=['total_financial_loss']).reset_index(drop=True)
    print(f"      Mean: ${data['total_financial_loss'].mean():,.0f}")
    print(f"      Max: ${data['total_financial_loss'].max():,.0f}")

    print("\nПрепроцессинг")
    data_copy = data.copy()
    preprocessor = Preprocessor(config)
    df_clean = preprocessor.clean_data(data_copy)
    labels = df_clean[config.data['target_column']].values.copy()
    processed = preprocessor.preprocess(data_copy)
    preprocessor.save()

    print("\nРазделение данных")
    X_train = processed['X_train']
    X_val = processed['X_val']
    X_test = processed['X_test']
    y_train = processed['y_train']
    y_val = processed['y_val']
    y_test = processed['y_test']
    feature_names = processed['feature_names']
    label_encoder = processed['label_encoder']

    print(f"      Train: {len(y_train):,}")
    print(f"      Val: {len(y_val):,}")
    print(f"      Test: {len(y_test):,}")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_names,
        'label_encoder': label_encoder,
        'df_clean': df_clean,
        'labeler': labeler
    }

def train_models(config: Config, data):
    print("обучение моделей")
    results = {}

    print("Random Forest Classification")
    start_time = time.time()
    try:
        clf_result = ModelTrainer.train_classification(
            RandomForestModel, config,
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            data['X_test'], data['y_test'],
            data['label_encoder'], data['feature_names']
        )
        clf_result['status'] = 'success'
        clf_result['train_time'] = time.time() - start_time
        print(f"      Accuracy: {clf_result['test_metrics']['accuracy']:.4f}")
        print(f"      F1 Score: {clf_result['test_metrics']['f1_score']:.4f}")
        print(f"      Time: {clf_result['train_time']:.1f}s")
    except Exception as e:
        print(f"      ERROR: {str(e)}")
        clf_result = {'status': 'failed', 'error': str(e)}
    results['classification'] = clf_result

    print("\nСоздание таблицы среднего ущерба по типу атаки")
    loss_by_label = data['labeler'].compute_loss_by_attack_type(data['df_clean'])
    loss_by_label['BENIGN'] = 0.0
    results['loss_by_label'] = loss_by_label

    return results

def create_visualizations(config: Config, results, data):
    print("СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    visualizer = Visualizer(config)

    clf_result = results.get('classification')
    if clf_result and clf_result['status'] == 'success':
        if clf_result.get('importance'):
            visualizer.plot_feature_importance(clf_result['importance'], 'classification')
            print("[OK] Feature importance: classification")

        y_pred_proba = clf_result['model_instance'].model.predict_proba(data['X_test'])
        visualizer.plot_roc_curve(
            data['y_test'], y_pred_proba,
            data['label_encoder'], 'classification'
        )
        print("[OK] ROC curve")

        visualizer.plot_confusion_matrix(
            clf_result['y_test'], clf_result['y_pred'],
            data['label_encoder'], 'classification'
        )
        print("[OK] Confusion matrix")

        visualizer.plot_confusion_matrix(
            clf_result['y_test'], clf_result['y_pred'],
            data['label_encoder'], 'classification', normalize=True
        )
        print("[OK] Normalized confusion matrix")

    print("[OK] Metrics comparison")
    visualizer.plot_metrics_comparison({'classification': clf_result})

def save_results(config: Config, results, data, elapsed_time):
    loss_by_label = results['loss_by_label']
    loss_path = Path(config.paths['models']) / 'loss_by_label.json'
    loss_path.parent.mkdir(parents=True, exist_ok=True)
    with open(loss_path, 'w', encoding='utf-8') as f:
        json.dump({str(k): float(v) for k, v in loss_by_label.items()}, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved loss lookup table: {loss_path}")

    metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_minutes': elapsed_time / 60,
        'data_stats': {
            'total_samples': len(data['df_clean']),
            'n_features': len(data['feature_names']),
            'n_classes': len(data['label_encoder'].classes_),
            'classes': data['label_encoder'].classes_.tolist()
        },
        'models': {}
    }

    if results['classification']['status'] == 'success':
        metadata['models']['classification'] = {
            k: float(v) if isinstance(v, (int, float, np.number)) else v
            for k, v in results['classification']['test_metrics'].items()
            if k != 'predictions'
        }
        metadata['models']['classification']['train_time_sec'] = results['classification']['train_time']

    results_dir = Path(config.paths['results']) / 'experiments'
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'results_{timestamp}.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return results_file

def main():
    start_time = time.time()
    config = Config('configs/full.yaml')
    data = prepare_data(config)
    results = train_models(config, data)
    create_visualizations(config, results, data)
    elapsed_time = time.time() - start_time
    results_file = save_results(config, results, data, elapsed_time)
    print(f"Общее время: {elapsed_time / 60:.2f} минут")
    print(f"Результаты: {results_file}")
    print(f"Графики: {config.paths['results']}/")
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")

if __name__ == "__main__":
    main()
