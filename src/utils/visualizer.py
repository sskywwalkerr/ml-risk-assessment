import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict
from sklearn.metrics import roc_curve, auc, confusion_matrix

from src.utils.config import Config


class Visualizer:
    def __init__(self, config: Config):
        self.config = config
        self.results_dir = Path(config.paths['results'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_training_history(self, history: Dict, model_name: str):
        """График истории обучения"""
        if 'train_loss' not in history or not history['train_loss']:
            return

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)

        if 'val_loss' in history and history['val_loss']:
            plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'{model_name} - Training History', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.results_dir / f'{model_name}_training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, importance: Dict, model_name: str):
        """График важности признаков"""
        if not importance:
            return

        features = list(importance.keys())
        values = list(importance.values())

        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        plt.barh(features, values, color=colors)

        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'{model_name} - Feature Importance', fontsize=14, fontweight='bold')
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()

        save_path = self.results_dir / f'{model_name}_feature_importance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       label_encoder, model_name: str = 'classification'):
        """ROC кривая для мультиклассовой классификации"""
        n_classes = len(label_encoder.classes_)

        plt.figure(figsize=(12, 9))
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))

        for i, (class_name, color) in enumerate(zip(label_encoder.classes_, colors)):
            y_true_binary = (y_true == i).astype(int)
            y_score = y_pred_proba[:, i]

            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f'{class_name} (AUC={roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-class Classification', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=9, ncol=2)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        save_path = self.results_dir / f'{model_name}_roc_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              label_encoder, model_name: str = 'classification',
                              normalize: bool = False):
        """Матрица ошибок"""
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'

        plt.figure(figsize=(14, 12))

        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_,
                    cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                    square=True)

        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        suffix = '_normalized' if normalize else ''
        save_path = self.results_dir / f'{model_name}_confusion_matrix{suffix}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_regression_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 model_name: str = 'regression'):
        """Анализ регрессии: predicted vs actual + residuals"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        axes[0, 0].scatter(y_true, y_pred, alpha=0.3, s=2, c='blue')
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 0].plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Loss ($)', fontsize=11)
        axes[0, 0].set_ylabel('Predicted Loss ($)', fontsize=11)
        axes[0, 0].set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        residuals = y_pred - y_true
        axes[0, 1].scatter(y_pred, residuals, alpha=0.3, s=2, c='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Loss ($)', fontsize=11)
        axes[0, 1].set_ylabel('Residuals ($)', fontsize=11)
        axes[0, 1].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 0].axvline(0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residuals ($)', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)

        bins = np.percentile(y_true, [0, 25, 50, 75, 100])
        bin_labels = ['Q1', 'Q2', 'Q3', 'Q4']
        bin_indices = np.digitize(y_true, bins[1:-1])

        errors_by_bin = [np.abs(residuals[bin_indices == i]) for i in range(len(bin_labels))]

        axes[1, 1].boxplot(errors_by_bin, labels=bin_labels, patch_artist=True)
        axes[1, 1].set_xlabel('Value Quartile', fontsize=11)
        axes[1, 1].set_ylabel('Absolute Error ($)', fontsize=11)
        axes[1, 1].set_title('Error Distribution by Quartile', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        save_path = self.results_dir / f'{model_name}_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_metrics_comparison(self, results: Dict):
        """Сравнение метрик моделей"""
        models = list(results.keys())
        classification_models = [m for m in models if 'classifier' in m or 'classification' in m]
        regression_models = [m for m in models if 'regressor' in m or 'regression' in m]

        if classification_models:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            accuracies = []
            f1_scores = []
            model_names = []

            for model in classification_models:
                if results[model]['status'] == 'success':
                    test_metrics = results[model]['test_metrics']
                    accuracies.append(test_metrics.get('accuracy', 0))
                    f1_scores.append(test_metrics.get('f1_score', 0))
                    model_names.append(model.replace('_', ' ').title())

            x = np.arange(len(model_names))
            width = 0.35

            ax1.bar(x - width / 2, accuracies, width, label='Accuracy', color='skyblue')
            ax1.bar(x + width / 2, f1_scores, width, label='F1 Score', color='lightcoral')

            ax1.set_xlabel('Models', fontsize=12)
            ax1.set_ylabel('Score', fontsize=12)
            ax1.set_title('Classification Metrics', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(model_names, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, axis='y', alpha=0.3)

            for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
                ax1.text(i - width / 2, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
                ax1.text(i + width / 2, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom', fontsize=9)

        if regression_models:
            if not classification_models:
                fig, ax2 = plt.subplots(1, 1, figsize=(10, 6))

            mae_scores = []
            r2_scores = []
            model_names = []

            for model in regression_models:
                if results[model]['status'] == 'success':
                    test_metrics = results[model]['test_metrics']
                    mae_scores.append(test_metrics.get('mae', 0))
                    r2_scores.append(test_metrics.get('r2', 0))
                    model_names.append(model.replace('_', ' ').title())

            x = np.arange(len(model_names))
            width = 0.35

            ax2_twin = ax2.twinx()

            ax2.bar(x - width / 2, mae_scores, width, label='MAE', color='lightgreen')
            ax2_twin.bar(x + width / 2, r2_scores, width, label='R² Score', color='orange')

            ax2.set_xlabel('Models', fontsize=12)
            ax2.set_ylabel('MAE', fontsize=12, color='green')
            ax2_twin.set_ylabel('R² Score', fontsize=12, color='orange')
            ax2.set_title('Regression Metrics', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(model_names, rotation=45, ha='right')
            ax2.grid(True, axis='y', alpha=0.3)

            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        save_path = self.results_dir / 'metrics_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_all_results(self, results: Dict):
        """Создание всех графиков для результатов"""
        for model_name, result in results.items():
            if result['status'] == 'success':
                if 'history' in result and result['history']:
                    self.plot_training_history(result['history'], model_name)

                if 'importance' in result and result['importance']:
                    self.plot_feature_importance(result['importance'], model_name)

        self.plot_metrics_comparison(results)
