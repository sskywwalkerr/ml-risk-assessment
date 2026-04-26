from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib import colormaps as cm
from sklearn.metrics import confusion_matrix as sk_confusion_matrix  # type: ignore
from sklearn.metrics import roc_auc_score, roc_curve  # type: ignore
from sklearn.preprocessing import label_binarize  # type: ignore

plt.style.use("seaborn-v0_8-darkgrid")
_DPI = 150
_MAX_SCATTER_POINTS = 50_000  # Ограничение для scatter-графиков


class Visualizer:
    """Генерирует все графики результатов обучения."""

    def __init__(self, output_dir: str = "results") -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[str],
        model_name: str,
    ) -> None:
        """Нормализованная матрица ошибок."""
        cm = sk_confusion_matrix(y_true, y_pred, normalize="true")
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Предсказанный класс", fontsize=12)
        ax.set_ylabel("Истинный класс", fontsize=12)
        ax.set_title(f"{model_name} — Матрица ошибок", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self._dir / f"{model_name}_confusion_matrix.png", dpi=_DPI)
        plt.close()

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        labels: list[str],
        model_name: str,
    ) -> None:
        """ROC-кривые для многоклассовой классификации."""
        classes = list(range(len(labels)))
        y_bin = label_binarize(y_true, classes=classes)

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = cm["tab10"](np.linspace(0, 1, len(labels)))

        for i, (label, color) in enumerate(zip(labels, colors, strict=True)):
            if y_bin[:, i].sum() == 0:
                # Класс отсутствует в тестовой выборке - пропуск
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
            ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{label} (AUC = {auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Случайный классификатор")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(
            f"{model_name} — ROC-кривые (One-vs-Rest)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        plt.tight_layout()
        plt.savefig(self._dir / f"{model_name}_roc_curve.png", dpi=_DPI)
        plt.close()

    def feature_importance(self, importance: dict, model_name: str) -> None:
        """График важности признаков (топ-15)."""
        if not importance:
            return
        features = list(importance.keys())
        values = list(importance.values())
        colors = cm["viridis"](np.linspace(0.2, 0.9, len(features)))

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(features, values, color=colors)
        ax.set_xlabel("Важность признака", fontsize=12)
        ax.set_title(f"{model_name} — Топ признаков", fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(self._dir / f"{model_name}_feature_importance.png", dpi=_DPI)
        plt.close()

    def regression_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
    ) -> None:
        if len(y_true) > _MAX_SCATTER_POINTS:
            idx = np.random.choice(len(y_true), _MAX_SCATTER_POINTS, replace=False)
            y_true_plot = y_true[idx]
            y_pred_plot = y_pred[idx]
        else:
            y_true_plot = y_true
            y_pred_plot = y_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].scatter(y_true_plot, y_pred_plot, alpha=0.3, s=2, c="steelblue")
        m = max(float(y_true_plot.max()), float(y_pred_plot.max()))
        axes[0].plot([0, m], [0, m], "r--", lw=2, label="Идеальное предсказание")
        axes[0].set_xlabel("Фактические потери (RUB)")
        axes[0].set_ylabel("Предсказанные потери (RUB)")
        axes[0].set_title("Предсказание vs Факт", fontweight="bold")
        axes[0].legend(loc="upper left")

        residuals = y_pred - y_true
        axes[1].hist(residuals, bins=60, color="coral", edgecolor="black", alpha=0.7)
        axes[1].axvline(0, color="red", linestyle="--", lw=2)
        axes[1].set_xlabel("Остатки (RUB)")
        axes[1].set_ylabel("Частота")
        axes[1].set_title("Распределение остатков", fontweight="bold")

        plt.suptitle(f"{model_name} - Анализ регрессии", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self._dir / f"{model_name}_regression.png", dpi=_DPI)
        plt.close()

    def metrics_comparison(self, results: dict) -> None:
        """Сравнение Accuracy и F1 всех классификаторов."""
        names, acc, f1 = [], [], []
        for name, r in results.items():
            if r.get("status") == "success":
                m = r["test_metrics"]
                names.append(name.replace("_", "\n"))
                acc.append(m.get("accuracy", 0))
                f1.append(m.get("f1_score", 0))

        if not names:
            return

        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(10, 6))
        b1 = ax.bar(x - 0.2, acc, 0.4, label="Accuracy", color="steelblue")
        b2 = ax.bar(x + 0.2, f1, 0.4, label="F1 Score", color="coral")

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        ax.set_ylabel("Метрика", fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.set_title("Сравнение классификаторов", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)

        for bar, val in zip(list(b1) + list(b2), acc + f1, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(self._dir / "metrics_comparison.png", dpi=_DPI)
        plt.close()

    def risk_distribution(self, df: pd.DataFrame) -> None:
        """Средние финансовые потери по категориям атак."""
        if "total_financial_loss" not in df.columns:
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        avg = (
            df.groupby("label")["total_financial_loss"]
            .mean()
            .sort_values(ascending=True)
        )
        colors = cm["RdYlGn_r"](np.linspace(0.1, 0.9, len(avg)))
        avg.plot(kind="barh", ax=axes[0], color=colors)
        axes[0].set_xlabel("Средние потери (RUB)", fontsize=11)
        axes[0].set_title(
            "Средний ущерб по категории атаки", fontsize=12, fontweight="bold"
        )
        axes[0].xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"RUB{x:,.0f}")
        )

        risk_map = {
            "Mirai": "CRITICAL",
            "BruteForce": "HIGH",
            "DDoS": "HIGH",
            "DoS": "MEDIUM",
            "Spoofing": "MEDIUM",
            "Web-based": "MEDIUM",
            "Recon": "LOW",
            "BenignTraffic": "LOW",
        }
        risk_colors = {
            "CRITICAL": "#e74c3c",
            "HIGH": "#e67e22",
            "MEDIUM": "#f1c40f",
            "LOW": "#2ecc71",
        }
        risk_counts = df["label"].map(risk_map).fillna("LOW").value_counts()
        axes[1].pie(
            risk_counts.values,
            labels=risk_counts.index,
            colors=[risk_colors.get(k, "grey") for k in risk_counts.index],
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[1].set_title("Распределение уровней риска", fontsize=12, fontweight="bold")

        plt.suptitle(
            "Финансовый анализ рисков — CICIoT2023",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(self._dir / "risk_distribution.png", dpi=_DPI)
        plt.close()

    def label_distribution(self, label_dist: dict) -> None:
        """Распределение классов в датасете."""
        labels = list(label_dist.keys())
        counts = list(label_dist.values())
        colors = cm["tab10"](np.linspace(0, 1, len(labels)))

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(labels, counts, color=colors)
        ax.set_xlabel("Категория атаки", fontsize=12)
        ax.set_ylabel("Количество записей", fontsize=12)
        ax.set_title("Распределение меток - CICIoT2023", fontsize=14, fontweight="bold")
        plt.xticks(rotation=30, ha="right")

        for bar, count in zip(bars, counts, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(self._dir / "label_distribution.png", dpi=_DPI)
        plt.close()
