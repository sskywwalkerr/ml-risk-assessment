import logging
from collections import Counter

import numpy as np

from app.application.interactors.load_dataset import LoadDatasetRequest
from app.application.interactors.train_classifier import TrainClassifierRequest
from app.application.interactors.train_regressor import TrainRegressorRequest
from app.infrastructure.config import Config
from app.main.di.container import Container

logger = logging.getLogger(__name__)

CLASSIFIERS = ["xgboost"]
REGRESSORS = ["xgboost_regressor"]


MAX_SAMPLES_PER_CLASS = 50_000


class Pipeline:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._container = Container(config)

    def _downsample(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Даунсэмплинг доминирующих классов (DDoS, DoS) до MAX_SAMPLES_PER_CLASS.
        Редкие классы (BruteForce, Web-based) оставляем полностью — их и так мало.
        Это лучше SMOTE для экстремального дисбаланса: не генерируем шум,
        а просто убираем избыток доминирующих классов.
        """
        counts = Counter(y_train)
        logger.info("Даунсэмплинг: порог %d записей на класс.", MAX_SAMPLES_PER_CLASS)

        indices = []
        rng = np.random.default_rng(42)

        for cls, count in sorted(counts.items()):
            cls_indices = np.where(y_train == cls)[0]
            if count > MAX_SAMPLES_PER_CLASS:
                chosen = rng.choice(cls_indices, MAX_SAMPLES_PER_CLASS, replace=False)
                logger.info(
                    "  класс %d: %d -> %d (даунсэмплинг)",
                    cls,
                    count,
                    MAX_SAMPLES_PER_CLASS,
                )
            else:
                chosen = cls_indices
                logger.info("  класс %d: %d (без изменений)", cls, count)
            indices.append(chosen)

        all_indices = np.concatenate(indices)
        rng.shuffle(all_indices)

        x_bal = x_train[all_indices]
        y_bal = y_train[all_indices]

        logger.info(
            "После даунсэмплинга: %d строк (было %d). Дисбаланс: %.1fx",
            len(y_bal),
            len(y_train),
            max(Counter(y_bal).values()) / min(Counter(y_bal).values()),
        )
        return x_bal, y_bal

    def run(self) -> None:
        logger.info("Загрузка данных.")
        load_resp = self._container.load_dataset()(
            LoadDatasetRequest(sample_size=self._config.data.sample_size)
        )
        df = load_resp.df
        logger.info("Строк: %d, столбцов: %d", load_resp.n_rows, load_resp.n_cols)

        viz = self._container.visualizer()
        viz.label_distribution(load_resp.label_distribution)
        viz.risk_distribution(df)

        logger.info("Предобработка.")
        preprocessor = self._container.preprocessor()
        splits = preprocessor.preprocess(df)
        preprocessor.save("models/metadata")
        logger.info(
            "Train: %s | Val: %s | Test: %s",
            f"{len(splits.x_train):,}",
            f"{len(splits.x_val):,}",
            f"{len(splits.x_test):,}",
        )
        logger.info(
            "Признаков: %d | Классов: %d", len(splits.feature_names), splits.n_classes
        )

        feature_names = list(splits.feature_names)
        classes = list(splits.label_encoder.classes_)

        x_train_bal, y_train_bal = self._downsample(splits.x_train, splits.y_train)

        logger.info("Обучение классификаторов.")
        clf_results: dict = {}

        for name in CLASSIFIERS:
            logger.info("\n %s", name.upper())
            try:
                interactor = self._container.train_classifier(name)
                response = interactor(
                    TrainClassifierRequest(
                        x_train=x_train_bal,
                        y_train=y_train_bal,
                        x_val=splits.x_val,
                        y_val=splits.y_val,
                        x_test=splits.x_test,
                        y_test=splits.y_test,
                        feature_names=feature_names,
                        model_name=name,
                    )
                )
                m = response.test_metrics
                logger.info("Accuracy: %.4f", m["accuracy"])
                logger.info("F1-score: %.4f", m["f1_score"])
                clf_results[name] = {"status": "success", "test_metrics": m}

                viz.confusion_matrix(
                    response.y_test, response.y_pred, labels=classes, model_name=name
                )
                viz.feature_importance(response.feature_importance, name)

                if response.y_prob is not None:
                    viz.plot_roc_curve(
                        response.y_test,
                        response.y_prob,
                        labels=classes,
                        model_name=name,
                    )
                    logger.info("ROC-кривая сохранена для %s", name)

            except Exception as e:
                logger.exception("Ошибка при обучении %s: %s", name, e)
                clf_results[name] = {"status": "error", "error": str(e)}

        viz.metrics_comparison(clf_results)

        logger.info("Обучение регрессора.")
        self._run_regressor(splits, preprocessor, feature_names, viz)

        for name, r in clf_results.items():
            if r["status"] == "success":
                m = r["test_metrics"]
                logger.info(
                    "%-20s Accuracy: %.4f, F1-score: %.4f",
                    name,
                    m["accuracy"],
                    m["f1_score"],
                )

    def _run_regressor(self, splits, preprocessor, feature_names, viz) -> None:
        fin_col = "total_financial_loss"
        clean_df = preprocessor.clean_df
        if clean_df is None or fin_col not in clean_df.columns:
            logger.info("Пропущено: столбец '%s' не найден", fin_col)
            return

        fin = clean_df[fin_col].values
        y_fin_train = fin[preprocessor.train_idx]
        y_fin_val = fin[preprocessor.val_idx]
        y_fin_test = fin[preprocessor.test_idx]

        for name in REGRESSORS:
            logger.info("\n %s", name.upper())
            try:
                response = self._container.train_regressor(name)(
                    TrainRegressorRequest(
                        x_train=splits.x_train,
                        y_train=y_fin_train,
                        x_val=splits.x_val,
                        y_val=y_fin_val,
                        x_test=splits.x_test,
                        y_test=y_fin_test,
                        feature_names=feature_names,
                        model_name=name,
                    )
                )
                m = response.test_metrics
                logger.info("MAE:  %12.2f", m["mae"])
                logger.info("RMSE: %12.2f", m["rmse"])
                logger.info("R2:   %13.4f", m["r2"])
                logger.info("MAPE: %11.2f%%", m["mape"])

                viz.regression_analysis(response.y_test, response.y_pred, name)
                viz.feature_importance(response.feature_importance, name)

            except Exception as e:
                logger.exception("Ошибка регрессора %s: %s", name, e)
