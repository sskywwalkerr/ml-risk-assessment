import logging
from collections import Counter

import numpy as np

from app.application.interactors.load_dataset import LoadDatasetRequest
from app.application.interactors.train_classifier import TrainClassifierRequest
from app.application.interactors.train_regressor import TrainRegressorRequest
from app.infrastructure.config import Config
from app.main.di.container import Container

logger = logging.getLogger(__name__)

CLASSIFIERS = ["random_forest", "lightgbm", "xgboost"]
REGRESSORS = ["random_forest_regressor", "lightgbm_regressor", "xgboost_regressor"]
TARGET_PER_CLASS = 15_000


class Pipeline:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._container = Container(config)

    def _balance(self, x_train, y_train):
        """Балансировка классов Undersampling"""
        counts = Counter(y_train)
        rng = np.random.default_rng(42)  # seed
        indices = []
        logger.info("Балансировка (цель: %d на класс):", TARGET_PER_CLASS)
        for cls, count in sorted(counts.items()):
            cls_idx = np.where(y_train == cls)[0]
            if count > TARGET_PER_CLASS:
                chosen = rng.choice(cls_idx, TARGET_PER_CLASS, replace=False)
                logger.info("класс %d: %d -> %d", cls, count, TARGET_PER_CLASS)
            else:
                chosen = cls_idx
                logger.info("класс %d: %d (без изменений)", cls, count)
            indices.append(chosen)
        all_idx = np.concatenate(indices)  # индексы всех классов в один массив
        rng.shuffle(all_idx)  # перемешать индексы
        c = Counter(y_train[all_idx])
        logger.info(
            "После: %d строк, дисбаланс: %.1fx",
            len(all_idx),
            max(c.values()) / min(c.values()),
        )
        return x_train[all_idx], y_train[all_idx]

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

        x_bal, y_bal = self._balance(
            splits.x_train, splits.y_train
        )  # балансировка train выборки

        logger.info("Обучение классификаторов.")
        clf_results: dict = {}

        for name in CLASSIFIERS:
            logger.info("\n %s ", name.upper())
            try:
                response = self._container.train_classifier(name)(
                    TrainClassifierRequest(
                        x_train=x_bal,
                        y_train=y_bal,
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
                    response.y_test,
                    response.y_pred,
                    labels=classes,
                    model_name=name,
                )
                viz.feature_importance(response.feature_importance, name)
                if response.y_prob is not None:
                    viz.plot_roc_curve(
                        response.y_test,
                        response.y_prob,
                        labels=classes,
                        model_name=name,
                    )

            except Exception as e:
                logger.exception("Ошибка при обучении %s: %s", name, e)
                clf_results[name] = {"status": "error", "error": str(e)}

        viz.metrics_comparison(clf_results, kind="classification")

        logger.info("\nИтоги классификаторов")
        for name, r in clf_results.items():
            if r["status"] == "success":
                m = r["test_metrics"]
                logger.info(
                    "%-25s Accuracy: %.4f | F1: %.4f",
                    name,
                    m["accuracy"],
                    m["f1_score"],
                )

        logger.info("\nОбучение регрессоров.")
        self._run_regressors(splits, preprocessor, feature_names, viz)

    def _run_regressors(self, splits, preprocessor, feature_names, viz) -> None:
        fin_col = "total_financial_loss"  # предсказание общих финансовых потерь
        clean_df = preprocessor.clean_df
        if clean_df is None or fin_col not in clean_df.columns:
            logger.info("Пропущено: столбец '%s' не найден", fin_col)
            return

        fin = clean_df[fin_col].values  # массив целевой переменной
        y_fin_train = fin[preprocessor.train_idx]
        y_fin_val = fin[preprocessor.val_idx]
        y_fin_test = fin[preprocessor.test_idx]

        reg_results: dict = {}

        for name in REGRESSORS:
            logger.info("\n %s ", name.upper())
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
                logger.info("MAE:  %12.2f руб", m["mae"])
                logger.info("RMSE: %12.2f руб", m["rmse"])
                logger.info("R2:   %13.4f", m["r2"])
                logger.info("MAPE: %11.2f%%", m["mape"])
                reg_results[name] = {"status": "success", "test_metrics": m}

                viz.regression_analysis(response.y_test, response.y_pred, name)

            except Exception as e:
                logger.exception("Ошибка регрессора %s: %s", name, e)
                reg_results[name] = {"status": "error", "error": str(e)}

        viz.metrics_comparison(reg_results, kind="regression")

        logger.info("\nИтог регрессоров")
        for name, r in reg_results.items():
            if r["status"] == "success":
                m = r["test_metrics"]
                logger.info(
                    "%-25s MAE: %10.2f | R2: %.4f | MAPE: %.2f%%",
                    name,
                    m["mae"],
                    m["r2"],
                    m["mape"],
                )
