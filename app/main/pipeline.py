import logging

from app.application.interactors.load_dataset import LoadDatasetRequest
from app.application.interactors.train_classifier import TrainClassifierRequest
from app.application.interactors.train_regressor import TrainRegressorRequest
from app.infrastructure.config import Config
from app.main.di.container import Container

logger = logging.getLogger(__name__)

CLASSIFIERS = ["random_forest", "xgboost", "lightgbm"]


class Pipeline:
    """Оркестрирует полный ML-пайплайн через interactors."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._container = Container(config)

    def run(self) -> None:
        logger.info("Загрузка данных.")
        load_resp = self._container.load_dataset()(
            LoadDatasetRequest(sample_size=self._config.data.sample_size)
        )
        df = load_resp.df
        logger.info(f"Строк: {load_resp.n_rows:}, столбцов: {load_resp.n_cols:}")

        viz = self._container.visualizer()
        viz.label_distribution(load_resp.label_distribution)
        viz.risk_distribution(df)

        logger.info("Предобработка.")
        preprocessor = self._container.preprocessor()
        splits = preprocessor.preprocess(df)
        preprocessor.save("models/metadata")
        logger.info(
            f"Train: {len(splits.x_train):,} | Val: {len(splits.x_val):,} | Test: {len(splits.x_test):,}"
        )
        logger.info(
            f"Признаков: {len(splits.feature_names)} | Классов: {splits.n_classes}"
        )

        feature_names = list(splits.feature_names)
        label_encoder = splits.label_encoder
        classes = list(label_encoder.classes_)

        logger.info("Обучение классификаторов.")
        clf_results: dict = {}

        for name in CLASSIFIERS:
            logger.info(f"\n {name.upper()}")
            try:
                response = self._container.train_classifier(name)(
                    TrainClassifierRequest(
                        x_train=splits.x_train,
                        y_train=splits.y_train,
                        x_val=splits.x_val,
                        y_val=splits.y_val,
                        x_test=splits.x_test,
                        y_test=splits.y_test,
                        feature_names=feature_names,
                        model_name=name,
                    )
                )
                m = response.test_metrics
                logger.info(f"Accuracy: {m['accuracy']:.4f}")
                logger.info(f"F1-score: {m['f1_score']:.4f}")

                clf_results[name] = {"status": "success", "test_metrics": m}

                viz.confusion_matrix(
                    response.y_test, response.y_pred, labels=classes, model_name=name
                )
                viz.feature_importance(response.feature_importance, name)
            except Exception as e:
                logger.exception("Ошибка при обучении %s, ошибка: %s", name, e)
                clf_results[name] = {"status": "error", "error": str(e)}
        viz.metrics_comparison(clf_results)

        logger.info("Обучение регрессора.")
        self._run_regressor(df, splits, preprocessor, feature_names, viz)

        for name, r in clf_results.items():
            if r["status"] == "success":
                m = r["test_metrics"]
                logger.info(
                    f"{name:<20} Accuracy: {m['accuracy']:.4f}, F1-score: {m['f1_score']:.4f}"
                )

    def _run_regressor(self, df, splits, preprocessor, feature_names, viz) -> None:
        """Запускает регрессор на финансовых метках."""
        fin_col = "total_financial_loss"
        if fin_col not in df.columns:
            logger.info("Пропущено: столбец 'total_financial_loss' не найден")
            return

        # Все финансовые метки из очищенного DataFrame
        fin = df[fin_col].values

        # Нарезает по тем же индексам, что preprocessor использовал для x и y
        y_fin_train = fin[preprocessor.train_idx]
        y_fin_val = fin[preprocessor.val_idx]
        y_fin_test = fin[preprocessor.test_idx]

        try:
            response = self._container.train_regressor()(
                TrainRegressorRequest(
                    x_train=splits.x_train,
                    y_train=y_fin_train,
                    x_val=splits.x_val,
                    y_val=y_fin_val,
                    x_test=splits.x_test,
                    y_test=y_fin_test,
                    feature_names=feature_names,
                    model_name="xgboost_regressor",
                )
            )
            m = response.test_metrics
            logger.info(f"MAE: {m['mae']:>12,.2f}")
            logger.info(f"RMSE: {m['rmse']:>12,.2f}")
            logger.info(f"R2: {m['r2']:>13,.4f}")
            logger.info(f"MAPE: {m['mape']:>12,.2f}%")

            viz.regression_analysis(
                response.y_test, response.y_pred, "xgboost_regressor"
            )
            viz.feature_importance(response.feature_importance, "xgboost_regressor")

        except Exception as e:
            logger.exception("Ошибка регрессора: %s", e)
