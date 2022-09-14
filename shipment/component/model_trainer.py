import pandas as pd

from shipment.exception import ShipmentException
import sys
from shipment.logger import logging
from typing import List
from shipment.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from shipment.entity.config_entity import ModelTrainerConfig
from shipment.utils.util import save_object, load_object
from shipment.entity.model_factory import MetricInfoArtifact, ModelFactory, GridSearchedBestModel
from shipment.entity.model_factory import evaluate_regression_model


class ShipmentEstimatorModel:
    def __init__(self, preprocessing_object, trained_model_object):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it performs prediction on transformed features
        """
        transformed_feature = X # self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(transformed_feature)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:

    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(f"Loading transformed training dataset for both target columns")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_dir
            X_train_t1 = pd.read_csv(transformed_train_file_path + r"\X_train_t1.csv")
            y_train_t1 = pd.read_csv(transformed_train_file_path + r"\y_train_t1.csv")
            X_train_t2 = pd.read_csv(transformed_train_file_path + r"\X_train_t2.csv")
            y_train_t2 = pd.read_csv(transformed_train_file_path + r"\y_train_t2.csv")

            logging.info(f"Loading transformed testing dataset for both target columns")
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_dir
            X_test_t1 = pd.read_csv(transformed_test_file_path + r"\X_test_t1.csv")
            y_test_t1 = pd.read_csv(transformed_test_file_path + r"\y_test_t1.csv")
            X_test_t2 = pd.read_csv(transformed_test_file_path + r"\X_test_t2.csv")
            y_test_t2 = pd.read_csv(transformed_test_file_path + r"\y_test_t2.csv")

            logging.info(f"Extracting model config file path")
            model_config_file_path = self.model_trainer_config.model_config_file_path

            logging.info(f"Initializing model factory class using above model config file: {model_config_file_path}")
            model_factory = ModelFactory(model_config_path=model_config_file_path)

            base_accuracy = self.model_trainer_config.base_accuracy
            logging.info(f"Expected accuracy: {base_accuracy}")

            logging.info(f"Initiating operation model selection for target 1")
            best_model_target_1 = model_factory.get_best_model(X=X_train_t1, y=y_train_t1, base_accuracy=base_accuracy)
            logging.info(f"Best model found on training dataset for target 1 : {best_model_target_1}")

            logging.info(f"Extracting trained model list.")
            grid_searched_best_model_list_target_1: \
                List[GridSearchedBestModel] = model_factory.grid_searched_best_model_list

            logging.info(f"Initiating operation model selection for target 2")
            best_model_target_2 = model_factory.get_best_model(X=X_train_t2, y=y_train_t2, base_accuracy=base_accuracy)

            logging.info(f"Best model found on training dataset for target 2: {best_model_target_2}")

            logging.info(f"Extracting trained model list.")
            grid_searched_best_model_list_target_2: \
                List[GridSearchedBestModel] = model_factory.grid_searched_best_model_list

            model_list_target_1 = [model.best_model for model in grid_searched_best_model_list_target_1]
            logging.info(f"Evaluation all trained model on training and testing dataset both for target 1")
            metric_info_target_1 = evaluate_regression_model(
                model_list=model_list_target_1, X_train=X_train_t1,
                y_train=y_train_t1, X_test=X_test_t1,
                y_test=y_test_t1,
                base_accuracy=base_accuracy)
            logging.info(f"Best found model on both training and testing dataset for target 1.")

            model_list_target_2 = [model.best_model for model in grid_searched_best_model_list_target_2]
            logging.info(f"Evaluation all trained model on training and testing dataset both for target 2")
            metric_info_target_2 = evaluate_regression_model(
                model_list=model_list_target_2, X_train=X_train_t2,
                y_train=y_train_t2, X_test=X_test_t2,
                y_test=y_test_t2,
                base_accuracy=base_accuracy)

            logging.info(f"Best found model on both training and testing dataset for target 2.")

            preprocessing_obj_target_1 = load_object(
                file_path=self.data_transformation_artifact.preprocessed_object_file_path.replace(
                    ".pkl", "_target_1.pkl"))
            preprocessing_obj_target_2 = load_object(
                file_path=self.data_transformation_artifact.preprocessed_object_file_path.replace(
                    ".pkl", "_target_2.pkl"))

            model_object_target_1 = metric_info_target_1.model_object
            model_object_target_2 = metric_info_target_2.model_object

            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            shipment_model_target_1 = ShipmentEstimatorModel(preprocessing_object=preprocessing_obj_target_1,
                                                             trained_model_object=model_object_target_1)
            logging.info(f"Saving model at path: {trained_model_file_path}")
            save_object(file_path=trained_model_file_path.replace(".pkl", "_1.pkl"), obj=shipment_model_target_1)

            shipment_model_target_2 = ShipmentEstimatorModel(preprocessing_object=preprocessing_obj_target_2,
                                                             trained_model_object=model_object_target_2)
            logging.info(f"Saving model at path: {trained_model_file_path}")
            save_object(file_path=trained_model_file_path.replace(".pkl", "_2.pkl"), obj=shipment_model_target_2)

            model_trainer_artifact = ModelTrainerArtifact(is_trained=True, message="Model Trained successfully",
                                                          trained_model_file_path_t1=trained_model_file_path.replace(
                                                              ".pkl", "_1.pkl"),
                                                          train_rmse_1=metric_info_target_1.train_rmse,
                                                          test_rmse_1=metric_info_target_1.test_rmse,
                                                          train_accuracy_1=metric_info_target_1.train_accuracy,
                                                          test_accuracy_1=metric_info_target_1.test_accuracy,
                                                          model_accuracy_1=metric_info_target_1.model_accuracy,
                                                          trained_model_file_path_t2=trained_model_file_path.replace(
                                                              ".pkl", "_2.pkl"),
                                                          train_rmse_2=metric_info_target_2.train_rmse,
                                                          test_rmse_2=metric_info_target_2.test_rmse,
                                                          train_accuracy_2=metric_info_target_2.train_accuracy,
                                                          test_accuracy_2=metric_info_target_2.test_accuracy,
                                                          model_accuracy_2=metric_info_target_2.model_accuracy
                                                          )

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 30}Model trainer log completed.{'<<' * 30} ")
