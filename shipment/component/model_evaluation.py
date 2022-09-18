from shipment.logger import logging
from shipment.exception import ShipmentException
from shipment.entity.config_entity import ModelEvaluationConfig
from shipment.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, ModelTrainerArtifact, \
    ModelEvaluationArtifact
from shipment.constant import *
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import sys
from shipment.utils.util import write_yaml_file, read_yaml_file, load_object, load_data
from shipment.entity.model_factory import evaluate_regression_model


def transform(X: pd.DataFrame):
    try:
        df: pd.DataFrame = X.copy()

        columns_remove = ['ID', 'PQ First Sent to Client Date', 'PO Sent to Vendor Date', 'Weight (Kilograms)',
                          'Freight Cost (USD)', 'PQ #', 'PO / SO #', 'ASN/DN #']
        for i in columns_remove:
            if i in df:
                df.drop(i, axis=1, inplace=True)
                logging.info(f"Dropped column {i} from {str(X)}")

        for column in ['Scheduled Delivery Date', 'Delivered to Client Date', 'Delivery Recorded Date']:
            if column in df:
                df[column] = pd.to_datetime(df[column])
                df[column + ' Year'] = df[column].apply(lambda x: x.year)
                df[column + ' Month'] = df[column].apply(lambda x: x.month)
                df[column + ' Day'] = df[column].apply(lambda x: x.day)
                df = df.drop(column, axis=1)
                logging.info(f"Converting date column ")

        binary_columns = ['Fulfill Via', 'First Line Designation']
        for i in binary_columns:
            if i in df:
                df['Fulfill Via'] = df['Fulfill Via'].replace({'Direct Drop': 0, 'From RDC': 1})
                df['First Line Designation'] = df['First Line Designation'].replace({'No': 0, 'Yes': 1})

        # Fill missing values
        df['Shipment Mode'] = df['Shipment Mode'].fillna(df['Shipment Mode'].mode()[0])
        df['Dosage'] = df['Dosage'].fillna(df['Dosage'].mode()[0])
        df['Line Item Insurance (USD)'] = df['Line Item Insurance (USD)'].fillna(
            df['Line Item Insurance (USD)'].mean())

        object_type = ['Country', 'Managed By', 'Vendor INCO Term',
                       'Shipment Mode', 'Product Group', 'Sub Classification', 'Vendor',
                       'Item Description', 'Molecule/Test Type', 'Brand', 'Dosage',
                       'Dosage Form', 'Manufacturing Site', 'Project Code']
        # One-hot encoding
        for column in object_type:
            dummies = pd.get_dummies(df[column], prefix=column)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(column, axis=1)

        train, test = train_test_split(df, test_size=0.2, random_state=42)
        return train, test
    except Exception as e:
        raise ShipmentException(e, sys)


class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            logging.info(f"{'>>' * 30}Model Evaluation log started.{'<<' * 30} ")
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def get_best_model(self):
        try:
            model = None
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path

            if not os.path.exists(model_evaluation_file_path):
                write_yaml_file(file_path=model_evaluation_file_path,
                                )
                return model
            model_eval_file_content = read_yaml_file(file_path=model_evaluation_file_path)

            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content

            if BEST_MODEL_KEY not in model_eval_file_content:
                return model

            model = load_object(file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def update_evaluation_report(self, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            model_eval_content = read_yaml_file(file_path=eval_file_path)
            model_eval_content = dict() if model_eval_content is None else model_eval_content

            previous_best_model = None
            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model = model_eval_content[BEST_MODEL_KEY]

            logging.info(f"Previous eval result: {model_eval_content}")
            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path,
                }
            }

            if previous_best_model is not None:
                model_history = {self.model_evaluation_config.time_stamp: previous_best_model}
                if HISTORY_KEY not in model_eval_content:
                    history = {HISTORY_KEY: model_history}
                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)

            model_eval_content.update(eval_result)
            logging.info(f"Updated eval result:{model_eval_content}")
            write_yaml_file(file_path=eval_file_path, data=model_eval_content)

        except Exception as e:
            raise ShipmentException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            trained_model_object = load_object(file_path=trained_model_file_path)

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            schema_file_path = self.data_validation_artifact.schema_file_path

            train_df = load_data(file_path=train_file_path,
                                 schema_file_path=schema_file_path,
                                 )
            test_df = load_data(file_path=test_file_path,
                                schema_file_path=schema_file_path,
                                )

            df = pd.concat(objs=[train_df, test_df], axis=0)
            train_dataframe, test_dataframe = transform(df)

            schema_content = read_yaml_file(file_path=schema_file_path)
            target_column_name = schema_content[TARGET_COLUMNS_KEY]

            # target_column
            logging.info(f"Converting target column into numpy array.")
            train_target_arr = np.array(train_dataframe[target_column_name])
            test_target_arr = np.array(test_dataframe[target_column_name])
            logging.info(f"Conversion completed target column into numpy array.")

            # dropping target column from the dataframe
            logging.info(f"Dropping target column from the dataframe.")
            train_dataframe.drop(target_column_name, axis=1, inplace=True)
            test_dataframe.drop(target_column_name, axis=1, inplace=True)
            logging.info(f"Dropping target column from the dataframe completed.")

            model = self.get_best_model()

            if model is None:
                logging.info("Not found any existing model. Hence accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")
                return model_evaluation_artifact

            model_list = [model, trained_model_object]

            metric_info_artifact = evaluate_regression_model(model_list=model_list,
                                                             X_train=train_dataframe,
                                                             y_train=train_target_arr,
                                                             X_test=test_dataframe,
                                                             y_test=test_target_arr,
                                                             base_accuracy=self.model_trainer_artifact.model_accuracy,
                                                             )
            logging.info(f"Model evaluation completed. model metric artifact: {metric_info_artifact}")

            if metric_info_artifact is None:
                response = ModelEvaluationArtifact(is_model_accepted=False,
                                                   evaluated_model_path=trained_model_file_path
                                                   )
                logging.info(response)
                return response

            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")

            else:
                logging.info("Trained model is no better than existing model hence not accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=False)
            return model_evaluation_artifact
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def __del__(self):
        logging.info(f"{'=' * 20}Model Evaluation log completed.{'=' * 20} ")
