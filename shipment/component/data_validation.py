from shipment.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from shipment.entity.config_entity import DataValidationConfig
from shipment.exception import ShipmentException
from shipment.utils.util import read_yaml_file
from shipment.logger import logging
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import pandas as pd
import sys, os
import json


class DataValidation:

    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info(f"{'>>' * 20}Data Validation log started.{'<<' * 20} ")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def get_train_and_test_df(self):
        try:
            train_df = self.data_ingestion_artifact.train_file_path
            test_df = self.data_ingestion_artifact.test_file_path
            return train_df, test_df
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def is_train_test_file_exists(self) -> bool:
        try:
            logging.info("Checking if training and test file is available")

            is_train_file_exist = os.path.exists(self.data_ingestion_artifact.train_file_path)
            is_test_file_exist = os.path.exists(self.data_ingestion_artifact.test_file_path)

            is_available = is_train_file_exist and is_test_file_exist

            logging.info(f"Is train and test file exists?-> {is_available}")

            if not is_available:
                training_file = self.data_ingestion_artifact.train_file_path
                testing_file = self.data_ingestion_artifact.test_file_path
                message = f"Training file: {training_file} or Testing file: {testing_file}" \
                          "is not present"
                raise Exception(message)

            return is_available
        except Exception as e:
            raise ShipmentException(e, sys) from e

    @staticmethod
    def check_columns_details(file_name, config) -> bool:
        try:
            read_file = pd.read_csv(file_name)
            logging.info(f"Reading {file_name} file to check columns")
            file_columns = read_file.columns
            config_columns = config["columns"].keys()
            column_name = None
            column_type = None
            for i in file_columns:
                if i in config_columns:
                    column_name = True
                else:
                    logging.error(f'{i} column is missing from {file_name} file')
                    column_name = False
                    break

            for i in file_columns:
                if config["columns"][i].replace(",", "") == read_file[i].dtype:
                    column_type = True
                else:
                    logging.error(f"column type of {i} in {file_name} is not matching with schema config")
                    column_type = False
                    break

            check = column_name and column_type
            return check
        except Exception as e:
            raise ShipmentException(e, sys) from e

    @staticmethod
    def check_num_and_target_col(file_name, config) -> bool:
        try:
            read_file = pd.read_csv(file_name)
            num_col_file = len(read_file.columns)
            config_col_num = config["number_of_column"]
            logging.info(f"Checking number of columns in {file_name}")
            if num_col_file == config_col_num:
                num_check = True
            else:
                logging.error(f"Number of columns in {file_name} file is not matching with Schema config"
                              f"Required is {config_col_num}, but in {file_name} file is {num_col_file}")
                num_check = False

            target_col = config["target_column"]
            logging.info(f"Checking Target columns in {file_name}")
            if target_col in read_file.columns:
                target_check = True
            else:
                logging.error(f"In {file_name} file Target column is not Matching with Schema Config")
                target_check = False

            result = num_check and target_check
            return result
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def validate_dataset_schema(self) -> bool:
        try:
            config = read_yaml_file(self.data_validation_config.schema_file_path)
            train_df, test_df = self.get_train_and_test_df()

            train_col_check = self.check_columns_details(train_df, config)
            train_num_target = self.check_num_and_target_col(train_df, config)
            test_col_check = self.check_columns_details(test_df, config)
            test_num_target = self.check_num_and_target_col(test_df, config)

            result = train_col_check and train_num_target and test_num_target and test_col_check
            return result

        except Exception as e:
            raise ShipmentException(e, sys) from e

    def get_and_save_data_drift_report(self):
        try:
            profile = Profile(sections=[DataDriftProfileSection()])

            train_df_str, test_df_str = self.get_train_and_test_df()
            train_df = pd.read_csv(train_df_str)
            test_df = pd.read_csv(test_df_str)

            profile.calculate(train_df, test_df)

            report = json.loads(profile.json())

            report_file_path = self.data_validation_config.report_file_path
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            with open(report_file_path, "w") as report_file:
                json.dump(report, report_file, indent=6)
            return report
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def save_data_drift_report_page(self):
        try:
            dashboard = Dashboard(tabs=[DataDriftTab()])
            train_df_str, test_df_str = self.get_train_and_test_df()
            train_df = pd.read_csv(train_df_str)
            test_df = pd.read_csv(test_df_str)
            dashboard.calculate(train_df, test_df)

            report_page_file_path = self.data_validation_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir, exist_ok=True)

            dashboard.save(report_page_file_path)
        except Exception as e:
            raise ShipmentException(e, sys)from e

    def is_data_drift_found(self) -> bool:
        try:
            self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()
            return True
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            self.is_train_test_file_exists()
            self.validate_dataset_schema()
            self.is_data_drift_found()

            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.data_validation_config.schema_file_path,
                report_file_path=self.data_validation_config.report_file_path,
                report_page_file_path=self.data_validation_config.report_page_file_path,
                is_validated=True,
                message="Data Validation performed successfully."
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 20}Data Validation log completed.{'<<' * 20} \n\n")
