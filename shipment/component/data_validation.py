from shipment.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from shipment.entity.config_entity import DataValidationConfig
from shipment.exception import ShipmentException
from shipment.utils.util import read_yaml_file
from shipment.logger import logging
import pandas as pd
import sys


class DataValidation:

    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info(f"{'>>' * 20}Data Validation log started.{'<<' * 20} ")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def get_raw_df(self):
        try:
            raw_df = pd.read_csv(self.data_ingestion_artifact.raw_file_path)
            return raw_df
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

            target_col_1 = config["target_column_1"]
            target_col_2 = config["target_column_2"]
            logging.info(f"Checking Target columns in {file_name}")
            if (target_col_1 and target_col_2) in read_file.columns:
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
            raw = self.data_ingestion_artifact.raw_file_path

            raw_col_check = self.check_columns_details(raw, config)
            raw_num_target = self.check_num_and_target_col(raw, config)

            result = raw_num_target and raw_col_check
            return result

        except Exception as e:
            raise ShipmentException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            self.validate_dataset_schema()
            schema_path = self.data_validation_config.schema_file_path
            data_validation_artifact = DataValidationArtifact(
                schema_file_path=schema_path,
                is_validated=True,
                message="Data Validation performed successfully."
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 20}Data Validation log completed.{'<<' * 20} \n\n")
