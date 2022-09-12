from shipment.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, \
    DataTransformationArtifact
from shipment.utils.util import load_data, read_yaml_file
from sklearn.model_selection import train_test_split
from shipment.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import StandardScaler
from shipment.exception import ShipmentException
from shipment.logger import logging
from shipment.constant import *
import pandas as pd
import sys, os


class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        try:
            logging.info(f"{'=' * 20}Data Transformation log started.{'=' * 20} ")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise ShipmentException(e, sys) from e

    @staticmethod
    def preprocess_inputs(df, target):
        try:
            df = df.copy()

            # Drop ID column
            df = df.drop('ID', axis=1)

            # Drop missing target rows
            missing_target_rows = df[df['Shipment Mode'].isna()].index
            df = df.drop(missing_target_rows, axis=0).reset_index(drop=True)

            # Fill missing values
            df['Dosage'] = df['Dosage'].fillna(df['Dosage'].mode()[0])
            df['Line Item Insurance (USD)'] = df['Line Item Insurance (USD)'].fillna(
                df['Line Item Insurance (USD)'].mean())
            df['Shipment Mode'] = df['Shipment Mode'].fillna(df['Shipment Mode'].mode()[0])

            # Drop date columns with too many missing values
            df = df.drop(['PQ First Sent to Client Date', 'PO Sent to Vendor Date'], axis=1)

            # Extract date features
            for column in ['Scheduled Delivery Date', 'Delivered to Client Date', 'Delivery Recorded Date']:
                df[column] = pd.to_datetime(df[column])
                df[column + ' Year'] = df[column].apply(lambda x: x.year)
                df[column + ' Month'] = df[column].apply(lambda x: x.month)
                df[column + ' Day'] = df[column].apply(lambda x: x.day)
                df = df.drop(column, axis=1)

            # Drop numeric columns with too many missing values
            df = df.drop(['Weight (Kilograms)', 'Freight Cost (USD)'], axis=1)

            # Drop high-cardinality columns
            df = df.drop(['PQ #', 'PO / SO #', 'ASN/DN #'], axis=1)

            # Binary encoding
            df['Fulfill Via'] = df['Fulfill Via'].replace({'Direct Drop': 0, 'From RDC': 1})
            df['First Line Designation'] = df['First Line Designation'].replace({'No': 0, 'Yes': 1})

            # One-hot encoding
            for column in df.select_dtypes('object'):
                dummies = pd.get_dummies(df[column], prefix=column)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(column, axis=1)

            # Split df into X and y
            y = df[target]
            X = df.drop(target, axis=1)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)

            # Scale X
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:

            logging.info(f"Obtaining raw file path.")
            raw_file_path = self.data_ingestion_artifact.raw_file_path

            schema_file_path = self.data_validation_artifact.schema_file_path

            logging.info(f"Loading raw data as pandas dataframe.")
            raw_df = load_data(file_path=raw_file_path, schema_file_path=schema_file_path)

            schema = read_yaml_file(file_path=schema_file_path)

            target_column_1 = schema[TARGET_COLUMNS_KEY_1]
            target_column_2 = schema[TARGET_COLUMNS_KEY_2]

            logging.info(f"Applying preprocessing object on raw dataframe")
            X_train_t1, X_test_t1, y_train_t1, y_test_t1 = self.preprocess_inputs(raw_df, target_column_1)
            X_train_t2, X_test_t2, y_train_t2, y_test_t2 = self.preprocess_inputs(raw_df, target_column_2)

            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            dir_path = os.path.dirname(transformed_train_dir + r"\train")
            os.makedirs(dir_path, exist_ok=True)
            dir_path = os.path.dirname(transformed_test_dir + r"\test")
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Transformed file saved in Transformed folder")
            X_train_t1.to_csv(transformed_train_dir + r"\X_train_t1.csv")
            y_train_t1.to_csv(transformed_train_dir + r"\y_train_t1.csv")
            X_train_t2.to_csv(transformed_train_dir + r"\X_train_t2.csv")
            y_train_t2.to_csv(transformed_train_dir + r"\y_train_t2.csv")

            X_test_t1.to_csv(transformed_test_dir + r"\X_test_t1.csv")
            y_test_t1.to_csv(transformed_test_dir + r"\y_test_t1.csv")
            X_test_t2.to_csv(transformed_test_dir + r"\X_test_t2.csv")
            y_test_t2.to_csv(transformed_test_dir + r"\y_test_t2.csv")

            logging.info(f"Transformed files Saved in Transformed Folder Successfully")

            data_transformation_artifact = \
                DataTransformationArtifact(is_transformed=True,
                                           message="Data transformation successfully.",
                                           transformed_train_dir=transformed_train_dir,
                                           transformed_test_dir=transformed_test_dir
                                           )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def __del__(self):
        logging.info(f"{'=' * 20}Data Transformation log completed.{'=' * 20} \n\n")
