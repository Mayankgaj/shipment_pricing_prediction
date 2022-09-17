from shipment.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, \
    DataTransformationArtifact
from shipment.utils.util import load_data, read_yaml_file, save_object, save_numpy_array_data
from shipment.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from shipment.exception import ShipmentException
from shipment.logger import logging
from shipment.constant import *
import pandas as pd
import numpy as np
import sys, os


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

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:

            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = StandardScaler()

            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            schema_file_path = self.data_validation_artifact.schema_file_path

            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df_na = load_data(file_path=train_file_path, schema_file_path=schema_file_path)

            test_df_na = load_data(file_path=test_file_path, schema_file_path=schema_file_path)

            df = pd.concat(objs=[train_df_na, test_df_na], axis=0)

            schema = read_yaml_file(file_path=schema_file_path)

            target_column_name = schema[TARGET_COLUMNS_KEY]
            train_df, test_df = transform(X=df)

            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace(".csv", ".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv", ".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")

            save_numpy_array_data(file_path=transformed_train_file_path, array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path, array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path, obj=preprocessing_obj)

            data_transformation_artifact = \
                DataTransformationArtifact(is_transformed=True,
                                           message="Data transformation successfully.",
                                           transformed_train_file_path=transformed_train_file_path,
                                           transformed_test_file_path=transformed_test_file_path,
                                           preprocessed_object_file_path=preprocessing_obj_file_path
                                           )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def __del__(self):
        logging.info(f"{'=' * 20}Data Transformation log completed.{'=' * 20} \n\n")
