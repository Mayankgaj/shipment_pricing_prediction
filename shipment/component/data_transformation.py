from shipment.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, \
    DataTransformationArtifact
from shipment.utils.util import load_data, read_yaml_file, save_object, save_numpy_array_data
from shipment.entity.config_entity import DataTransformationConfig
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from shipment.exception import ShipmentException
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from shipment.logger import logging
from shipment.constant import *
import pandas as pd
import numpy as np
import sys, os


class FeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        try:
            self.columns = columns
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def fit(self, X):
        return self

    def transform(self, X):
        try:
            if self.columns is None:
                df: pd.DataFrame = X.copy()
            else:
                df = pd.DataFrame(X, columns=self.columns)

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

            object_type = ['Country', 'Managed By', 'Vendor INCO Term',
                           'Shipment Mode', 'Product Group', 'Sub Classification', 'Vendor',
                           'Item Description', 'Molecule/Test Type', 'Brand', 'Dosage',
                           'Dosage Form', 'Manufacturing Site','Project Code']

            if self.columns is not None:
                one_hot_encoder = OneHotEncoder()
                df_cat = one_hot_encoder.fit_transform(df[object_type])
                transform_cat = df_cat.toarray()
                return transform_cat
            else:
                return df
        except Exception as e:
            raise ShipmentException(e, sys) from e


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

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path

            dataset_schema = read_yaml_file(file_path=schema_file_path)

            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]

            columns_remove = ['ID', 'PQ First Sent to Client Date', 'PO Sent to Vendor Date', 'Weight (Kilograms)',
                              'Freight Cost (USD)', 'PQ #', 'PO / SO #', 'ASN/DN #']

            for i in columns_remove:
                if i in numerical_columns:
                    numerical_columns.remove(i)
                elif i in categorical_columns:
                    categorical_columns.remove(i)

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('feature_generator', FeatureGenerator()),
                ('scaler', StandardScaler())
            ]
            )

            cat_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy="most_frequent")),
                ('feature_generator', FeatureGenerator(columns=categorical_columns)),
                ('scaler', StandardScaler())
            ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessing = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns),
            ])
            return preprocessing

        except Exception as e:
            raise ShipmentException(e, sys) from e

    @staticmethod
    def remove_unique_values(train, test):
        try:
            logging.info(f"Started Removing values from Columns which where not present in train dataset")
            for i in train.select_dtypes('object').drop(
                    ['Scheduled Delivery Date', 'Delivered to Client Date', 'Delivery Recorded Date'], axis=1):
                list_test = test[i].unique()
                remove = []
                for j in list_test:
                    if j in train[i].unique():
                        pass
                    else:
                        remove.append(j)
                for k in remove:
                    index = test.index[test[i] == k].to_list()
                    test.drop(index=index, axis=0, inplace=True)
            logging.info(f"Finished Removing values from Columns which where not present in train dataset")
            return train, test
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:

            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            schema_file_path = self.data_validation_artifact.schema_file_path

            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df_na = load_data(file_path=train_file_path, schema_file_path=schema_file_path)

            test_df_na = load_data(file_path=test_file_path, schema_file_path=schema_file_path)

            train_df, test_df = self.remove_unique_values(train=train_df_na, test=test_df_na)

            schema = read_yaml_file(file_path=schema_file_path)

            target_column_name = schema[TARGET_COLUMNS_KEY]

            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

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
