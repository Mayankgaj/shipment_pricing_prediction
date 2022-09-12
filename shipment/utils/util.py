from shipment.exception import ShipmentException
from shipment.constant import *
import pandas as pd
import numpy as np
import yaml
import sys
import dill


def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ShipmentException(e, sys) from e


def load_data(file_path: str, schema_file_path: str) -> pd.DataFrame:
    try:
        dataset_schema = read_yaml_file(schema_file_path)

        schema = dataset_schema["columns"]

        dataframe = pd.read_csv(file_path)

        error_message = ""

        for column in dataframe.columns:
            if column in list(schema.keys()):
                dataframe[column].astype(schema[column])
            else:
                error_message = f"{error_message} \nColumn: [{column}] is not in the schema."
        if len(error_message) > 0:
            raise Exception(error_message)
        return dataframe

    except Exception as e:
        raise ShipmentException(e, sys) from e
