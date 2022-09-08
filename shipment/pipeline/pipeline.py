from shipment.config.configuration import Configuration
from shipment.exception import ShipmentException
from shipment.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from shipment.component.data_ingestion import DataIngestion
from shipment.component.data_validation import DataValidation
import sys


class Pipeline:

    def __init__(self, config: Configuration = Configuration()) -> None:
        try:
            self.config = config

        except Exception as e:
            raise ShipmentException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact
                                             )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def run_pipeline(self):
        try:
            # data ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)

        except Exception as e:
            raise ShipmentException(e, sys) from e
        