from shipment.logger import logging
from shipment.exception import ShipmentException
from shipment.entity.artifact_entity import ModelPusherArtifact
from shipment.entity.config_entity import ModelPusherConfig, ModelTrainerConfig
import os, sys
import shutil


class ModelPusher:

    def __init__(self, model_pusher_config: ModelPusherConfig,
                 model_trainer_config: ModelTrainerConfig
                 ):
        try:
            logging.info(f"{'>>' * 30}Model Pusher log started.{'<<' * 30} ")
            self.model_pusher_config = model_pusher_config
            self.model_trainer_config = model_trainer_config

        except Exception as e:
            raise ShipmentException(e, sys) from e

    def export_model(self) -> ModelPusherArtifact:
        try:
            model_file_path_t1 = self.model_trainer_config.trained_model_file_path.replace(".pkl", "_1.pkl")
            model_file_path_t2 = self.model_trainer_config.trained_model_file_path.replace(".pkl", "_2.pkl")
            export_dir = self.model_pusher_config.export_dir_path
            model_file_name_t1 = "model_1.pkl"
            model_file_name_t2 = "model_2.pkl"
            export_model_file_path_t1 = os.path.join(export_dir, model_file_name_t1)
            export_model_file_path_t2 = os.path.join(export_dir, model_file_name_t2)
            logging.info(f"Exporting model file: [{export_model_file_path_t1}]")
            os.makedirs(export_dir, exist_ok=True)
            logging.info(f"Exporting model file: [{export_model_file_path_t2}]")
            os.makedirs(export_dir, exist_ok=True)

            shutil.copy(src=model_file_path_t1, dst=export_model_file_path_t1)
            logging.info(
                f"Trained model: {model_file_path_t1} is copied in export dir:[{export_model_file_path_t1}]")
            shutil.copy(src=model_file_path_t2, dst=export_model_file_path_t2)
            logging.info(
                f"Trained model: {model_file_path_t2} is copied in export dir:[{export_model_file_path_t2}]")

            model_pusher_artifact = ModelPusherArtifact(is_model_pusher=True,
                                                        export_model_file_path_t1=export_model_file_path_t1,
                                                        export_model_file_path_t2=export_model_file_path_t2,
                                                        )
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            return model_pusher_artifact
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            return self.export_model()
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 20}Model Pusher log completed.{'<<' * 20} ")
