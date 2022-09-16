from collections import namedtuple

DataIngestionConfig = namedtuple("DataIngestionConfig",
                                 ["author_username", "raw_data_dir", "ingested_train_dir",
                                  "kaggel_dataset_name", "ingested_test_dir"])

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])

DataValidationConfig = namedtuple("DataValidationConfig",
                                  ["schema_file_path"])

DataTransformationConfig = namedtuple("DataTransformationConfig", ["transformed_train_dir",
                                                                   "transformed_test_dir",
                                                                   "preprocessed_object_file_path"])

ModelTrainerConfig = namedtuple("ModelTrainerConfig", ["trained_model_file_path", "base_accuracy",
                                                       "model_config_file_path"])

ModelPusherConfig = namedtuple("ModelPusherConfig", ["export_dir_path"])
