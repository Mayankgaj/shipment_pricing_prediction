from collections import namedtuple

DataIngestionConfig = namedtuple("DataIngestionConfig",
                                 ["author_username", "raw_data_dir",
                                  "kaggel_dataset_name"])

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])

DataValidationConfig = namedtuple("DataValidationConfig",
                                  ["schema_file_path"])

DataTransformationConfig = namedtuple("DataTransformationConfig", ["transformed_train_dir",
                                                                   "transformed_test_dir",
                                                                   "preprocessed_object_file_path"])
