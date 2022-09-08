from collections import namedtuple

DataIngestionConfig = namedtuple("DataIngestionConfig",
                                 ["author_username", "raw_data_dir", "ingested_train_dir",
                                  "kaggel_dataset_name", "ingested_test_dir"])

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])

DataValidationConfig = namedtuple("DataValidationConfig",
                                  ["schema_file_path", "report_file_path", "report_page_file_path"])
