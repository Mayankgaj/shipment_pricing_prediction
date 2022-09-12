from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact",
                                   ["raw_file_path", "is_ingested", "message"])

DataValidationArtifact = namedtuple("DataValidationArtifact",
                                    ["is_validated", "schema_file_path", "message"])

DataTransformationArtifact = namedtuple("DataTransformationArtifact",
                                        ["is_transformed", "message", "transformed_train_dir",
                                         "transformed_test_dir"])
