from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact",
                                   ["train_file_path", "test_file_path", "is_ingested", "message"])

DataValidationArtifact = namedtuple("DataValidationArtifact",
                                    ["is_validated", "schema_file_path", "message"])

DataTransformationArtifact = namedtuple("DataTransformationArtifact",
                                        ["is_transformed", "message", "transformed_train_dir",
                                         "transformed_test_dir", "preprocessed_object_file_path"])

ModelTrainerArtifact = namedtuple("ModelTrainerArtifact", ["is_trained", "message", "trained_model_file_path_t1",
                                                           "train_rmse_1", "test_rmse_1", "train_accuracy_1",
                                                           "test_accuracy_1", "model_accuracy_1",
                                                           "trained_model_file_path_t2", "train_rmse_2",
                                                           "test_rmse_2", "train_accuracy_2", "test_accuracy_2",
                                                           "model_accuracy_2"])

ModelPusherArtifact = namedtuple("ModelPusherArtifact", ["is_model_pusher", "export_model_file_path_t1",
                                                         "export_model_file_path_t2"])
