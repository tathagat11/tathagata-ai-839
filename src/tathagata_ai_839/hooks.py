import os
import subprocess
import logging
import pytest
import mlflow
from kedro.framework.hooks import hook_impl

# class SparkHooks:
#     @hook_impl
#     def after_context_created(self, context) -> None:
#         """Initialises a SparkSession using the config
#         defined in project's conf folder.
#         """

#         # Load the spark configuration in spark.yaml using the config loader
#         parameters = context.config_loader["spark"]
#         spark_conf = SparkConf().setAll(parameters.items())

#         # Initialise the spark session
#         spark_session_conf = (
#             SparkSession.builder.appName(context.project_path.name)
#             .enableHiveSupport()
#             .config(conf=spark_conf)
#         )
#         _spark_session = spark_session_conf.getOrCreate()
#         _spark_session.sparkContext.setLogLevel("WARN")

logger = logging.getLogger(__name__)

class MLflowModelDeploymentHook:
    @hook_impl
    def after_pipeline_run(self):
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions("Model", stages=["None"])[0]
        artifact_uri = latest_version.source
        local_path = artifact_uri.replace("file://", "")
        absolute_model_path = os.path.abspath(local_path)

        subprocess.run(["docker", "stop", "mlflow-model-server"], check=False)
        subprocess.run(["docker", "rm", "mlflow-model-server"], check=False)

        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                "mlflow-model-server",
                "-p",
                "5002:5002",
                "-v",
                f"{absolute_model_path}:/models",
                "mlflow-server",
            ],
            check=True,
        )

        print(f"Deployed latest model version: {latest_version.version}")
        # pass


class GenerateCardsHook:
    @hook_impl
    def after_pipeline_run(self):
        try:
            base_dir = os.getcwd()
            os.chdir(os.path.join(base_dir, "docs-quarto"))
            subprocess.run(["quarto", "render"], check=True)
            os.chdir(base_dir)
            print("Cards updated in quarto documentation.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred during post-pipeline actions: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

class TestingHook:
    @hook_impl
    def before_pipeline_run(self):
        """Hook implementation to run tests after the pipeline completes."""
        logger.info("Running tests after pipeline completion...")
        
        try:
            # Run only the data transformation tests that don't create artifacts
            test_files = [
                "tests/pipelines/data_processing/test_nodes.py",
                "tests/pipelines/data_processing/test_pipeline.py",
                "tests/pipelines/data_science/test_nodes.py",
                "tests/pipelines/data_science/test_pipeline.py"
            ]
            
            result = pytest.main(["-v"] + test_files)
            
            if result == 0:
                logger.info("All tests passed successfully!")
            else:
                logger.error("Post-pipeline tests failed")
                
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")