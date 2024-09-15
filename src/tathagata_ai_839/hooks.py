from kedro.framework.hooks import hook_impl
from pyspark import SparkConf
from pyspark.sql import SparkSession
import subprocess
import mlflow
import os


class SparkHooks:
    @hook_impl
    def after_context_created(self, context) -> None:
        """Initialises a SparkSession using the config
        defined in project's conf folder.
        """

        # Load the spark configuration in spark.yaml using the config loader
        parameters = context.config_loader["spark"]
        spark_conf = SparkConf().setAll(parameters.items())

        # Initialise the spark session
        spark_session_conf = (
            SparkSession.builder.appName(context.project_path.name)
            .enableHiveSupport()
            .config(conf=spark_conf)
        )
        _spark_session = spark_session_conf.getOrCreate()
        _spark_session.sparkContext.setLogLevel("WARN")

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
        
        subprocess.run([
            "docker", "run", 
            "-d",
            "--name", "mlflow-model-server",
            "-p", "5002:5002", 
            "-v", f"{absolute_model_path}:/models",
            "mlflow-server"
        ], check=True)
        
        print(f"Deployed latest model version: {latest_version.version}")