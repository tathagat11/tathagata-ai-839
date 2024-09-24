from kedro.framework.hooks import hook_impl
# from pyspark import SparkConf
# from pyspark.sql import SparkSession
import subprocess
import mlflow
import os


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

class MLflowModelDeploymentHook:
    @hook_impl
    def after_pipeline_run(self, catalog):
        selected_model_name = catalog.load("selected_model_name")
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions("Model_" + selected_model_name, stages=["None"])[0]
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

        latest_version = client.get_latest_versions("Model_" + selected_model_name, stages=["None"])[0].version

        client.transition_model_version_stage(
            name="Model_"+selected_model_name,
            version=latest_version,
            stage="Production"
        )

        not_selected_model_name = "Model_B" if selected_model_name[-1] == "A" else "Model_A"
        not_selected_latest_version = client.get_latest_versions("Model_" + not_selected_model_name, stages=["None"])[0].version
        
        client.transition_model_version_stage(
            name="Model_"+selected_model_name,
            version=latest_version - 1,
            stage="None"
        )
        client.transition_model_version_stage(
            name="Model_"+not_selected_model_name,
            version=not_selected_latest_version,
            stage="None"
        )
    

        print(f"Transitioned {selected_model_name} version {latest_version} to Production stage")