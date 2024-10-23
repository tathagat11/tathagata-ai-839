import mlflow

model_uri = "runs:/6e7595e82a184f03946a91bd65373b13/model"

loaded_model = mlflow.pyfunc.load_model(model_uri)

model_signature = loaded_model.metadata.signature

if model_signature:
    print("Model Signature:")
    print(f"Inputs: {model_signature.inputs}")
    print(f"Outputs: {model_signature.outputs}")
