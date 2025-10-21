import os
import joblib
import pandas as pd
import numpy as np
from azureml.inference.schema.schema_decorators import input_schema, output_schema
from azureml.inference.schema.data_types import PandasParameterType

# Runs once when the server starts
def init():
    global model_pipeline
    # Make sure this filename matches the .pkl file you downloaded
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "flight_delay_pipeline_no_weather.pkl")
    model_pipeline = joblib.load(model_path)
    print("Pipeline loaded successfully.")

# Define sample input data structure
input_sample = pd.DataFrame({
    "Month": [1],
    "DayOfWeek": ["Wednesday"],
    "Hour": [14],
    "UniqueCarrier": ["WN"],
    "Origin": ["JFK"],
    "Dest": ["LAX"],
    "Distance": [2475.0]
})
# Define sample output
output_sample = [15.5] # Predicted delay in minutes

# Runs for every prediction request
@input_schema('data', PandasParameterType(input_sample))
@output_schema(PandasParameterType(output_sample))
def run(data):
    try:
        print(f"Received data: {data}")
        # Ensure input columns match expected types before prediction
        data["Distance"] = data["Distance"].astype(float)
        categorical_features = ["Month", "DayOfWeek", "Hour", "UniqueCarrier", "Origin", "Dest"]
        for col in categorical_features:
             if col in data.columns:
                 data[col] = data[col].astype(str)

        predictions = model_pipeline.predict(data)
        print(f"Predictions: {predictions}")

        # Clean up potential NaNs/Infs from prediction
        cleaned_predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

        return cleaned_predictions.tolist()
    except Exception as e:
        error = str(e)
        print(f"Error during prediction: {error}")
        return [f"Error: {error}"]