from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
import sys

# --- 1. Connect to Azure ML Workspace ---
try:
    # !!! IMPORTANT: REPLACE WITH YOUR ACTUAL SUBSCRIPTION ID !!!
    subscription_id = "a56afc44-f09b-4883-84e5-aaf8f5334b79"
    resource_group = "Flight-Tracking" # Should match resource group you created
    workspace = "Flight-track-2111"      # Should match workspace you created

    if subscription_id == "<YOUR_SUBSCRIPTION_ID>":
        print("❌ Error: Please replace <YOUR_SUBSCRIPTION_ID> with your actual Azure Subscription ID in deploy.py")
        sys.exit(1)

    ml_client = MLClient(
        DefaultAzureCredential(), subscription_id, resource_group, workspace
    )
    print("✅ Connected to Azure ML Workspace.")
except Exception as e:
     print(f"❌ Error connecting to Azure ML Workspace: {e}")
     sys.exit(1)

# --- 2. Define Names ---
#endpoint_name = "flight-delay-predictor-v1"

# Option 2: Add a date/time stamp for more uniqueness
import datetime
endpoint_name = "flight-pred-" + datetime.datetime.now().strftime("%Y%m%d%H%M")


model_local_path = "./flight_delay_pipeline_no_weather.pkl" # Your downloaded file
model_name = "flight-delay-pipeline-no-weather"
deployment_name = "blue"
environment_name = "sklearn-flight-env"
code_location = "./" # Directory containing score.py

# --- 3. Register Model ---
print("Registering model...")
try:
    model = Model(
        path=model_local_path,
        name=model_name,
        description="Sklearn pipeline (OHE + HGBoost) for flight delay."
    )
    registered_model = ml_client.models.create_or_update(model)
    print(f"✅ Model '{registered_model.name}' version {registered_model.version} registered.")
except Exception as e:
    print(f"❌ Error registering model: {e}")
    sys.exit(1)

# --- 4. Create Environment ---
print("Creating environment...")
try:
    env = Environment(
        name=environment_name,
        description="Sklearn environment for flight pipeline",
        conda_file="./environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )
    created_env = ml_client.environments.create_or_update(env)
    print(f"✅ Environment '{created_env.name}' version {created_env.version} created.")
except Exception as e:
    print(f"❌ Error creating environment: {e}")
    sys.exit(1)

# --- 5. Create/Update Endpoint ---
print(f"Creating/Updating endpoint '{endpoint_name}'...")
try:
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        auth_mode="key",
        description="Real-time flight delay prediction endpoint."
    )
    endpoint_result = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"✅ Endpoint '{endpoint_result.name}' provisioning state: {endpoint_result.provisioning_state}.")
except Exception as e:
    print(f"❌ Error creating/updating endpoint: {e}")
    sys.exit(1)

# --- 6. Create Deployment ---
print(f"Creating deployment '{deployment_name}'...")
try:
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=registered_model,
        environment=created_env,
        code_configuration=CodeConfiguration(
            code=code_location,
            scoring_script="score.py",
        ),
        instance_type="Standard_F2s_v2", # Small VM
        instance_count=1,
    )
    deployment_result = ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"✅ Deployment '{deployment_result.name}' provisioning state: {deployment_result.provisioning_state}.")
except Exception as e:
    print(f"❌ Error creating deployment: {e}")
    print("   Check Azure ML Studio for endpoint/deployment logs.")
    sys.exit(1)

# --- 7. Allocate Traffic ---
print(f"Allocating 100% traffic to deployment '{deployment_name}'...")
try:
    # Retrieve the latest endpoint details before updating traffic
    endpoint = ml_client.online_endpoints.get(name=endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    traffic_result = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"✅ Traffic allocation updated for endpoint '{traffic_result.name}'.")
except Exception as e:
    print(f"❌ Error updating traffic: {e}")
    sys.exit(1)

print(f"\n🎉 Deployment complete! Endpoint '{endpoint_name}' should be ready soon.")
print(f"   Monitor its status and test it in the Azure ML Studio.")