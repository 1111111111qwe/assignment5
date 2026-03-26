
import mlflow
import sys
import os

tracking_uri = "file://" + os.path.abspath("mlruns")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("assignment5")

print(f"Tracking URI: {tracking_uri}")

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Run ID: {run_id}")

client = mlflow.tracking.MlflowClient()

runs = client.search_runs(
    experiment_ids=["286619755676499285"],
    filter_string=f"run_id = '{run_id}'"
)

if not runs:
    print("Run not found in experiment!")
    sys.exit(1)

accuracy = runs[0].data.metrics["accuracy"]

print(f"Accuracy: {accuracy}")

if accuracy < 0.85:
    print("FAILED: accuracy is below 0.85 threshold!")
    sys.exit(1)
else:
    print("PASSED: accuracy meets the 0.85 threshold!")