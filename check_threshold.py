import mlflow
import sys

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Run ID: {run_id}")

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)
accuracy = run.data.metrics["accuracy"]

print(f"Accuracy: {accuracy}")

if accuracy < 0.85:
    print("FAILED: accuracy is below 0.85 threshold!")
    sys.exit(1)
else:
    print("PASSED: accuracy meets the 0.85 threshold!")