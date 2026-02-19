import kfp
import time
import sys
import os

KFP_ENDPOINT = os.environ["KFP_ENDPOINT"]

client = kfp.Client(host=KFP_ENDPOINT)

run = client.create_run_from_pipeline_package(
    pipeline_file="pipeline.yaml",
    arguments={}
)

print("Run submitted:", run.run_id)

# Wait for completion
client.wait_for_run_completion(run.run_id, timeout=3600)

run_detail = client.get_run(run.run_id)

status = run_detail.run.state
print("Final status:", status)

if status != "Succeeded":
    print("Pipeline failed!")
    sys.exit(1)

print("Pipeline succeeded!")
