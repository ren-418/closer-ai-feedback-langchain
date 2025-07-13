import os
import time
import json
from openai import OpenAI

# Set your OpenAI Project API Key here (or use environment variable)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "sk-..."  # <-- Replace with your key if not using env var

client = OpenAI(api_key=OPENAI_API_KEY)

# 1. Prepare a few simple test prompts
batch_tasks = []
for i in range(3):
    prompt = f"Say hello from test batch job, item {i+1}."
    batch_tasks.append({
        "custom_id": f"test-{i+1}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 50
        }
    })

# 2. Write to a .jsonl file
batch_file_path = "test_batch_tasks.jsonl"
with open(batch_file_path, "w", encoding="utf-8") as f:
    for task in batch_tasks:
        f.write(json.dumps(task) + "\n")

print(f"Batch file written: {batch_file_path}")

# 3. Upload the file
batch_file_obj = client.files.create(
    file=open(batch_file_path, "rb"),
    purpose="batch"
)
print(f"Batch file uploaded: {batch_file_obj.id}")

# 4. Create the batch job
batch_job = client.batches.create(
    input_file_id=batch_file_obj.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
print(f"Batch job created: {batch_job.id}")

# 5. Poll for completion
while True:
    job_status = client.batches.retrieve(batch_job.id)
    print(f"Batch job status: {job_status.status}")
    if job_status.status == "completed":
        break
    elif job_status.status in ("failed", "expired", "cancelled"):
        raise Exception(f"Batch job failed: {job_status.status}")
    time.sleep(10)

# 6. Retry for output_file_id
result_file_id = job_status.output_file_id
retries = 0
max_retries = 10
while not result_file_id and retries < max_retries:
    print("Waiting for output_file_id to become available...")
    time.sleep(10)
    job_status = client.batches.retrieve(batch_job.id)
    result_file_id = job_status.output_file_id
    retries += 1

if not result_file_id:
    raise Exception("Batch job completed but no output_file_id found. Cannot download results.")

print(f"Result file id: {result_file_id}")

# 7. Download and print results
result_content = client.files.content(result_file_id).content
result_file_path = "test_batch_results.jsonl"
with open(result_file_path, "wb") as f:
    f.write(result_content)
print(f"Results downloaded to: {result_file_path}")

# 8. Print results
with open(result_file_path, "r", encoding="utf-8") as f:
    for line in f:
        res = json.loads(line.strip())
        print(json.dumps(res, indent=2))

print("Batch API test complete.") 