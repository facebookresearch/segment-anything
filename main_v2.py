import os
import boto3
import subprocess
import sys

DEST_DIR = os.getcwd()

class StepFunctionReporter:
    def __init__(self, task_token=None):
        self.task_token = task_token or os.environ.get("TASK_TOKEN")
        self.client = boto3.client("stepfunctions") if self.task_token else None

    def send_success(self, output: dict = None):
        if self.client and self.task_token:
            print("✅ Sending task success to Step Functions...")
            self.client.send_task_success(
                taskToken=self.task_token,
                output=json.dumps(output or {"status": "done"})
            )

    def send_failure(self, error="TaskFailed", cause=None):
        if self.client and self.task_token:
            print("❌ Sending task failure to Step Functions...")
            self.client.send_task_failure(
                taskToken=self.task_token,
                error=error,
                cause=cause or "Unknown failure"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            self.send_success()
        else:
            cause = ''.join(traceback.format_exception(exc_type, exc_value, tb))
            self.send_failure(error=exc_type.__name__, cause=cause)
            return False  # Re-raise exception


def download_file(s3_client, bucket, key, dest_path):
    print(f"⬇️ Downloading s3://{bucket}/{key} to {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, 'wb') as f:
        s3_client.download_fileobj(bucket, key, f)

def download_order_files(bucket, s3_prefix, order_id):
    s3 = boto3.client("s3")
    base_path = f"{s3_prefix}/{order_id}"
    image_key = f"{base_path}/user_image.jpg"
    label_key = f"{base_path}/processed/labels/user_image.txt"

    local_image = os.path.join(DEST_DIR, "user_image.jpg")
    local_label = os.path.join(DEST_DIR, "user_image.txt")

    download_file(s3, bucket, image_key, local_image)
    download_file(s3, bucket, label_key, local_label)

def main():
    with StepFunctionReporter():
        # Load environment variables
        bucket = os.environ.get("DATA_BUCKET")
        order_id = os.environ.get("ORDER_ID")
        s3_prefix = os.environ.get("S3_PREFIX", "orders")

        if not bucket or not order_id:
            raise ValueError("Missing required environment variables: DATA_BUCKET and/or ORDER_ID")

        base_path = f"{s3_prefix}/{order_id}"

        print(f"📦 Starting download for order: {order_id}")
        download_order_files(bucket, s3_prefix, order_id)
        print("✅ Downloads complete. Running ferrule mask script...")

        # Run the mask script (writes mask.png when --write is provided)
        command = [
            "python3", "-u", "find_ferrule_measurements_v2.py",
            "--image", "user_image.jpg",
            "--yolo", "user_image.txt"
        ]
        try:
            subprocess.run(command, check=True, stdout=sys.stdout, stderr=sys.stderr)
            print("✅ Ferrule mask script completed")
        except subprocess.CalledProcessError as e:
            print("❌ Ferrule mask script failed")
            raise e

        # List of expected output files
        outputs = ["mask.png", "frustum.jpg", "processed.jpg", "measurements.json"]

        missing = [f for f in outputs if not os.path.exists(f)]

        if missing:
            raise FileNotFoundError(f"Expected output file(s) not found: {', '.join(missing)}")

        # Upload files to S3
        s3 = boto3.client("s3")
        for file_name in outputs:
            if not os.path.exists(file_name):
                raise FileNotFoundError(f"Expected output not found: {file_name}")

            s3_key = f"{base_path}/processed/v2/{file_name}"
            print(f"📤 Uploading {file_name} to s3://{bucket}/{s3_key}")
            with open(file_name, 'rb') as f:
                s3.upload_fileobj(f, bucket, s3_key)

        print("✅ All expected files uploaded. Job complete.")

if __name__ == "__main__":
    main()