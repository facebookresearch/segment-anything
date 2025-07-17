import os
import boto3
import subprocess

S3_PREFIX = "orders"
DEST_DIR = os.getcwd()

def parse_s3_path(s3_path):
    if not s3_path.startswith("s3://"):
        raise ValueError("Invalid S3 path")
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    return bucket, key

def download_file(s3_client, bucket, key, dest_path):
    print(f"⬇️ Downloading s3://{bucket}/{key} to {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, 'wb') as f:
        s3_client.download_fileobj(bucket, key, f)

def download_order_files(order_id):
    s3 = boto3.client("s3")
    bucket = os.environ.get("DATA_BUCKET")
    if not bucket:
        raise ValueError("Missing DATA_BUCKET environment variable")

    # File keys
    image_key = f"{S3_PREFIX}/{order_id}/user_image.jpg"
    label_key = f"{S3_PREFIX}/{order_id}/processed/labels/user_image.txt"

    # Destination paths
    local_image = os.path.join(DEST_DIR, "user_image.jpg")
    local_label = os.path.join(DEST_DIR, "user_image.txt")

    # Download both
    download_file(s3, bucket, image_key, local_image)
    download_file(s3, bucket, label_key, local_label)

def main():
    order_id = os.environ.get("ORDER_ID")
    if not order_id:
        raise ValueError("Missing ORDER_ID environment variable")

    print(f"📦 Starting download for order: {order_id}")
    download_order_files(order_id)
    print("✅ Downloads complete. Running measurement script...")

    # Run the analysis script
    command = [
        "python3", "find_ferrule_measurements.py",
        "--image", "user_image.jpg",
        "--yolo", "user_image.txt",
        "--write"
    ]
    try:
        subprocess.run(command, check=True)
        print("✅ Measurement script completed")
    except subprocess.CalledProcessError as e:
        print("❌ Measurement script failed")
        raise e

    # Check for expected outputs
    outputs = ["processed.jpg", "masked.jpg", "measurements.json"]
    missing = [f for f in outputs if not os.path.exists(f)]

    if missing:
        raise FileNotFoundError(f"Expected output file(s) not found: {', '.join(missing)}")

    print("📤 Uploading processed results to S3...")
    s3 = boto3.client("s3")
    bucket = os.environ.get("DATA_BUCKET")
    for filename in outputs:
        s3_key = f"{S3_PREFIX}/{order_id}/processed/{filename}"
        print(f"⬆️ Uploading {filename} to s3://{bucket}/{s3_key}")
        with open(filename, 'rb') as f:
            s3.upload_fileobj(f, bucket, s3_key)

    print("✅ All done.")

if __name__ == "__main__":
    main()
