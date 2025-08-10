import os
import boto3
import subprocess

DEST_DIR = os.getcwd()

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
        "python3", "find_ferrule_mask.py",
        "--image", "user_image.jpg",
        "--yolo", "user_image.txt",
        "--write",
    ]
    try:
        subprocess.run(command, check=True)
        print("✅ Ferrule mask script completed")
    except subprocess.CalledProcessError as e:
        print("❌ Ferrule mask script failed")
        raise e

    # Ensure mask.png exists
    mask_file = "mask.png"
    if not os.path.exists(mask_file):
        raise FileNotFoundError(f"Expected output not found: {mask_file}")

    # Upload mask.png to S3
    print("📤 Uploading mask to S3...")
    s3 = boto3.client("s3")
    s3_key = f"{base_path}/processed/{mask_file}"
    print(f"⬆️ Uploading {mask_file} to s3://{bucket}/{s3_key}")
    with open(mask_file, 'rb') as f:
        s3.upload_fileobj(f, bucket, s3_key)

    print("✅ All done.")

if __name__ == "__main__":
    main()
