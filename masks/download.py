#!/usr/bin/env python3
import boto3
import os

BUCKET_NAME = "dev-golfstripes-ferrule-data"   # <-- change this
OUTPUT_DIR = "./"           # where {key}.png will be saved
PREFIX = "orders/"

s3 = boto3.client("s3")

def main():
    paginator = s3.get_paginator("list_objects_v2")
    matches = []

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Expecting: orders/{key}/processed/mask.png
            if key.count("/") >= 3 and key.endswith("/processed/mask.png"):
                parts = key.split("/")
                order_key = parts[1]  # This is the {key} part
                filename = f"{order_key}.png"
                matches.append((key, filename))

    if not matches:
        print("No matches found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for s3_key, filename in matches:
        local_path = os.path.join(OUTPUT_DIR, filename)
        print(f"⬇️ Downloading s3://{BUCKET_NAME}/{s3_key} -> {local_path}")
        s3.download_file(BUCKET_NAME, s3_key, local_path)

if __name__ == "__main__":
    main()
