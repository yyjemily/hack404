import requests
import os

BASE_URL = "http://127.0.0.1:5000"
UPLOAD_URL = f"{BASE_URL}/upload"
ANALYZE_URL = f"{BASE_URL}/analyze"

TEST_IMAGE_PATH = "test_images/teeth1.jpg"  # Make sure this image exists relative to where you run this script

if not os.path.exists(TEST_IMAGE_PATH):
    print(f"Test image not found at {TEST_IMAGE_PATH}, please update the path.")
    exit(1)

# Upload image
print(f"Uploading image: {TEST_IMAGE_PATH}")
with open(TEST_IMAGE_PATH, "rb") as img_file:
    files = {"file": img_file}
    upload_resp = requests.post(UPLOAD_URL, files=files)

if upload_resp.status_code != 200:
    print("Upload failed:", upload_resp.text)
    exit(1)

upload_data = upload_resp.json()
print("Upload response:", upload_data)

filepath = upload_data.get("filepath")
if not filepath:
    print("No filepath returned after upload.")
    exit(1)

# Analyze image
print(f"Analyzing image at: {filepath}")
analyze_resp = requests.post(ANALYZE_URL, json={"filepath": filepath})

if analyze_resp.status_code != 200:
    print("Analyze failed:", analyze_resp.text)
    exit(1)

analyze_data = analyze_resp.json()
print("Analyze response:", analyze_data)
