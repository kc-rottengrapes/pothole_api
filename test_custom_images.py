from ultralytics import YOLO
import glob

# Load your model
model = YOLO('best.pt')

# Test on all images in a folder
print("Available image folders:")
print("1. train/images/ - Training images")
print("2. valid/images/ - Validation images")
print("3. Custom folder")

choice = input("Enter folder path or press Enter for train/images/: ").strip()
if not choice:
    choice = "train/images/"

# Get all images
image_files = glob.glob(f"{choice}/*.jpg") + glob.glob(f"{choice}/*.png")
print(f"Found {len(image_files)} images")

# Test with adjustable confidence
conf = float(input("Enter confidence threshold (0.1-0.9, default 0.3): ") or 0.3)

print(f"\nTesting {len(image_files)} images with confidence {conf}...")

detected_count = 0
for i, img_path in enumerate(image_files[:10]):  # Test first 10 images
    results = model(img_path, conf=conf, save=True)
    
    if len(results[0].boxes) > 0:
        detected_count += 1
        print(f"✓ {img_path}: {len(results[0].boxes)} potholes")
    else:
        print(f"✗ {img_path}: No potholes")

print(f"\nSummary: {detected_count}/{min(10, len(image_files))} images had detections")
print("Results saved to runs/detect/predict/")