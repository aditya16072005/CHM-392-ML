from ultralytics import YOLO

print("✅ Script started")

try:
    model = YOLO("yolov8n.pt")  # Load pre-trained model
    print("✅ YOLOv8 model loaded")

    model.train(data="data.yaml", epochs=50, imgsz=512)
    print("✅ Training finished")

except Exception as e:
    print("❌ An error occurred:", e)
