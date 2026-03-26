from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"
IMAGE_PATH = "data/imagem.png"

def main():
    model = YOLO(MODEL_NAME)
    results = model(IMAGE_PATH, save=True, project="outputs", name="predict")

    total_boxes = 0
    for result in results:
        total_boxes += len(result.boxes)

    print(f"Objetos detectados: {total_boxes}")

if __name__ == "__main__":
    main()