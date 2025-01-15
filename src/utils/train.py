from ultralytics import YOLO

def train_model():
    # Inicializar modelo YOLOv8
    model = YOLO('yolov8s.pt')  # cargar modelo pre-entrenado
    
    # Entrenar el modelo
    # Necesitarás preparar un archivo data.yaml con la configuración de tu dataset
    results = model.train(
        data='path/to/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='vehicle_detection'
    )
    
    # Guardar el modelo entrenado
    model.save('app/views/static/models/best.pt')

if __name__ == "__main__":
    train_model() 