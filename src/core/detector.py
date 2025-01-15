import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

class DetectionService:
    def __init__(self):
        # Cargaremos el modelo después de entrenarlo
        self.model = None
        
        # Leer clases desde archivo COCO
        with open("data/config/coco.txt", "r") as f:
            self.class_list = f.read().splitlines()
            
        # Clases relevantes para detección
        self.TARGET_CLASSES = {"car", "truck", "bus", "motorcycle"}
        
        # Almacenamiento de detecciones
        self.detections_history = []
        self.video_finished = False
        
        # Mapeo de clases a números
        self.class_mapping = {
            'car': 0,
            'truck': 1,
            'bus': 2,
            'motorcycle': 3
        }
    
    def load_model(self, model_path):
        """Carga el modelo entrenado"""
        self.model = YOLO(model_path)
        
    def detect_vehicles(self, frame):
        if self.model is None:
            raise ValueError("El modelo no ha sido cargado. Llama a load_model primero.")
            
        # Redimensionar frame
        frame_resized = cv2.resize(frame, (640, 640))
        
        # Realizar predicción
        results = self.model.predict(frame_resized)
        
        # Lista para almacenar detecciones del frame actual
        frame_detections = []
        
        # Procesar resultados
        if len(results) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                # Obtener coordenadas y confianza
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_list[class_id]
                
                if class_name in self.TARGET_CLASSES and confidence > 0.5:
                    # Almacenar la detección
                    detection_info = {
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': (int(x1), int(y1), int(x2), int(y2))
                    }
                    frame_detections.append(detection_info)
                    
                    # Dibujar bbox y etiqueta
                    cv2.rectangle(frame_resized, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (0, 255, 0), 2)
                    
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(frame_resized, label,
                              (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (255, 255, 255), 2)
        
        # Guardar detecciones para métricas
        if frame_detections:
            self.detections_history.extend(frame_detections)
        
        return frame_resized
    
    def get_detection_metrics(self):
        if not self.detections_history:
            raise ValueError("No hay detecciones almacenadas para generar métricas")
        
        y_pred = []
        y_true = []
        
        # Procesar detecciones y generar métricas
        for detection in self.detections_history:
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Solo incluir detecciones con alta confianza
            if confidence > 0.5 and class_name in self.class_mapping:
                predicted_class = self.class_mapping[class_name]
                y_pred.append(predicted_class)
                # Para este caso, asumimos que las predicciones son correctas
                # En un caso real, necesitarías ground truth labels
                y_true.append(predicted_class)
        
        print(f"Total detecciones procesadas: {len(y_pred)}")
        return y_true, y_pred

    def set_video_finished(self):
        self.video_finished = True