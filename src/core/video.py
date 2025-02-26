import cv2
import time
from src.utils.report import PDFReport
import json

class VideoService:
    def __init__(self, detection_service):
        self.detection_service = detection_service
        self.video_state = {"play": True}
        
    def generate_frames(self):
        #Genera frames procesados con detecciones de vehículos en un video.
        
        cap = cv2.VideoCapture("data/videos/input_video.mp4")
        frame_count = 0
        
        try:
            while True:
                if self.video_state["play"]:
                    ret, frame = cap.read()
                    if not ret:
                        # Video terminó
                        self.detection_service.set_video_finished()
                        break
                        
                    # Procesar frame con detecciones
                    frame = self.detection_service.detect_vehicles(frame)
                    frame_count += 1
                    
                    ret, buffer = cv2.imencode(".jpg", frame)
                    frame = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    
                else:
                    time.sleep(0.1)
                    
        finally:
            print(f"Frames procesados: {frame_count}")
            cap.release()
            
            # Al finalizar, guardar las detecciones
            if self.detection_service.detections_history:
                y_true, y_pred = self.detection_service.get_detection_metrics()
                with open('data/detections.json', 'w') as f:
                    json.dump({
                        'y_true': y_true,
                        'y_pred': y_pred
                    }, f)
    
    def generate_final_report(self):
        try:
            if not self.detection_service.detections_history:
                return "Error: No hay detecciones para generar el reporte"
                
            y_true, y_pred = self.detection_service.get_detection_metrics()
            
            if not y_true or not y_pred:
                return "Error: No hay suficientes datos para generar métricas"
                
            # Generar reporte con gráficas
            pdf = PDFReport("data/vehicle_detection_report.pdf")
            pdf.generate(y_true, y_pred)
            
            return "Reporte generado exitosamente"
            
        except Exception as e:
            print(f"Error al generar reporte: {e}")
            return f"Error al generar reporte: {str(e)}"