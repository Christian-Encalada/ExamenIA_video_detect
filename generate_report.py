from utils.report import PDFReport
import json
import os

def generate_report():
    print("Generando reporte de detecciones...")
    
    # Verificar si existe el archivo de detecciones
    if not os.path.exists('data/detections.json'):
        print("Error: No se encontraron datos de detecciones.")
        print("Primero debes ejecutar la detección de vehículos.")
        return
    
    try:
        # Cargar datos de las detecciones
        with open('data/detections.json', 'r') as f:
            detections = json.load(f)
        
        # Generar el reporte
        pdf = PDFReport('data/output/vehicle_detection_report.pdf')
        pdf.generate(detections['y_true'], detections['y_pred'])
        
        print("\n✓ Reporte generado exitosamente!")
        print(f"  Ubicación: data/output/vehicle_detection_report.pdf")
        
    except Exception as e:
        print(f"\n❌ Error al generar el reporte: {str(e)}")

if __name__ == "__main__":
    generate_report() 