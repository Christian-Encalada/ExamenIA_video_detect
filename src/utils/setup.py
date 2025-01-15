from ultralytics import YOLO
import os
import shutil

def setup_model():
    """Configura el modelo YOLOv8 para detección de vehículos"""
    
    print("Iniciando configuración del modelo...")
    
    # 1. Crear directorios necesarios
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/videos', exist_ok=True)
    os.makedirs('data/output', exist_ok=True)
    
    try:
        # 2. Descargar y configurar el modelo
        print("Descargando modelo YOLOv8...")
        model = YOLO('yolov8s.pt')  # Descarga el modelo pequeño de YOLOv8
        
        # 3. Guardar el modelo
        model_path = 'data/models/yolov8s.pt'
        model.save(model_path)
        print(f"✓ Modelo guardado en: {model_path}")
        
        # 4. Verificar clases disponibles
        print("\nClases disponibles para detección:")
        print("- car (coche)")
        print("- truck (camión)")
        print("- bus (autobús)")
        print("- motorcycle (motocicleta)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error al configurar el modelo: {e}")
        return False

if __name__ == "__main__":
    # Ejecutar configuración
    print("=== Configuración del Modelo de Detección de Vehículos ===\n")
    success = setup_model()
    
    if success:
        print("\n✓ Configuración completada exitosamente!")
        print("\nPara ejecutar la aplicación:")
        print("1. Coloca tu video en 'data/videos/input_video.mp4'")
        print("2. Ejecuta 'python main.py'")
        print("3. Abre http://localhost:5000 en tu navegador")
    else:
        print("\n❌ La configuración falló. Revisa los errores anteriores.") 