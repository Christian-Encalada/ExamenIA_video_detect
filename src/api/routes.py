from flask import Blueprint, Response, jsonify, render_template, send_from_directory
from src.core.detector import DetectionService
from src.core.video import VideoService

routes = Blueprint('routes', __name__)

# Inicializar servicios
detection_service = DetectionService()
detection_service.load_model('data/models/yolov8s.pt')
video_service = VideoService(detection_service)

@routes.route('/')
def index():
    return render_template('index.html')

@routes.route('/video_feed')
def video_feed():
    return Response(
        video_service.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@routes.route('/play', methods=['POST'])
def play():
    video_service.video_state["play"] = True
    return jsonify({"status": "success", "playing": True})

@routes.route('/pause', methods=['POST'])
def pause():
    video_service.video_state["play"] = False
    return jsonify({"status": "success", "playing": False})

@routes.route('/replay', methods=['POST'])
def replay():
    # Reiniciar el servicio de video
    global video_service
    video_service = VideoService(detection_service)
    video_service.video_state["play"] = True
    return jsonify({"status": "success"})

@routes.route('/video_status')
def video_status():
    return jsonify({
        "finished": detection_service.video_finished,
        "playing": video_service.video_state["play"]
    })

@routes.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('data', filename)

@routes.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        message = video_service.generate_final_report()
        if message.startswith("Error"):
            return jsonify({"error": message}), 400
            
        return jsonify({
            "message": message,
            "pdf_url": "/static/vehicle_detection_report.pdf"
        })
    except Exception as e:
        print(f"Error generating report: {e}")
        return jsonify({"error": str(e)}), 500 