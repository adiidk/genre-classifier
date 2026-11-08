"""
Music Genre Classification using Machine Learning - Flask app.

    python app.py     ->  http://127.0.0.1:5000
"""
import os

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

import config
from services import audio_service, feature_service
from services.errors import AppError
from services.model_service import ModelRegistry


def error_response(message, code, status):
    return jsonify({"error": {"code": code, "message": message}}), status


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    audio_service.cleanup_uploads()

    registry = ModelRegistry(config.MODELS_DIR)   # loads nothing heavy; no training here
    feature_service.warm_up()

    # ---------------------------------------------------------------- pages
    @app.get("/")
    def index():
        return render_template("index.html", max_mb=config.MAX_UPLOAD_MB)

    # ------------------------------------------------------------------ API
    @app.get("/api/models")
    def api_models():
        registry.refresh()
        return jsonify({"models": registry.list_models(), "issues": registry.issues})

    @app.post("/api/inspect")
    def api_inspect():
        upload = request.files.get("file")
        with audio_service.temporary_upload(upload) as path:
            info = audio_service.inspect_audio(path)
            info["filename"] = upload.filename
            info["size_bytes"] = path.stat().st_size
        return jsonify(info)

    @app.post("/api/extract")
    def api_extract():
        upload = request.files.get("file")
        with audio_service.temporary_upload(upload) as path:
            result = feature_service.extract(path)
        result["filename"] = upload.filename
        return jsonify(result)

    @app.post("/api/predict")
    def api_predict():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise AppError("The request must be JSON.", "bad_request", 400)
        model_key = data.get("model")
        if not isinstance(model_key, str) or not model_key:
            raise AppError("Please choose a model.", "no_model", 400)
        frame = feature_service.segments_to_frame(data.get("segments"))
        return jsonify(registry.predict(model_key, frame))

    @app.post("/api/predict-file")
    def api_predict_file():
        """One-shot endpoint (handy with curl): file + model -> prediction."""
        upload = request.files.get("file")
        model_key = request.form.get("model", "")
        if not model_key:
            raise AppError("Please choose a model.", "no_model", 400)
        with audio_service.temporary_upload(upload) as path:
            result = feature_service.extract(path)
        frame = feature_service.segments_to_frame(result["segments"])
        return jsonify(registry.predict(model_key, frame))

    # --------------------------------------------------------------- errors
    @app.errorhandler(AppError)
    def handle_app_error(err):
        return error_response(err.message, err.code, err.status)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_too_large(err):
        return error_response(
            f"The file is too large. The maximum size is {config.MAX_UPLOAD_MB} MB.",
            "file_too_large", 413,
        )

    @app.errorhandler(HTTPException)
    def handle_http(err):
        if request.path.startswith("/api/"):
            return error_response(err.description or err.name, "http_error", err.code or 500)
        return err

    @app.errorhandler(Exception)
    def handle_unexpected(err):
        app.logger.exception("Unhandled error")
        if request.path.startswith("/api/"):
            return error_response("An unexpected server error occurred. Check the terminal for details.",
                                  "server_error", 500)
        return "Internal server error", 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=os.environ.get("FLASK_DEBUG") == "1",
        threaded=True,
    )