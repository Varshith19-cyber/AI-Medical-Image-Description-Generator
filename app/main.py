from flask import Flask, render_template, request
import os

from app.modules.router import detect_image_type
from app.modules.document_module import process_document
from app.modules.radiology_module import process_radiology
from app.modules.validator import validate_result

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("app", "static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    image_type = detect_image_type(filepath)

    if image_type == "document":
        result = process_document(filepath)
        result.setdefault("top_finding", "Not Applicable")
        result.setdefault("confidence", 0.0)
        result.setdefault("status", "Document Analysis")
        result.setdefault("top_findings", [])
        result.setdefault("prescription_details", [])

    elif image_type == "radiology":
        result = process_radiology(filepath)
        result.setdefault("prescription_details", [])

    else:
        result = {
            "description": "Unknown image type.",
            "top_finding": "Unknown",
            "confidence": 0.0,
            "status": "Unsupported",
            "top_findings": [],
            "prescription_details": []
        }

    result = validate_result(result)

    return render_template(
        "index.html",
        filename=file.filename,
        file_url=f"/static/uploads/{file.filename}",
        type=image_type,
        result=result["description"],
        warning=result["warning"],
        top_finding=result.get("top_finding", "Not Available"),
        confidence=result.get("confidence", 0.0),
        status=result.get("status", "Unknown"),
        top_findings=result.get("top_findings", []),
        prescription_details=result.get("prescription_details", [])
    )