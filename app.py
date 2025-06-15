from flask import Flask, request, render_template, send_file
from processing import process_image_and_plot
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            # Guardar imagen con un nombre único
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Procesar imagen y obtener el gráfico
            result_path, entropy_before, entropy_after = process_image_and_plot(filepath)

            return render_template(
                "index.html",
                ent_before=round(entropy_before, 4),
                ent_after=round(entropy_after, 4),
                plot_file=os.path.basename(result_path)
            )
    return render_template("index.html")

@app.route("/results/<filename>")
def result_image(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), mimetype="image/png")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Usa el puerto de Render o 5000 local
    app.run(host="0.0.0.0", port=port)
