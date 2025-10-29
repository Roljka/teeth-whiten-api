import os
import io
import base64
import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# OpenAI klients (paļaujas uz OPENAI_API_KEY vidē)
client = OpenAI()

@app.get("/health")
def health():
    return jsonify(ok=True, model="gpt-image-1")

@app.post("/whiten")
def whiten():
    """
    Pieņem multipart/form-data ar lauku 'file'.
    Sūta uz OpenAI Images Edit ar precīzu promptu, lai balina TIKAI zobus.
    Atgriež tieši attēla baitus (PNG), ko frontend var parādīt.
    """
    try:
        if "file" not in request.files:
            return jsonify(error="File missing: send multipart/form-data with field 'file'"), 400

        up = request.files["file"]
        if up.filename == "":
            return jsonify(error="Empty filename"), 400

        # Saglabājam uz /tmp, jo OpenAI SDK .images.edit grib failu objektu
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_in:
            in_path = tmp_in.name
            up.stream.seek(0)
            tmp_in.write(up.read())

        prompt = (
            "Whiten only the TEETH in the photo. "
            "Do not change lips, gums, skin, tongue, hair or background. "
            "Keep natural texture and preserve shading; avoid overexposure or halo. "
            "No global brightness/contrast changes. Subtle, realistic teeth whitening only."
        )

        # Rediģējam bez maskas — ļaujam modelim pašam detektēt zobus
        with open(in_path, "rb") as f:
            resp = client.images.edit(
                model="gpt-image-1",
                image=f,
                prompt=prompt,
                size="1024x1024"  # drošs noklusējums; var likt "768x768" ātrumam/izmaksai
            )

        # Rezultāts bāzēts uz base64 PNG
        img_b64 = resp.data[0].b64_json
        img_bytes = base64.b64decode(img_b64)

        # Notīrām tmp
        try:
            os.remove(in_path)
        except Exception:
            pass

        return send_file(
            io.BytesIO(img_bytes),
            mimetype="image/png",
            as_attachment=False,
            download_name="whitened.png"
        )

    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
