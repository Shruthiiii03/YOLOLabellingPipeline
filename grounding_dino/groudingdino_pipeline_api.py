import cv2
from PIL import Image
from io import BytesIO
from flask import Flask, request, send_file
from groundingdino.util.inference import load_model, load_image, predict, annotate

BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

app = Flask(__name__)

@app.route("/detect", methods=["POST"])
def detect():
    image_file = request.files["image"]
    prompt = request.form["prompt"]
    image_bytes = image_file.read()

    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "./weights/groundingdino_swint_ogc.pth")

    image_source, image = load_image(image_bytes)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=prompt,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device="cpu"
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    img_io = BytesIO()
    Image.fromarray(annotated_frame).save(img_io, format="PNG")
    img_io.seek(0)

    return send_file(img_io, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)