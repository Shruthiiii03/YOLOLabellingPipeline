from flask import Flask, request, send_file
from PIL import Image, ImageDraw
import os, json, zipfile, tempfile, traceback
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.v2_task import create_task_with_local_image_auto_resize

app = Flask(__name__)
TOKEN = "YOUR TOKEN"
client = Client(Config(TOKEN))

@app.route("/annotate", methods=["POST"])
def annotate():
    try:
        zip_file = request.files["images_zip"]
        prompt = json.loads(request.form["prompt"])
        min_conf = float(request.form.get("threshold", 0.4))

        zip_path = os.path.join(tempfile.gettempdir(), "dataset.zip")
        zip_file.save(zip_path)

        extract_dir = os.path.join(tempfile.gettempdir(), "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zipf:
            zipf.extractall(extract_dir)

        # Save each reference image
        ref_image_map = {}
        for key in request.files:
            if key.startswith("reference_image_"):
                img_file = request.files[key]
                ref_path = os.path.join(tempfile.gettempdir(), img_file.filename)
                img_file.save(ref_path)
                ref_image_map[img_file.filename] = ref_path

        # Create image paths
        for item in prompt.get("visual_images", []):
            fname = item.get("filename")
            if fname and fname in ref_image_map:
                item["image_path"] = ref_image_map[fname]

        output_dir = os.path.join(tempfile.gettempdir(), "output")
        os.makedirs(output_dir, exist_ok=True)

        for root, _, files in os.walk(extract_dir):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(root, fname)
                    try:
                        task = create_task_with_local_image_auto_resize(
                            api_path="/v2/task/trex/detection",
                            api_body_without_image={
                                "model": "T-Rex-2.0",
                                "targets": ["bbox"],
                                "bbox_threshold": 0.2,
                                "iou_threshold": 0.8,
                                "prompt": prompt
                            },
                            image_path=img_path,
                        )
                        client.run_task(task)
                        save_results_as_yolo(task.result, img_path, output_dir, min_conf)
                    except Exception as e:
                        print(f"Failed on {fname}: {e}")

        zip_out = os.path.join(tempfile.gettempdir(), "annotated_labels.zip")
        with zipfile.ZipFile(zip_out, "w") as zipf:
            for f in os.listdir(output_dir):
                zipf.write(os.path.join(output_dir, f), arcname=f)

        return send_file(zip_out, mimetype="application/zip")

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}, 500

# output format 
def save_results_as_yolo(result, img_path, out_dir, min_conf):
    img = Image.open(img_path)
    W, H = img.size
    draw = ImageDraw.Draw(img)
    name = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(out_dir, f"{name}.txt")

    with open(txt_path, "w") as f:
        for obj in result.get("objects", []):
            score = obj.get("score", 1.0)
            if score < min_conf:
                continue
            x1, y1, x2, y2 = obj["bbox"]
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            w, h = (x2 - x1) / W, (y2 - y1) / H
            cat = obj.get("category_id", 0)
            f.write(f"{cat} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {score:.4f}\n")
            draw.rectangle([x1, y1, x2, y2], outline="red")
    img.save(os.path.join(out_dir, f"{name}_vis.png"))

if __name__ == "__main__":
    print("Backend running...")
    app.run(debug=True, use_reloader=False)