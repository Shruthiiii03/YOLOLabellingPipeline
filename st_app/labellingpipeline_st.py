import os
import zipfile
import json
import requests
import tempfile
from io import BytesIO
from PIL import Image, ImageDraw
from pathlib import Path
import streamlit as st
from streamlit_drawable_canvas import st_canvas

def st_app():
    st.set_page_config(page_title="YOLO Labelling Pipeline")
    st.header("Upload and Annotate Images")

    st.sidebar.subheader("Upload Image Dataset")
    zip_file = st.sidebar.file_uploader("Upload a .zip of images", type="zip")

    st.subheader("Reference Images for Annotation")
    ref_imgs = st.file_uploader("Upload multiple reference images (JPEG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="manual")

    canvas_results = []
    annotated_refs = []
    if ref_imgs:
        for idx, manual_img in enumerate(ref_imgs):
            st.markdown(f"#### Annotate Image {idx + 1}: {manual_img.name}")
            image = Image.open(manual_img)
            W, H = image.size
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=2,
                stroke_color="red",
                background_image=image,
                update_streamlit=True,
                height=H,
                width=W,
                drawing_mode="rect",
                key=f"canvas_{idx}"
            )
            canvas_results.append((manual_img, canvas_result))

    threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.01)
    model_choice = st.sidebar.selectbox("Choose Model", ["T-Rex", "Gemini", "DINO"], index=0)
    
    run = st.button("Run T-Rex Annotation")

    if run:
        if not ref_imgs:
            st.error("Upload at least one reference image.")
        elif not zip_file:
            st.error("Upload a .zip file with images.")
        elif any(not res or not res.json_data or not res.json_data["objects"] for _, res in canvas_results):
            st.error("Draw at least one bounding box on each reference image.")
        else:
            visual_images = []
            for manual_img, canvas_result in canvas_results:
                interactions = []
                for obj in canvas_result.json_data["objects"]:
                    x, y, w, h = obj["left"], obj["top"], obj["width"], obj["height"]
                    x2, y2 = x + w, y + h
                    interactions.append({"type": "rect", "category_id": 1, "rect": [int(x), int(y), int(x2), int(y2)]})

                visual_images.append({
                    "filename": manual_img.name,
                    "interactions": interactions
                })

            prompt = {"type": "visual_images", "visual_images": visual_images}

            files = {"images_zip": (zip_file.name, zip_file.getvalue(), "application/zip")}
            for i, (manual_img, _) in enumerate(canvas_results):
                files[f"reference_image_{i}"] = (manual_img.name, manual_img.getvalue(), manual_img.type)

            data = {
                "prompt": json.dumps(prompt),
                "threshold": str(threshold),
                "model": model_choice
            }

            st.info("Sending to T-Rex backend...")
            try:
                response = requests.post("http://localhost:5000/annotate", files=files, data=data)
                if response.status_code == 200:
                    st.success("Received annotated labels!")
                    st.download_button("Download ZIP", data=response.content, file_name="annotated_labels.zip", mime="application/zip")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

if __name__ == "__main__":
    st_app()