# YOLOLabellingPipeline

A lightweight pipeline to auto-generate YOLO-compatible annotations from reference images using **T-Rex** or optionally **Gemini** AI models. Combines a simple **Streamlit frontend** with a Flask-based backend and lets you generate bounding box labels with minimal effort.


## What This Project Does

- Lets user upload:
  - Reference image(s) of an object (e.g., bollards)
  - A zip of target images to annotate
  - And choose a confidence threshold

- Sends it to backend
- Runs detection via **T-Rex 2.0** (hosted on DDS Cloud)
- Outputs YOLO `.txt` annotations + visual overlays
- Zips them all for download

---

## Project Structure

```
YOLOLabellingPipeline/
│
├── st_app/                 # Streamlit frontend
│   ├── labellingpipeline.py
│   ├── input_data/         # Holds uploaded reference images
│   └── export/             # Outputs from backend (YOLO + visual)
│
├── T-Rex/                  # Flask backend to interface with DDS Cloud
│   ├── trex_server.py      # Main API used by frontend
│
├── gemini2/                # Placeholder for Gemini backend
│   └── gemini_pipeline_api.py
│
├── complete_dataset/       # Fully annotated dataset output
├── requirements.txt        # Shared dependencies
└── README.md               # This file
```

---

## Setup Instructions

### 1. Clone the repo



### 2. Clone T-Rex into this repo

This repo depends on the T-Rex model, which is not directly included.

Please clone it manually:

```bash
git clone https://github.com/IDEA-Research/T-Rex.git
```
### 3. Create a python virtual environment

```
python3 -m venv .venv
source .venv/bin/activate

# to test whether the environment has been activated 
which python # should point to `.venv/bin/python`
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> There are separate `requirements.txt` in root and some folders (e.g., `gemini2/`). Install those too if needed.

For example, 
```
cd T-Rex
pip install -r requirements.txt
```
---

## Running the App

### Backend (Flask)

```bash
python trex_server.py
```

Make sure to:
- Replace `TOKEN` in `trex_server.py` with your own DDS Cloud API token
- Check that port 5000 is available

### Frontend (Streamlit)

```bash
cd st_app
streamlit run labellingpipeline_st.py
```

Then open the UI in your browser (usually http://localhost:8501).

---

## API – How `/annotate` Works

The Flask backend exposes:

```
POST /annotate
```

### Expected Inputs:
| Field               | Type         | Description                              |
|--------------------|--------------|------------------------------------------|
| `images_zip`       | File         | .zip containing unlabeled images         |
| `reference_image_X`| File(s)      | Reference image(s) of target object(s)   |
| `prompt`           | JSON string  | Includes metadata + visual_images array |
| `threshold`        | float        | Detection threshold (optional)           |

### Returns:
- `.zip` file of:
  - YOLO `.txt` annotation files
  - Visual `.png` files with bounding boxes

---

## YOLO Output Format

Each `.txt` file is named after the image and contains:

```
<class_id> <x_center> <y_center> <width> <height> <confidence>
```

- Normalized to (0–1)
- Example:
  ```
  0 0.543 0.233 0.121 0.093 0.8876
  ```

---

## Known Limitations of T-Rex

| Limitation                  | Elaboration |
|-----------------------------|----------------|
| T-Rex is one-shot only      | Doesn’t generalize well across object variations |
| Must look visually similar  | Needs tight match between reference + target |
| No segmentation             | Bounding boxes only |

---

## Optional Improvements 

- Connect a database server where the data can be retrieved from 
- Add CLI version for batch jobs
- Plug into YOLO training loop

---

## Limitations of the codebase
- T-Rex token is IP-locked, meaning it only works on one device. For every new device, please create a new token.

## Summary

This was built as a fast way to prototype YOLO datasets using one-shot examples. Still has rough edges, but works well when reference images closely match the targets. 

---