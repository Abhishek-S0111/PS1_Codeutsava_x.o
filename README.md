# 🏭 Steel Billet Inspection & Traceability System

**CodeUtsava X.0 — Team INIT**

AI-powered automated inspection system for steel manufacturing. Detects defects, reads billet IDs, measures dimensions, and maintains complete traceability records — 100% offline-capable.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📹 **Video/Image Processing** | Upload factory videos, process frame-by-frame with YOLO detection |
| 🔍 **Defect Detection** | Surface defect detection with confidence scoring and patch-based analysis |
| 📏 **Dimensional Measurement** | Calibrated pixel-to-physical conversion with tolerance checking |
| 🔤 **OCR Extraction** | Read laser-etched, stamped, painted markings; QR/barcode decoding |
| ⚖️ **Decision Engine** | PASS / FAIL / REVIEW / UNVERIFIED / REWORK with transparent rules |
| 📊 **Dashboard** | Real-time KPIs, trend charts, defect summary, alerts |
| 🤖 **AI Chatbot** | Query inspection history in natural language |
| 📥 **Excel/CSV Export** | One-click download with safe append-mode logging |
| 🔒 **100% Offline** | SQLite database, no internet required |

---

## 🚀 Quick Start (Windows)

```bat
cd steel-billet-inspection

REM Create virtual environment
python -m venv venv
venv\Scripts\activate

REM Install dependencies
pip install -r requirements.txt

REM Launch
streamlit run app.py
```

Or simply double-click **`run.bat`**.

Open your browser to **http://localhost:8501**.

---

## 🚀 Quick Start (Linux / macOS)

```bash
cd steel-billet-inspection
chmod +x run.sh
./run.sh
```

---

## 📋 System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.8+ | 3.10+ |
| RAM | 4 GB | 8 GB |
| CPU | Dual-core | Quad-core |
| Storage | 2 GB | 5 GB |
| GPU | Not required | NVIDIA + CUDA |

---

## 🎮 Demo Mode

No factory footage? Use the built-in demo:

1. Go to **Upload & Process**
2. Click **Demo Mode** tab
3. Click **Run Demo Inspection**

This generates a synthetic billet frame and runs the full pipeline.

To generate a demo video:
```bash
python training/generate_synthetic_data.py
```

---

## 📐 Calibration

Physical measurements require camera calibration:

1. Go to **Calibration** page
2. Choose a method:
   - **Reference Object** — upload image with known dimension, click two points
   - **Checkerboard** — upload 5+ checkerboard images
   - **Manual** — enter known pixels-per-mm value
3. Calibration is valid for 30 days

⚠️ Without calibration, dimensions are shown in pixels only and dimensional PASS cannot be issued.

---

## 📂 Project Structure

```
steel-billet-inspection/
├── app.py                          # Streamlit entry point
├── pages/                          # UI pages
│   ├── dashboard.py                # KPI dashboard
│   ├── upload_process.py           # Video/image upload & processing
│   ├── calibration.py              # Camera calibration & OCR test
│   ├── history_analysis.py         # Search, filter, export, trends
│   ├── chatbot_page.py             # AI chatbot interface
│   └── settings.py                 # Profiles, models, system info
├── vision/                         # Computer vision
│   └── detection.py                # Billet detection & tracking
├── defects/                        # Defect analysis
│   └── defect_detection.py         # YOLO + fallback defect detector
├── measurement/                    # Dimensional measurement
│   └── calibration.py              # Calibration & tolerance checking
├── ocr_module/                     # OCR & barcode
│   └── ocr_extraction.py           # EasyOCR/Tesseract, QR, barcode
├── rules/                          # Quality decisions
│   └── decision_engine.py          # PASS/FAIL/REVIEW rules
├── storage/                        # Data persistence
│   └── database.py                 # SQLite + Excel/CSV export
├── chatbot/                        # Chatbot
│   └── chatbot_engine.py           # Rule-based query engine
├── training/                       # Model training
│   ├── dataset_converter.py        # NEU-DET & Severstal → YOLO
│   ├── train_defect_model.py       # YOLOv8 training script
│   ├── evaluate_model.py           # mAP evaluation
│   └── generate_synthetic_data.py  # Demo data generator
├── tests/                          # Automated tests
│   └── test_core.py                # pytest suite
├── utils/                          # Shared utilities
│   ├── config.py                   # Configuration
│   └── logger.py                   # Logging
├── data/                           # Runtime data
├── models/                         # Trained models (.pt)
├── logs/                           # Application logs
├── requirements.txt
├── config.example.yaml
├── .env.example
├── run.bat                         # Windows launcher
├── run.sh                          # Linux/macOS launcher
└── README.md
```

---

## 🧠 Training a Custom Model

### Step 1: Prepare Dataset
```bash
python training/dataset_converter.py \
    --neu-dir /path/to/NEU-DET \
    --severstal-csv /path/to/train.csv \
    --severstal-images /path/to/train_images \
    --output-dir data/training_dataset
```

### Step 2: Train
```bash
python training/train_defect_model.py \
    --data-dir data/training_dataset \
    --epochs 50 --batch-size 16
```

### Step 3: Deploy
Copy `models/defect_detector.pt` and restart the app.

---

## 🤖 Chatbot Commands

| Query | What it does |
|---|---|
| *How many billets failed today?* | Count by status + date |
| *Show billets with defects* | List defective records |
| *Which heat numbers have repeated defects?* | Aggregate by heat # |
| *Show billets with unreadable IDs* | Find OCR failures |
| *Flag batch B-123 for review because cracks* | Create review flag |
| *Show snapshot for record ABC* | Full inspection detail |
| *What is the defect rate?* | Compute defect % |
| *Show recent alerts* | Unacknowledged alerts |

---

## 🔧 Troubleshooting

| Issue | Solution |
|---|---|
| `EasyOCR not available` | `pip install easyocr` |
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` |
| Slow processing | Increase frame skip, reduce resolution |
| No physical measurements | Calibrate the camera first |
| Excel locked | Records are safe in SQLite; export retries as CSV |

---

## 📜 License & Attribution

**CodeUtsava X.0 — Team INIT**

Uses: OpenCV (Apache 2.0), PyTorch (BSD), Ultralytics YOLO (AGPL-3.0), EasyOCR (Apache 2.0), Streamlit (Apache 2.0)

---

**Version 1.0.0** · ✅ Production Ready · 100% Offline
