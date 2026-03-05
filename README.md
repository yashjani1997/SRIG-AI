# 🚦 SmartTraffic AI — YOLOv8s + Bidirectional LSTM
**Real-time Indian Traffic Detection & Congestion Forecasting System**

---

## 🌐 Project Links

- 🔴 **Live App:** [SmartTraffic AI on Streamlit](https://srig-ai-paxofjsb6rfhj5rgb7hfof.streamlit.app/)
- 🌍 **Demo Page:** [GitHub Pages](https://yashjani1997.github.io/SRIG-AI/)
- 💻 **GitHub Repo:** [yashjani1997/SRIG-AI](https://github.com/yashjani1997/SRIG-AI)
- 🤗 **HuggingFace Models:** [yash2024/smarttraffic-ai](https://huggingface.co/yash2024/smarttraffic-ai)

---

## 📌 Problem Statement

India ke major cities (Mumbai, Delhi, Bengaluru, Hyderabad) mein traditional traffic monitoring systems mein yeh limitations hain:

- Indian vehicle types (auto rickshaw, tempo, etc.) properly identify nahi kar paate
- Real-time congestion prediction nahi hoti — sirf current count milta hai
- PCU-based analysis nahi hoti jo actual road load measure kare
- Emergency vehicle delays — ambulance, fire brigade ka rasta block hota hai

**SmartTraffic AI** in sab problems ka solution deta hai — ek AI-powered system jo real-time Indian traffic detect kare, PCU calculate kare, aur agle **10 minute ka congestion forecast** kare.

---

## 🏗️ Pipeline

```
Video → YOLOv8s Detection → PCU Calculation → BiLSTM Forecast → Streamlit Dashboard
```

| Step | Description |
|---|---|
| Video Input | Traffic camera ya uploaded video file |
| YOLOv8s Detection | Har frame mein 8 Indian vehicle types detect |
| PCU Calculation | IRC standard PCU weight apply |
| BiLSTM Forecast | 60 frames ka sequence → agle 10 min predict |
| Dashboard | Streamlit app mein live stats, alerts aur graphs |

---

## 🗂️ Dataset

### YOLOv8s — Indian Driving Dataset (IDD)
- **Total Images:** 41,962
- **Training Images:** 33,569
- **Validation Images:** 4,196 (~10%)
- **Source:** Kaggle — IDD Detection Dataset

### BiLSTM — Indian Traffic Videos
- **Total Videos:** 44 videos (v1.mp4 to v44.mp4)
- **Processing:** YOLOv8s se har 5th frame process, PCU time-series CSV generated
- **Output:** `traffic_timeseries.csv` — timestamp, vehicle counts, PCU score

---

## 🚗 Vehicle Classes & PCU Values

| Vehicle | PCU Value | mAP50 |
|---|---|---|
| 🛺 Auto Rickshaw | 1.2 | 72.8% |
| 🚲 Bicycle | 0.5 | 49.6% |
| 🚌 Bus | 3.0 | 71.0% |
| 🚗 Car | 1.0 | 61.2% |
| 🏍️ Motorbike | 0.5 | 63.8% |
| 🚛 Truck | 3.0 | 68.1% |
| 🚐 Van | 1.5 | 48.1% |
| 🚶 Person | 0.0 | 50.9% |

> PCU (Passenger Car Unit) — IRC (Indian Roads Congress) standard jo actual road load measure karta hai.

---

## 🧠 Model 1 — YOLOv8s (Vehicle Detection)

**Why YOLOv8s?**  
Speed aur accuracy ka best balance — small model fast inference deta hai with 21.5 MB lightweight size.

### Training Configuration

| Parameter | Value |
|---|---|
| Base Model | YOLOv8s (COCO pretrained) |
| Epochs | 50 (early stopping, patience=15) |
| Image Size | 640 x 640 px |
| Batch Size | 16 |
| Optimizer | AdamW |
| Learning Rate | 0.001 (cosine LR schedule) |
| Augmentation | Mosaic, Mixup, HorizontalFlip |
| Hardware | Kaggle T4 GPU |
| Training Time | ~8 hours |
| Model Size | 21.5 MB |
| **Overall mAP50** | **~62%** |

---

## 🧠 Model 2 — Bidirectional LSTM (Congestion Forecasting)

**Why BiLSTM?**  
Traffic patterns mein past aur future context dono important hote hain — Bidirectional architecture dono directions se information process karta hai.

### Model Architecture

| Layer | Details |
|---|---|
| Layer 1 | BiLSTM (128 units) + Dropout(0.3) + BatchNorm |
| Layer 2 | BiLSTM (64 units) + Dropout(0.2) + BatchNorm |
| Layer 3 | Dense (64 units, ReLU) |
| Layer 4 | Dense (32 units, ReLU) |
| Output | Dense (1 unit) — PCU prediction |
| Loss | Huber Loss (robust to outliers) |
| Optimizer | Adam (lr=0.001) |
| Sequence Length | 60 frames |
| Best Epoch | 23 (early stopping) |
| **MAE** | **25.68 PCU** |
| **RMSE** | **31.07 PCU** |
| Model Size | 3.8 MB |

---

## 📊 Results Summary

| Metric | Value |
|---|---|
| Overall mAP50 (YOLO) | ~62% |
| Best Class mAP50 | Auto Rickshaw — 72.8% |
| BiLSTM MAE | 25.68 PCU |
| BiLSTM RMSE | 31.07 PCU |
| YOLO Model Size | 21.5 MB |
| BiLSTM Model Size | 3.8 MB |
| Training Time | ~8 hours (Kaggle T4 GPU) |

---

## 💡 Key Design Decisions

**PCU-Based Analysis** — Simple vehicle counting ki jagah IRC standard PCU weights use kiye — actual road load zyada accurately measure hoti hai.

**Transfer Learning** — YOLOv8s ko COCO pretrained weights se fine-tune kiya — training time kam hua aur small dataset pe bhi good performance mili.

**Two-Model Pipeline** — YOLO aur LSTM alag alag train kiye — dono independently optimized hain aur independently improve ho sakte hain.

**Label Remapping** — IDD ke 12 original classes mein se sirf 8 relevant Indian traffic classes rakhe — model confusion kam hua aur accuracy improve hui.

**Deployment Strategy** — Models HuggingFace pe host kiye, app Streamlit Cloud pe — GitHub pe sirf code, app first run pe automatically models download karti hai.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| YOLOv8s (Ultralytics) | Real-time vehicle detection |
| TensorFlow / Keras | BiLSTM training & inference |
| OpenCV | Video frame processing |
| Streamlit | Web dashboard & UI |
| Pandas / NumPy | Data processing |
| Scikit-learn | MinMaxScaler |
| Plotly | Interactive charts |
| HuggingFace Hub | Model hosting & auto-download |
| Kaggle T4 GPU | Training hardware |

---

## ⚙️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/yashjani1997/SRIG-AI.git
cd SRIG-AI

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
# Models auto-download from HuggingFace on first run
```

---

## 🧠 Key Learnings

- End-to-end two-model AI pipeline design
- Transfer learning with YOLOv8s on custom Indian dataset
- PCU-based traffic density modeling (IRC standards)
- Bidirectional LSTM for time-series forecasting
- Model hosting on HuggingFace + Streamlit Cloud deployment
- Label remapping and dataset preprocessing for real-world data

---

## 👤 Author

**Yash Jani**  
Data Analyst & Machine Learning Enthusiast  
[GitHub: yashjani1997](https://github.com/yashjani1997) | [HuggingFace: yash2024](https://huggingface.co/yash2024/smarttraffic-ai)
