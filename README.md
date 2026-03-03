# 🚦 SmartTraffic AI — YOLOv8s + Bidirectional LSTM

> Real-time Indian Traffic Detection & Congestion Forecasting System

[![Demo](https://img.shields.io/badge/🌐_Live_Demo-Click_Here-00ff88?style=for-the-badge)](https://srig-ai-paxofjsb6rfhj5rgb7hfof.streamlit.app/)

## 🌐 Live Demo
👉 **[Demo Page Open Karo](https://yashjani1997.github.io/SRIG-AI/)**

## 📌 Overview
SmartTraffic AI ek two-model AI pipeline hai:
- **YOLOv8s** — 8 Indian vehicle types real-time detect karta hai (41,962 IDD images pe trained)
- **Bidirectional LSTM** — 60 frames ka sequence lekar agle 10 min ka congestion predict karta hai

## 🏗️ Pipeline
Video → YOLOv8s Detection → PCU Calculation → BiLSTM Forecast → Streamlit Dashboard

## 🚗 Vehicle Classes
| Class | PCU |
|-------|-----|
| 🛺 Auto Rickshaw | 1.2 |
| 🚲 Bicycle | 0.5 |
| 🚌 Bus | 3.0 |
| 🚗 Car | 1.0 |
| 🏍️ Motorbike | 0.5 |
| 🚛 Truck | 3.0 |
| 🚐 Van | 1.5 |
| 🚶 Person | 0.0 |

## 📊 Models
- **YOLOv8s**: 21.5 MB | 33,569 train images | Kaggle T4 GPU
- **BiLSTM**: 3.8 MB | 128→64→Dense | Huber Loss | Dropout + BatchNorm

## 🛠️ Stack
YOLOv8s · TensorFlow · Streamlit · OpenCV · Pandas · Kaggle T4
