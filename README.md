# 🌍 ESP32 Disaster Detection with TinyML

**Title:** Improved Accuracy in Edge-Based Disaster Detection: Revisiting Prior Models Using TinyML on ESP32  
**Authors:** Samik, Ankit Ghosh, and Dr. Jyoti Yadav  
**Affiliation:** Delhi Skill and Entrepreneurship University (DSEU), India  
**Contact:** samik.workofficial@gmail.com

---

## 📌 Overview

This repository contains all resources related to our research work on deploying a **real-time calamity detection system using TinyML on ESP32**. The model detects disasters such as floods, fires, building collapse, and traffic accidents from aerial images captured via UAVs and uses onboard GPS for geolocation. All processing is done on-device using a quantized MobileNetV2 model, ensuring edge-only operation without cloud dependency.

---

## 🚀 Key Features

- ✅ Lightweight **MobileNetV2** model optimized for **ESP32-S3**
- ✅ Input image support for **96×96×3** and **126×126×3**
- ✅ Real-time **on-device inference (~1.1 FPS)**
- ✅ Uses **AIDER dataset** (5 disaster classes + normal)
- ✅ **Quantization** with TensorFlow Lite for Microcontrollers
- ✅ **Custom focal loss** and **dynamic learning rate scheduling**
- ✅ Integrated **GPS geolocation** + **store-and-forward communication**
- ✅ Compatible with **UAVs and tower installations**

---

## 🧠 Model Architecture

- MobileNetV2 (α=0.25) as base
- Lightweight custom classifier head:
  - Flatten → Dense(256, ReLU) → Dropout(0.3) → Dense(5, Softmax)
- Loss: **Custom Focal Loss** (α=0.2, γ=1.5)
- Learning Rate: **Custom decaying scheduler**
- Exported as `.tflite` and converted to C array for deployment

---

## 🧪 Dataset

- **AIDER Dataset** – Aerial images for emergency response
- 5 disaster classes:
  - Fire, Flood, Collapsed Building, Traffic Accident, Normal
- Balanced subset used: 485 images/class (2,425 total)
- Full dataset support and augmentation pipeline included

---

## ⚙️ Hardware Setup

| Component       | Model             |
|----------------|------------------|
| Microcontroller | ESP32-S3-CAM     |
| Camera          | Built-in OV2640  |
| GPS Module      | NEO-6M           |
| Storage         | MicroSD via SD_MMC |
| Battery & Charger | Li-ion + TP4056 |
| Comms (optional) | LoRa or Wi-Fi   |

---

## 🛠️ Tools & Libraries

- **TensorFlow / TensorFlow Lite**
- **TFLite Micro Interpreter**
- **TinyGPS++**
- **Arduino IDE / PlatformIO**
- **Netron** (for model visualization)
- **Matplotlib**, **Seaborn**, **Scikit-learn** (for metrics)

---

## 📊 Results

| Metric               | 96×96×3      | 126×126×3     |
|----------------------|--------------|----------------|
| Accuracy             | 94%          | 98%            |
| Inference Time       | 850 ms/frame | 850 ms/frame   |
| Power Consumption    | 160mA @ 5V    | 160mA @ 5V      |
| FPS                  | ~1.1         | ~1.1           |

> All results are validated with on-device testing on ESP32-S3.

---

## 🔗 Directory Structure


This repo is for the resources that has been used in the research of "Improving Accurecy: ESP32 disasteer detection using TinyML"
dataset - https://www.kaggle.com/datasets/samik2005/aider-dataset-partitioned-256485full
