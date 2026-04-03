# Time-Series Anomaly Detector with a Temporal Convolutional Network Autoencoder and Streamlit
---

##  Description

This project implements an end-to-end unsupervised anomaly detection system for multivariate time-series data. A Temporal Convolutional Network (TCN) Autoencoder is trained on normal data to learn its patterns. During inference, deviations from these learned patterns are detected using reconstruction error, enabling identification of anomalies in real-world datasets like NASA telemetry data.

---

## Features

* Unsupervised anomaly detection
* TCN Autoencoder for sequence modeling
* Reconstruction error-based detection
* Exponential Moving Average (EMA) smoothing
* Dynamic thresholding using percentile
* Interactive Streamlit dashboard
* Fully containerized using Docker
* JSON report generation
* Advanced thresholding using Peak-Over-Threshold (POT)

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Streamlit
* Docker

---

## Workflow

1. Data ingestion from NASA dataset
2. Data preprocessing (normalization + windowing)
3. Training TCN Autoencoder on normal data
4. Reconstruction of test data
5. Calculation of reconstruction error
6. Smoothing using EMA
7. Thresholding (percentile-based)
8. Detection of anomalies
9. Visualization using Streamlit dashboard

---

## Architecture

```
Raw Data → Preprocessing → TCN Autoencoder → Reconstruction
        → Error Calculation → EMA Smoothing → Thresholding
        → Detected Anomalies → Streamlit Dashboard
```

---

## Project Structure

```
tcn-anomaly-detector/
│
├── app/
│   └── main.py
│
├── scripts/
│   ├── preprocess_data.py
│   ├── train.py
│   └── evaluate.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── results/
│
├── docker-compose.yml
├── Dockerfile
├── submission.json
├── requirements.txt
└── README.md
```

---

## How to Run by Cloning

```bash
git clone https://github.com/ashifa-1/tcn-anomaly-detector.git
cd tcn-anomaly-detector
```

---

## How to Run Manually

```bash
pip install -r requirements.txt

python scripts/preprocess_data.py
python scripts/train.py
python scripts/evaluate.py

streamlit run app/main.py
```

---

## How to Run Using Docker

```bash
docker-compose up --build
```

Open in browser:
http://localhost:8501

---

## Results

* Training loss reduced from **1.06 → 0.0010**
* Percentile-based anomalies detected: **1875**
* POT-based anomalies detected: **153**
* Smooth anomaly score visualization using EMA
* Interactive threshold control via Streamlit dashboard
* Multi-channel training improved model generalization

---

## Configuration

Defined in `submission.json`:

* Dataset: NASA SMAP/MSL
* Window Size: 50
* TCN Layers: 2
* Kernel Size: 3
* Threshold: 95th percentile
* POT initial quantile: 0.98

---

## Resources Used

* NASA SMAP & MSL Dataset
* Telemanom Research Paper
* PyTorch Documentation
* Streamlit Documentation
* Scikit-learn Documentation

---
