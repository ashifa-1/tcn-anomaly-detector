# TCN vs LSTM for Time-Series Anomaly Detection

## Introduction

Time-series anomaly detection requires models capable of capturing temporal dependencies. Two popular approaches are:

- Recurrent Neural Networks (LSTM)
- Temporal Convolutional Networks (TCN)

In this project, we chose **TCN Autoencoder** over LSTM due to its efficiency and stability.

---

## 1. Parallelization & Training Speed

### LSTM
- Processes data sequentially
- Cannot parallelize across time steps
- Slower training

### TCN
- Uses convolution → processes entire sequence at once
- Fully parallelizable
- Faster training

**Observation from this project:**
- TCN training was significantly faster and stable across epochs.

---

## 2. Gradient Flow

### LSTM
- Suffers from vanishing/exploding gradients
- Long sequences are difficult to learn

### TCN
- Uses residual connections
- Stable gradient flow
- Handles long dependencies better

 **In our implementation:**
- TCN showed smooth loss convergence (0.52 → 0.0007)

---

## 3. Receptive Field (Capturing Dependencies)

### LSTM
- Depends on hidden state memory
- Hard to control how much history is captured

### TCN
- Uses dilated convolutions
- Receptive field grows exponentially
- Can capture long-range dependencies efficiently

---

## 4. Architectural Simplicity

### LSTM
- Complex sequential logic
- Slower inference

### TCN
- Simple convolutional structure
- Faster inference

---

## 5. Performance in This Project

| Metric | TCN |
|-------|-----|
| Training Loss | 0.52 → 0.0007 |
| Stability | High |
| Training Speed | Fast |
| Anomaly Detection | Effective |

---

## Conclusion

TCN outperforms LSTM for this task due to:

- Faster training via parallel computation  
- Better gradient stability  
- Efficient long-range dependency capture  
- Simpler and scalable architecture  

Therefore, TCN is a more suitable choice for real-time anomaly detection systems.

---