#  Full Waveform Inversion — Seismic Velocity Prediction

A deep learning project that predicts **subsurface velocity maps** from **seismic waveform recordings** using a UNet architecture — built for the Kaggle Full Waveform Inversion Competition.

---

## What is this project?

When seismic waves travel through the earth, different rock layers transmit them at different speeds. This project uses deep learning to work **backwards** from recorded seismic waves to predict what's underground — without drilling.

- **Input:** 5 seismic shot recordings — shape `(5, 1000, 70)`
- **Output:** 2D velocity map — shape `(1, 70, 70)` in range 1500–4500 m/s

---

## Model

Custom **UNet** with a seismic encoder that compresses 1000 time steps into a 70×70 spatial map, then predicts the underground velocity structure using encoder-decoder with skip connections.

**12.5M parameters**

---

## Project Structure
```
waveform-inversion/
├── data/
│   └── train_samples/        # Download from OpenFWI (see below)
├── notebooks/
│   └── 01_eda.ipynb          # Full pipeline notebook
├── src/
│   ├── dataset.py            # PyTorch Dataset
│   ├── model.py              # UNet model
│   ├── train.py              # Training loop
│   └── utils.py
├── outputs/
│   └── checkpoints/          # Saved model weights
├── requirements.txt
└── README.md
```

---

## Getting Started
```bash
# 1. Clone the repo
git clone https://github.com/mudavatuchandana/waveform-inversion.git
cd waveform-inversion

# 2. Create virtual environment
python -m venv env
source env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Download Dataset

Download the dataset from **[OpenFWI](https://openfwi-lanl.github.io/)** and place it inside `data/train_samples/`.

The dataset contains 8 geological scenario types:
- CurveVel A/B, FlatVel B
- CurveFault A, FlatFault B
- Style A/B

---

## Results (2 epochs)

| Epoch | Val Loss | Val MAE |
|-------|----------|---------|
| 1 | 0.1556 | 415 m/s |
| 2 | 0.1426 | 383 m/s |

---

## Acknowledgements

- [Kaggle FWI Competition](https://www.kaggle.com/competitions/waveform-inversion)
- [OpenFWI Dataset](https://openfwi-lanl.github.io/)
- [UNet Paper](https://arxiv.org/abs/1505.04597)