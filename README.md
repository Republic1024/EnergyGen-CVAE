# EnergyGen-CVAE

**Conditional Variational Autoencoder for Synthetic Power Load Generation Based on Environmental Features**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🔍 Overview

`EnergyGen-CVAE` is a lightweight and extensible generative framework for synthesizing realistic power consumption data, conditioned on external environmental factors such as temperature, humidity, wind speed, and solar radiation. This project leverages the **Conditional Variational Autoencoder (CVAE)** to model uncertainty and generate diverse yet physically-plausible time-series samples, supporting downstream applications in **forecasting**, **scenario simulation**, and **data augmentation**.

This work is inspired by real-world challenges in smart grid load simulation, where labeled power consumption data is often limited, noisy, or costly to collect.

---

## 📌 Key Features

- ✅ **Multi-zone modeling**: Simultaneously generate consumption data for multiple power zones (Zone1, Zone2, Zone3).
- 📊 **Uncertainty visualization**: Output confidence intervals for generated time series.
- ⚡ **Condition-aware generation**: Capture dynamic influence of temperature, humidity, wind, and irradiance.
- 🧠 **Scalable CVAE structure**: Easily replace encoder/decoder architectures.
- 🔁 **Data augmentation ready**: Useful for enriching load prediction datasets.


---

## 🧠 Methodology

We implement a CVAE where the conditional input $x \in \mathbb{R}^d$ includes external weather features and the target $y \in \mathbb{R}^k$ is the actual consumption vector.

Key techniques:

* KL annealing
* MSE + reconstruction loss
* Per-zone decoder output
* Random z sampling for diverse generation

---

## 📈 Evaluation

We compare model performance using:

* RMSE / MAE between generated and actual data
* Coverage & width of confidence intervals
* Visual fidelity across zones and time spans

---

## 💼 Applications

* 🔌 Smart grid load simulation
* 🧪 Synthetic scenario testing
* 📉 Data-driven forecasting model pretraining
* 📊 Enhancing robustness of anomaly detection models

---

## 🤝 Acknowledgements

This work was initially developed as part of a time-series modeling research module. The structure draws inspiration from academic works on CVAE and real-world energy modeling.

---

## 📄 License

This project is released under the MIT License.


