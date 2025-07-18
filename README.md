# EnergyGen-CVAE

**Conditional Variational Autoencoder for Synthetic Power Load Generation Based on Environmental Features**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)![PyTorch](https://img.shields.io/badge/built%20with-PyTorch-red?logo=pytorch&logoColor=white)![Task](https://img.shields.io/badge/task-Data%20Augmentation%20%7C%20Extrapolation-orange?logo=chart-bar)[![Application](https://img.shields.io/badge/application-Energy%20Simulation-lightgrey?logo=lightning&logoColor=black)]()

---

## 🔍 Overview

`EnergyGen-CVAE` is a lightweight and extensible generative framework for synthesizing realistic power consumption data, conditioned on external environmental factors such as temperature, humidity, wind speed, and solar radiation. This project leverages the **Conditional Variational Autoencoder (CVAE)** to model uncertainty and generate diverse yet physically-plausible time-series samples, supporting downstream applications in **forecasting**, **scenario simulation**, and **data augmentation**.

This work is inspired by real-world challenges in smart grid load simulation, where labeled power consumption data is often limited, noisy, or costly to collect.

![what_we_do](./assets/cvae_architecture_diagram.svg)

![8cafc92ce67384a64dc33254065d8f4](./assets/8cafc92ce67384a64dc33254065d8f4.png)

------

### 📊 Dataset Description

We use a structured time-series dataset consisting of **environmental regressors** and **zone-wise electricity consumption targets**. Each row corresponds to a 10-minute interval sample describing both external environmental conditions and internal energy usage across multiple zones.

#### ➤ Input Features (Regressors):

| Feature Name          | Description                     | Unit |
| --------------------- | ------------------------------- | ---- |
| `Temperature`         | Outdoor temperature             | °C   |
| `Humidity`            | Relative humidity               | %    |
| `WindSpeed`           | Wind speed                      | m/s  |
| `GeneralDiffuseFlows` | Global diffuse solar irradiance | W/m² |
| `DiffuseFlows`        | Diffuse irradiance              | W/m² |

These regressors serve as the **conditioning inputs** to the CVAE model, enabling the generation of energy consumption patterns under different weather and solar scenarios.

------

#### 🎯 Target Variables (Outputs):

| Target Name              | Description                               | Unit |
| ------------------------ | ----------------------------------------- | ---- |
| `PowerConsumption_Zone1` | Power usage in Zone 1 (e.g., office area) | kWh  |
| `PowerConsumption_Zone2` | Power usage in Zone 2 (e.g., HVAC system) | kWh  |
| `PowerConsumption_Zone3` | Power usage in Zone 3 (e.g., lighting)    | kWh  |
| `TotalLoad`              | Aggregated consumption across all zones   | kWh  |

The **model aims to learn the conditional distribution** of these target values given the environmental inputs, and to generate realistic, uncertainty-aware consumption predictions.

### 

| Feature Name | Description                     |
| ------------ | ------------------------------- |
| `Hour`       | Hour of day (0–23)              |
| `DayOfWeek`  | Day of week (0=Sun, ..., 6=Sat) |
| `Month`      | Month index (1–12)              |

---

## 📌 Key Features

- ✅ **Multi-zone modeling**: Simultaneously generate consumption data for multiple power zones (Zone1, Zone2, Zone3).
- 📊 **Uncertainty visualization**: Output confidence intervals for generated time series.
- ⚡ **Condition-aware generation**: Capture dynamic influence of temperature, humidity, wind, and irradiance.
- 🧠 **Scalable CVAE structure**: Easily replace encoder/decoder architectures.
- 🔁 **Data augmentation ready**: Useful for enriching load prediction datasets.

![7bfebf08824bd51932bf9918011a4e0](./assets/7bfebf08824bd51932bf9918011a4e0.png)


---

## 🧠 Methodology

We implement a CVAE where the conditional input $x \in \mathbb{R}^d$ includes external weather features and the target $y \in \mathbb{R}^k$ is the actual consumption vector.

### Key techniques:

* KL annealing
* MSE + reconstruction loss
* Per-zone decoder output
* Random z sampling for diverse generation

### Loss function:

![image-20250713235656107](./assets/image-20250713235656107.png)

---

## 📈 Evaluation

We compare model performance using:

* RMSE / KLD / MMD between generated and actual data
* Coverage & width of confidence intervals
* Visual fidelity across zones and time spans

Visualization:

This visualization demonstrates the predictive capability of our CVAE-based energy consumption model. Given a set of environmental inputs — temperature, humidity, lighting intensity, time, weekday, and occupancy rate — the model generates energy usage forecasts across multiple building zones.

What’s significant is the model’s **ability to generalize beyond the training distribution**. By inputting modified or unseen contextual variables (such as slightly shifted `cons` patterns or new time slots), the system extrapolates plausible power consumption across zones, enabling simulation in out-of-distribution (OOD) scenarios and fine-grained energy control optimization.

![image-20250713235900865](./assets/image-20250713235900865.png)

---

### 🧪 Example Usage

```python
example_input = {
    "Temperature": 9.0,
    "Humidity": 86.7,
    "WindSpeed": 0.08,
    "GeneralDiffuseFlows": 117.2,
    "DiffuseFlows": 30.43,
    "Hour": 9,
    "DayOfWeek": 0,
    "Month": 1
}

generated = generate_power_consumption(example_input)
for zone, value in generated.items():
    print(f"{zone}: {value:.2f} kWh")

```

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

## 📚 Citation

If you find this project helpful in your research, please consider citing:

```latex
@misc{energygen_cvae_2025,
  title        = {EnergyGen-CVAE: Conditional Variational Autoencoder for Synthetic Power Load Generation},
  author       = {Yao Wu},
  institution  = {Westlake University},
  year         = {2025},
  howpublished = {\url{https://github.com/Republic1024/EnergyGen-CVAE}},
  note         = {Available at GitHub}
}

```

---

## 📄 License

This project is released under the MIT License.

----

## ⭐️ Note

If you find this project helpful, please ⭐️ star the repo or cite our work!
