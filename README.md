# Smart Estimator — Interactive ML Valuation & EDA Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[🇮🇷 مستندات فارسی (Persian Documentation)](README.fa.md) • [Report Issues](https://github.com/abbasi0abolfazl/smart-estimator/issues)

An interactive, production-structured machine learning application built with **Streamlit** and **Scikit-Learn** for multi-domain market valuation. The platform combines ensemble regressors for consumer hardware pricing (laptops) and non-linear tree models for metropolitan real-estate valuation (over 160,000 property records in Iran), supplemented by an automated Exploratory Data Analysis (EDA) engine.

---

## 📌 Architecture & Data Pipeline

The system is organized into three decoupled layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Data Ingestion Layer                          │
│  - Laptop Market Dataset: Hardware specs, PPI, weight, MSRP (€)        │
│  - Iran Real Estate Dataset: 160,000+ scraped listings (Tehran & hubs) │
│  - User Upload Bus: Custom tabular ingest (.csv, .xlsx)                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Feature & Model Pipeline                        │
│  - Preprocessing: Outlier pruning, LabelEncoding, log transforms       │
│  - Laptop Regressor: Random Forest Ensemble (variance reduction)       │
│  - Property Regressor: Decision Tree Regressor (regional segmentation) │
│  - Evaluation Telemetry: Live R² score, MAE, MSE, and residual curves  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       Interactive Presentation UI                      │
│  - RTL-compatible Streamlit dashboard with custom CSS injection       │
│  - Dual-currency valuation engine: Local IRR/Toman and EUR (€)         │
│  - EDA Suite: Correlation heatmaps, distribution plots, metrics matrix │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Modeling & Formulation

### 1. Laptop Valuation Engine (Random Forest Regression)
Predicts global hardware value using an ensemble of decision trees:
$$\hat{y}_{\text{laptop}} = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})$$
Where $\mathbf{x}$ comprises engineered hardware features:
- **Display:** Screen size, panel resolution, Pixels Per Inch ($\text{PPI} = \frac{\sqrt{w^2 + h^2}}{\text{diag}}$).
- **Compute:** CPU brand, clock frequency, core architecture, and RAM allocation.
- **Form Factor:** Storage configuration (SSD/HDD capacity) and unit weight.

### 2. Real Estate Valuation Engine (Decision Tree Regression)
Trained on over **160,000 verified listings** to handle multi-modal price distributions across geographic sub-markets:
- **Features:** Categorical neighborhood index ($\text{LabelEncoder}$), gross floor area ($m^2$), building age, floor level, and binary amenities (elevator, parking, storage).
- **Target Metrics:** Total sale valuation and full mortgage equivalent (Rahn-e Kamel).

### 3. Evaluation Telemetry
Model performance is evaluated against held-out validation sets:
- **Coefficient of Determination:** $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
- **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|$
- **Mean Squared Error (MSE):** $\text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$

---

## ✨ Features

- **Multi-Domain Estimator:** Seamless switching between electronics valuation and real-estate asset estimation.
- **Exploratory Data Analysis (EDA) Suite:**
  - Dynamic correlation matrix and Seaborn heatmaps.
  - Interactive feature distribution histograms.
  - Raw versus engineered dataset diff viewer.
- **Responsive RTL Design:** Styled using custom CSS and components for right-to-left layout clarity.
- **Custom Data Upload:** Users can inject proprietary `.csv` or `.xlsx` files for immediate inspection.

---

## 🛠️ Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/abbasi0abolfazl/smart-estimator.git
cd smart-estimator

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Application

```bash
streamlit run main.py
```

Open `http://localhost:8501` in your browser.

---

## 🚀 Cloud Deployment

The application is fully configured for zero-configuration deployment on **Streamlit Community Cloud**:
1. Fork or push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Select this repository and set `main.py` as the entry point.

---

## 📂 Project Structure

```
smart-estimator/
├── main.py              # Application entry point & page navigation logic
├── requirements.txt     # Runtime dependencies
├── CSV_data/            # Training and sample dataset storage
├── Processing/          # Data cleaning, encoding, and model pipelines
├── src/                 # Custom HTML/CSS styling assets
├── README.md            # English technical documentation
└── README.fa.md         # Persian technical documentation
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
