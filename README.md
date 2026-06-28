# 🛤️ The Apex Rail — Market Intelligence Dashboard

> **MBA Data Insights Capstone Project**  
> **Author:** Nilay Bhagat  
> **Institution:** SP Jain School of Global Management  

---

## 📌 Business Problem

As a Mechanical Engineering + MBA student, I designed **"The Apex Rail"** — a premium,
precision-machined 6061 billet aluminum interior rail system that mounts non-destructively
into existing vehicle dashboard panel gaps using a proprietary zero-tolerance wedge-locking
mechanism.

**The core SCM strategy is Assemble-to-Order (Postponement):**
- Universal rails and modules are mass-produced cheaply
- Only the small, vehicle-specific mounting bracket is made on demand
- This keeps finished-goods inventory lean and eliminates overstock risk

This dashboard validates market demand through a 1,000-respondent consumer survey
using four analytics dimensions and three ML algorithms.

---

## 🎯 Dashboard Pages

| Page | Analytics Type | Key Output |
|------|---------------|------------|
| 🏠 Home | Overview | Project summary + navigation |
| 📊 Market Overview | **Descriptive** | Distributions, pain points, module demand |
| 🎯 K-Means Clustering | **Diagnostic + Clustering** | 4 buyer personas, elbow + silhouette |
| 💰 Pricing Strategy | **Prescriptive + Statistical** | Van Westendorp OPP = $106 |
| 🛒 Market Basket Analysis | **Diagnostic + Association** | 42 Apriori bundle rules |
| 🤖 Random Forest | **Predictive + Classification** | 87.2% accuracy, 94.2% AUC, live predictor |
| 📝 Recommendations | **Prescriptive** | 6 evidence-based action points |

---

## 🤖 Algorithms Used

### 1. K-Means Clustering (Unsupervised)
- Groups 1,000 respondents into 4 distinct buyer personas
- Validated using **Elbow Method** + **Silhouette Score**
- Optimal K=4 confirmed by both metrics
- 3D interactive scatter plot for visualisation

### 2. Random Forest Classifier (Supervised)
- Predicts pre-order intent (binary: reserve / don't reserve)
- 300 decision trees, stratified 75/25 train-test split
- **5-Fold Stratified Cross-Validation** for academic rigour
- Results: Train 100% | Test 87.2% | ROC-AUC 94.2% | CV AUC 94.01% ± 1.32%
- Includes live interactive predictor tool

### 3. Apriori Association Rules (Unsupervised)
- Finds which modules buyers select together
- min_support=0.10, lift > 1.0
- 42 rules found; filterable by confidence, lift, and segment
- Co-purchase heatmap + segment bundle strategy

### 4. Van Westendorp Price Sensitivity Model (Statistical)
- 4 price perception curves
- Key intersections: OPP=$106 | IAP=$184 | PMC=$157 | PME=$128
- Tiered pricing strategy across 4 segments

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Respondents | 1,000 |
| Features | 21 columns |
| Missing values | 0 |
| Outliers | 15 injected (realistic) |
| Skewness | Log-normal pricing distributions |
| Binary noise | 3% random flag flips |

**4 Market Segments:**
| Segment | Size | Pre-Order Intent | Bargain WTP |
|---------|------|-----------------|-------------|
| Elite Enthusiast | 27% | 95.5% | $170 |
| Family / WorkMom | 29% | 40.7% | $118 |
| Tech Purist | 24% | 75.6% | $141 |
| Practical Upgrader | 20% | 16.6% | $65 |

---

## 🚀 Installation & Running Locally

**Requirements:** Python 3.11+

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/apex-rail-market-intelligence.git
cd apex-rail-market-intelligence

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

Upload `apex_rail_survey_data_v2.csv` using the sidebar to load the data.

---

## 📦 Project Structure

```
apex-rail-market-intelligence/
├── app.py                          # Complete Streamlit dashboard (single file)
├── requirements.txt                # Python dependencies
├── apex_rail_survey_data_v2.csv    # Synthetic survey dataset (1,000 rows)
└── README.md                       # This file
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-purple?logo=plotly)
![MLxtend](https://img.shields.io/badge/MLxtend-0.23+-green)

---

## 📈 Key Results

- **Pre-order intent rate:** 59.1% of 1,000 respondents
- **Optimal launch price:** $106 (Van Westendorp OPP)
- **Best ML model:** Random Forest — 87.2% accuracy, 94.2% ROC-AUC
- **Cross-validation AUC:** 94.01% ± 1.32% (5-fold, stable model)
- **Top module demand:** MagSafe Mount at 76%
- **Primary target segment:** Elite Enthusiasts (95.5% intent, $170 WTP)

---

## 📝 Survey Design

The 1,000-respondent synthetic dataset mirrors a 13-question hybrid survey:

- **Section 1:** Vehicle demographics (brand, year, trim)
- **Section 2:** Design philosophy (Purist / Functionalist / Enthusiast)
- **Section 3:** Cabin friction pain points (binary flags + severity scale)
- **Section 4:** Module demand (5 binary module flags)
- **Section 5:** Van Westendorp pricing (4 continuous price points)
- **Section 6:** Pre-order intent (binary target variable)

---

*Built with Streamlit · Plotly · Scikit-learn · MLxtend*  
*MBA Data Insights Project — Nilay Bhagat*
