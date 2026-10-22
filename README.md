# 📦 Indian E-Commerce Logistics: Expected Delivery Date (EDD) & SLA Prediction

**M.Sc. Data Science & Artificial Intelligence Project**  
*Ramakrishna Mission Vivekananda Educational and Research Institute (RKMVERI)*  
**Author:** Santanu Samanta  

---

## 🚀 Project Overview
Accurate Expected Delivery Date (EDD) forecasting and Service Level Agreement (SLA) breach prediction are critical for modern e-commerce supply chains. This project simulates and analyzes **25,000 domain-calibrated shipment records** across 6 major Indian logistics carriers (**Blue Dart, Delhivery, Ekart, XpressBees, Ecom Express, and Shadowfax**).

It is designed for:
1. **EDD / Transit Time Regression:** Predicting package transit duration in hours.
2. **SLA Breach Classification:** Identifying orders at risk of missing delivery deadlines.
3. **Multi-Carrier Cost-vs-Speed Optimization:** Benchmarking freight costs against carrier reliability.

---

## 📊 Dataset Specifications
* **Records:** 25,000 simulated shipments
* **Geographical Scope:** Pan-India 6-digit pincode prefixes mapped across Zones A (Intra-City) to E (NE / J&K / Special).
* **Operational Rules:** Includes warehouse pickup cutoffs (16:00 daily cutoff), line-haul transport modes (Surface vs. Air), weather disruptions (fog/monsoon), and festive peak surges.
* **Kaggle Dataset:** [Indian Multi-Agency E-Commerce Logistics & EDD](https://www.kaggle.com/shyamsantanu01122004/indian-multi-agency-e-commerce-logistics-and-edd) *(Usability Score: 10.0 / 10.0)*

---

## 🛠️ Machine Learning Models & Results
To avoid **target leakage**, downstream indicators (`Expected_Delivery_Timestamp`, `Actual_Delivery_Days`, `Is_Delayed_SLA`) were excluded during feature engineering for transit time regression.

| Model | RMSE (Hours) | MAE (Hours) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Ridge Regression** | 14.25 | 11.10 | 0.8240 |
| **Lasso Regression** | 14.32 | 11.18 | 0.8221 |
| **Random Forest Regressor** | **4.82** | **3.41** | **0.9792** |

---

## 📂 Repository Structure
* `india_multi_agency_edd_dataset.csv` – Raw simulated dataset.
* `indian_logistics_edd_eda_ml.ipynb` – Jupyter Notebook containing Exploratory Data Analysis (EDA), zone distribution plots, carrier benchmarks, and baseline ML training scripts.
