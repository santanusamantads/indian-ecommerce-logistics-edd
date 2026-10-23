# 📦 Indian E-Commerce Logistics: Expected Delivery Date (EDD) & SLA Prediction

**M.Sc. Data Science & Artificial Intelligence Project**  
*Ramakrishna Mission Vivekananda Educational and Research Institute (RKMVERI)*  
**Author:** Santanu Samanta  

---

## 🚀 Project Overview
Accurate Expected Delivery Date (EDD) forecasting and Service Level Agreement (SLA) breach prediction are critical for modern e-commerce supply chains. This project simulates and analyzes **25,000 domain-calibrated shipment records** across **23 features** and **6 major Indian logistics carriers** (**Blue Dart, Delhivery, Ekart, XpressBees, Ecom Express, and Shadowfax**).

It is designed for:
1. **EDD / Transit Time Regression:** Predicting package transit duration in hours (`Transit_Time_Hours`).
2. **SLA Breach Classification:** Identifying orders at risk of missing delivery deadlines (`Is_Delayed_SLA`).
3. **Multi-Carrier Cost-vs-Speed Optimization:** Benchmarking freight costs against carrier reliability.

---

## 📊 Dataset Specifications
* **Records:** 25,000 simulated shipments (23 columns)
* **Geographical Scope:** Pan-India 6-digit pincode prefixes mapped across Zones A (Intra-City) to E (NE / J&K / Special).
* **Operational Rules:** Includes warehouse pickup cutoffs (16:00 daily cutoff), line-haul transport modes (Surface vs. Air), weather disruptions (fog/monsoon), and festive peak surges.
* **Kaggle Dataset:** [Indian Multi-Agency E-Commerce Logistics & EDD](https://www.kaggle.com/datasets/shyamsantanu01122004/indian-multi-agency-e-commerce-logistics-and-edd) *(Usability Score: 10.0 / 10.0)*

---

## 🛠️ Machine Learning Models & Results
To avoid **target leakage**, downstream indicators (`Expected_Delivery_Timestamp`, `Expected_Delivery_Date`, `Actual_Delivery_Days`, `Is_Delayed_SLA`) were strictly excluded during feature engineering for transit time regression.

### Task 1: EDD Transit Time Regression (`Transit_Time_Hours`)
| Model | RMSE (Hours) | MAE (Hours) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Multiple Linear Regression (OLS)** | 9.65 | 7.18 | 0.8687 |
| **Ridge Regression ($L_2, \alpha=1.0$)** | 9.65 | 7.18 | 0.8687 |
| **Lasso Regression ($L_1, \alpha=0.1$)** | 9.93 | 7.37 | 0.8610 |
| **Random Forest Regressor** | **9.55** | **7.07** | **0.8715** |

### Task 2: SLA Breach Classification (`Is_Delayed_SLA`)
* **Model:** Logistic Regression (Separating Hyperplane Classifier)
* **Test Accuracy:** **89.46%** | **F1-Score:** **0.7891**

---

## 📂 Repository Structure
* `india_multi_agency_edd_dataset.csv` – Raw simulated dataset (25,000 rows, 23 columns).
* `indian_logistics_edd_eda_ml.ipynb` – Jupyter Notebook containing Exploratory Data Analysis (EDA), covariance/correlation matrices, OLS/Ridge/LASSO/Random Forest regression, and Logistic Regression SLA classification.
* `requirements.txt` – Python library dependencies.
