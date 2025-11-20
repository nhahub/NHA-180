# proMaint: Predictive Maintenance Project

## 🧾 Project Overview
**proMaint** is a data science project aimed at **predicting machine failures before they occur**.  
The project uses operational data from industrial machines, including temperatures, torque, rotational speed, tool wear, and product information.  
The main goal is to build a model capable of **accurately detecting potential failures** and improving maintenance efficiency.

---

## 📊 Exploratory Data Analysis (EDA)
The project started with exploratory data analysis (EDA), which revealed:

- The dataset is **clean and complete with no missing values**.  
- There is a **significant class imbalance** in the machine failure target.

### Observed Relationships:
- **Air temperature ↔ Process temperature:** Strong positive correlation.  
- **Rotational speed ↔ Torque:** Negative correlation due to machine dynamics.  
- **HDF:** Strongest individual predictor of machine failure.

### Outlier Analysis:
- Extremely high **rotational speeds** and **excessive torque** are **not errors**, but direct **indicators of machine failure**.  
- These outliers were transformed into meaningful features during **feature engineering**.

---

## 🧠 Feature Engineering
Instead of removing outliers, we **converted them into new features** that reflect machine behavior and operating conditions:

- **Torque & Rotational Speed Levels:** Categorized as *low*, *normal*, and *high*, where high values indicate potential failure risk.  
- **Tool Wear Severity:** Categorized as *low*, *medium*, and *high*, where high values indicate imminent failure.  
- **Heat Dissipation Gap:** Difference between process temperature and air temperature, reflecting the machine’s ability to dissipate heat.  
- **Power Indicator:** Combination of torque and rotational speed, representing mechanical load intensity.

> These engineered features added **operational context** and improved the model’s ability to predict failures.

---

## 🤖 Model & Evaluation
A **Random Forest Classifier** was trained on the machine failure dataset.  

**Performance Metrics:**
- **Average F1 Score (Cross-Validation):** 0.987  
- **F1 Score (Test Set):** 0.985  
- **Accuracy:** 0.999  
- **Recall:** 0.971  
- **ROC AUC:** 0.993  

> The model can **accurately detect failures** even with highly imbalanced data.

---

## 💾 Deployment & Usage
The trained model along with preprocessing steps was saved for production use:  

- Users can **load the model** and generate **real-time predictions** for machine failure.  
- Maintains the **operational context** of the data for accurate predictions.

---

## ⚡ Key Insights
- Outliers in torque and rotational speed are **true indicators of failure**, not noise.  
- **HDF** is the strongest predictor of machine failure.  
- Engineered features like **Heat Dissipation Gap** and **Power Indicator** reflect **load, wear, and heat**, improving predictive accuracy.  
- The Random Forest model is **highly effective** in handling imbalanced data.

---
 

