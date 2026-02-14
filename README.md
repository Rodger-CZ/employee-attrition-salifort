# Employee Attrition Risk Analysis – Salifort Motors

## 📌 Project Overview

Employee turnover can significantly impact productivity, operational costs, and organizational stability.
This project analyzes workforce data from **Salifort Motors** to identify key drivers of employee attrition and build a predictive model that estimates the probability of employees leaving the company.

The analysis is structured as a real-world HR analytics case study, combining exploratory analysis, machine learning, and business recommendations.

---

## 🎯 Objectives

* Understand workforce characteristics and attrition patterns
* Identify factors associated with employee turnover
* Build predictive models to estimate attrition risk
* Provide actionable recommendations for HR decision-making

---

## 🛠️ Tools & Technologies

* Python
* Jupyter Notebook
* pandas, numpy
* matplotlib, seaborn
* scikit-learn

---

## 📂 Project Structure

```
employee-attrition-salifort/
│
├── data/
│   └── HR_capstone_dataset.csv
├── notebooks/
│   └── employee_attrition_salifort.ipynb
├── README.md
└── requirements.txt
```

---

## 🔍 Methodology

### 1. Data Preparation

* Checked for missing values and duplicates
* Encoded categorical variables
* Split data into training and testing sets

### 2. Exploratory Data Analysis

Key patterns identified:

* Employees with **low satisfaction levels** are more likely to leave
* **High working hours and workload** increase attrition risk
* Lack of **promotion** is associated with higher turnover
* Certain departments show higher attrition rates

---

### 3. Modeling

Models evaluated:

* Decision Tree Classifier
* Random Forest Classifier (with GridSearchCV)

Model tuning was performed using cross-validation with **ROC-AUC** as the primary evaluation metric.

---

## 📊 Results

* Random Forest achieved strong predictive performance
* Key predictors of attrition include:

  * Satisfaction level
  * Average monthly hours
  * Number of projects
  * Time spent at company
  * Promotion history

---

## 💡 Business Recommendations

* Monitor employees with low satisfaction and high workload
* Implement workload balancing and burnout prevention programs
* Improve internal promotion opportunities
* Focus retention efforts on high-risk departments

---

## ⚠️ Disclaimer

This project is for educational and portfolio purposes only. The dataset is simulated and does not represent real employee data.

---

## 👤 Author

**Faustine Rodgers**
Data Scientist | Machine Learning | People Analytics
