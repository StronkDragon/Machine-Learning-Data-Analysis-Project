# 📊 OUALD Student Performance & Engagement Analysis

A Data Science and Machine Learning project focused on analyzing student demographic data, early VLE (Virtual Learning Environment) interaction metrics, and assessment performance from the **Open University Analysis Learning Dataset (OUALD)** to perform exploratory data analysis and build predictive models.

---

## 📌 Project Overview

Understanding student behavior and early engagement indicators is crucial for identifying students at risk of academic failure or withdrawal. This project explores learning analytics to:

1. **Perform Exploratory Data Analysis (EDA)** on demographics, early assessment scores, registration/unregistration dates, and platform activity (clicks).
2. **Preprocess and Clean Data** for accurate statistical inference and machine learning ingestion.
3. **Analyze Multi-Channel Student Engagement** (forum interactions, quiz usage, glossary references, wiki page activity, etc.).
4. **Train and Evaluate Machine Learning Models** (Decision Trees, Linear Regression, Ridge Regression) for predictive modeling tasks.

---

## 📁 Dataset Details

The analysis utilizes a custom subset of the Open University Learning Analytics Dataset:
* **`CB_OUALD_train.csv`**: Training dataset containing **18,802 instances**.
* **`CB_OUALD_test.csv`**: Evaluation dataset for model verification.
Full dataset can be checked [here](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad)

### Key Feature Categories
* **Demographics & Background:** `gender`, `region`, `highest_education`, `imd_band`, `age_band`, `disability`, `studied_credits`, `num_of_prev_attempts`.
* **Course & Timeline:** `code_module`, `code_presentation`, `module_presentation_length`, `date_registration`, `date_unregistration`.
* **Early Performance Metrics:** `n_assessments_early`, `mean_score_early`, `weighted_mean_score_early`, `submission_rate_early`.
* **VLE Click Interactions (`clicks_*`):** Early interactions across platform components including `clicks_forumng`, `clicks_homepage`, `clicks_oucontent`, `clicks_quiz`, `clicks_subpage`, `clicks_resource`, `clicks_url`, and others.

---

## 🛠️ Project Workflow & Methodology

```
Data Ingestion ➔ Exploratory Data Analysis ➔ Feature Engineering & Preprocessing ➔ Model Training & Evaluation
```

### 1. Data Ingestion & Setup
* Environment setup using Python data science libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`).
* Ingestion of training and testing sets.

### 2. Exploratory Data Analysis (EDA)
* Categorized variables into **32 Numerical Attributes** and **9 Categorical Attributes**.
* Generated statistical summaries (Mean, Standard Deviation, Percentiles, Min/Max) across all continuous numerical attributes.
* Examined missing values across temporal data (e.g., `date_unregistration`) and activity logs.

### 3. Data Preprocessing & Feature Engineering
* **Missing Value Imputation:** Handled sparse attributes using `SimpleImputer`.
* **Encoding:** Applied `OneHotEncoder` and `LabelEncoder` for categorical variables.
* **Scaling:** Standardized continuous variables using `StandardScaler` for regularized models.

### 4. Machine Learning Implementation
* **Classification:** Built Decision Tree Classifiers (`DecisionTreeClassifier`) to model categorical performance or retention flags.
* **Regression:** Implemented Linear Regression (`LinearRegression`) and L2 Regularized Ridge Regression (`Ridge`) to predict numerical performance metrics (e.g., scores or total early engagement).
* **Evaluation:** Assessed models using standard metrics including `classification_report`, `confusion_matrix`, `accuracy_score`, `mean_squared_error` (MSE), and `mean_absolute_error` (MAE).

---


## 🚀 How to Run

**Just upload the ipynb file and train + test datasets in Google Colab and run the code!**
---

## 📈 Summary of Visualizations

The notebook includes visualizations using `seaborn` and `matplotlib` to demonstrate:
* Distribution of early assessment scores and click frequencies.
* Correlation between early active days (`active_days_early`) and weighted assessment scores.
* Feature statistics across different student demographic groups.
