# Machine Learning Practical Assignment
## From Theory to Implementation: Data Preprocessing & Feature Selection

* **Course:** Machine Learning
* **Assignment Type:** Group Practical Assignment
* **Learning Philosophy:** Theory → Formula/Logic → From-Scratch Code → Output → Library Verification → Interpretation

---

## 1. Project Title
**End-to-End From-Scratch Data Preprocessing and Feature Selection Pipeline for Employee Attrition Classification**

---

## 2. Problem Statement
Employee attrition poses significant financial, organizational, and operational challenges to modern enterprises. Unplanned departures lead to high replacement costs (recruitment, onboarding, training), loss of institutional memory, decreased team morale, and project delays.

The objective of this project is to build a mathematically rigorous, leakage-free data preprocessing and filter-based feature selection pipeline entirely from scratch using fundamental Python and NumPy. We identify the key organizational, behavioral, and demographic drivers of workforce attrition and validate their predictive impact using machine learning classifiers.

---

## 3. Group Members

| S.No. | Student Name | Roll Number |
| :---: | :--- | :---: |
| 1 | **Chetna Yadav** | 23001390015 |
| 2 | **Suryansh Gupta** | 23001390045 |
| 3 | **Shraddha Yadav** | 23001390041 |

---

## 4. Dataset Description
* **Dataset Name:** IBM HR Analytics Employee Attrition & Performance Dataset
* **Total Observations:** 1,470 records
* **Total Initial Features:** 35 features (34 input features + 1 target variable)
* **Target Variable:** `Attrition` (Binary: `"Yes"` = 1, `"No"` = 0)
* **Variable Types:**
  * **Numerical (26):** Age, DailyRate, DistanceFromHome, Education, HourlyRate, MonthlyIncome, MonthlyRate, NumCompaniesWorked, PercentSalaryHike, TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, etc.
  * **Categorical (9):** Attrition, BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, Over18, OverTime.

---

## 5. Dataset Source
* **Source:** IBM Watson Analytics / [Kaggle IBM HR Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
* **Local Path:** [`dataset/hr_attrition_assignment.csv`](dataset/hr_attrition_assignment.csv)

---

## 6. Preprocessing Techniques Implemented
1. **Initial Inspection & Statistical Auditing:** Minimum, Maximum, Mean, Median, Mode, Variance, Standard Deviation, Range from scratch.
2. **Data Integrity & Consistency Checking:** Detection and elimination of impossible ages ($<18$ or $>100$), negative metrics, and whitespace casing issues.
3. **Leakage-Free Train-Test Splitting:** 80/20 train-test partition using random index permutations from scratch.
4. **Missing Value Imputation:** Parameterized training-set median imputation for skewed numerical features (`MonthlyIncome`, `TotalWorkingYears`).
5. **Categorical Encoding:**
   * **Label Encoding:** Binary and ordinal mapping using custom dictionary lookups.
   * **One-Hot Encoding:** From-scratch expansion of nominal multi-class categories into orthogonal binary indicator vectors.
6. **Outlier Detection:** IQR boundaries ($Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}$) and Z-Score rule ($|Z| > 3$).
7. **Data Transformation:** Log transformation ($\ln(1 + X)$) to reduce right-skewness in compensation distributions.
8. **Feature Scaling:** From-scratch Min-Max Normalization ($[0, 1]$) and Standardization ($Z = \frac{X-\mu}{\sigma}$) using training parameters.

---

## 7. Feature-Selection Techniques Implemented
1. **Variance Threshold:** Calculated population variance from scratch to detect and discard constant zero-variance features (`EmployeeCount`, `StandardHours`, `Over18_Y`).
2. **Pearson Correlation Analysis:** Calculated correlation coefficient $r$ from scratch; detected and eliminated severe collinearity ($r = 0.95$) between `JobLevel` and `MonthlyIncome`.
3. **Chi-Square Test of Independence ($\chi^2$):** Formulated contingency matrices, observed/expected cell frequencies, and degrees of freedom $df = (r-1)(c-1)$ from scratch to evaluate categorical feature associations (`OverTime` vs `Attrition`).
4. **One-Way ANOVA F-Test:** Computed overall mean, group means, between-group variance (MSB), and within-group variance (MSW) from scratch to evaluate continuous predictors across attrition classes (`MonthlyIncome` vs `Attrition`).
5. **Mutual Information (Information Gain):** Computed Shannon Entropy $H(Y)$ in bits, conditional entropy $H(Y|X)$, and Mutual Information $MI(X; Y) = H(Y) - H(Y|X)$ from scratch for discrete/categorical features (`WorkLifeBalance`).

---

## 8. From-Scratch Implementations & Verification Table

| Technique | Mathematical Formula / Algorithm | From-Scratch Function | Library Benchmark | Verification Result |
| :--- | :---: | :--- | :--- | :---: |
| **Mean & Variance** | $\bar{X} = \frac{\sum X_i}{n}, \ \sigma^2 = \frac{\sum (X_i-\bar{X})^2}{n}$ | `calculate_stats_from_scratch` | `pd.Series.mean()`, `.var(ddof=0)` | 100% Match |
| **Median & Mode** | 50th Percentile & $\arg\max \text{count}(X)$ | `calculate_stats_from_scratch` | `pd.Series.median()`, `.mode()` | 100% Match |
| **Train-Test Split** | Random Permutation Indexing | `manual_train_test_split` | `sklearn.model_selection.train_test_split` | Verified |
| **One-Hot Encoding** | Category Indicator Matrix | `manual_one_hot_encode` | `sklearn.preprocessing.OneHotEncoder` | 100% Match |
| **Standardization** | $Z = \frac{X - \mu_{train}}{\sigma_{train}}$ | `manual_standardization` | `sklearn.preprocessing.StandardScaler` | 100% Match |
| **Min-Max Scaling** | $X' = \frac{X - X_{min}}{X_{max} - X_{min}}$ | `manual_min_max_scale` | `sklearn.preprocessing.MinMaxScaler` | 100% Match |
| **Pearson $r$** | $\frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \sum(y_i-\bar{y})^2}}$ | `manual_pearson_correlation` | `scipy.stats.pearsonr` | 100% Match |
| **Chi-Square $\chi^2$**| $\sum \frac{(O-E)^2}{E}, \ E=\frac{R \cdot C}{N}$ | `manual_chi_square` | `scipy.stats.chi2_contingency` | 100% Match |
| **ANOVA F-Test** | $F = \frac{\text{MSB}}{\text{MSW}}$ | `manual_anova_f` | `scipy.stats.f_oneway` | 100% Match |
| **Mutual Information**| $H(Y) - H(Y\|X)$ (in bits) | `manual_mutual_information` | `sklearn.feature_selection.mutual_info_classif` | 100% Match |

---

## 9. Results & Model Comparison

To validate the practical utility of our preprocessing and feature selection, we benchmarked a Random Forest classifier on **Model A (All Features)** vs **Model B (Selected Features)**:

| Metric / Parameter | Model A (All Features) | Model B (Selected Features) | Practical Impact |
| :--- | :---: | :---: | :--- |
| **Total Features** | 48 | **46** | Pruned uninformative & collinear inputs |
| **Training Time** | ~145 ms | **~125 ms** | Faster training & lower memory footprint |
| **Test Accuracy** | 86.73% | **87.07%** | Maintained / Slight improvement |
| **Test F1-Score** | 0.4444 | **0.4615** | Improved minority class detection |
| **ROC-AUC Score** | 0.8120 | **0.8195** | Enhanced class separability |

---

## 10. Selected Features
* **Retained Features (46 ML-Ready Features):**
  * `OverTime` (Strongest categorical predictor, $\chi^2 = 62.48, p < 0.001$)
  * `MonthlyIncome` & `MonthlyIncome_Log` (Strongest financial predictor, $F = 38.45, p < 0.001$)
  * `TotalWorkingYears` ($F = 39.80, p < 0.001$)
  * `Age` ($F = 30.12, p < 0.001$)
  * `WorkLifeBalance` ($MI = 0.012\text{ bits}$)
  * `JobSatisfaction`, `EnvironmentSatisfaction`, `DistanceFromHome`, `YearsAtCompany`, `YearsWithCurrManager`, One-hot encoded `JobRole`, `Department`, `MaritalStatus`.
* **Pruned / Dropped Features (5 Features):**
  1. `EmployeeCount` (Zero variance, constant value = 1)
  2. `StandardHours` (Zero variance, constant value = 80)
  3. `Over18_Y` (Zero variance, constant value = 1)
  4. `EmployeeNumber` (Arbitrary identifier)
  5. `JobLevel` (Severe multicollinearity with `MonthlyIncome`, $r = 0.950$)

---

## 11. Key Findings
1. **Overtime is the #1 Behavioral Risk Factor:** Employees working overtime experience nearly triple the attrition rate compared to non-overtime staff.
2. **Income & Tenure Disparity:** Lower salary bands in early career stages (Job Levels 1-2, $<5$ years total experience) drive the highest proportion of departures.
3. **Departmental Differences:** Sales and HR departments experience significantly higher turnover (~20.6% and ~19.0%) than Research & Development (~13.8%).
4. **Multicollinearity Elimination:** Removing `JobLevel` eliminated redundancy with `MonthlyIncome` while preventing coefficient instability in linear models.

---

## 12. Instructions to Run the Code

### Prerequisites
* Python 3.9+
* Required Libraries: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `ipython`

### Running the Notebook
```bash
# Clone the repository
git clone https://github.com/Chetna015/ML_ASSIGNMENT.git
cd ML_ASSIGNMENT

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter Notebook
jupyter notebook notebooks/main_analysis.ipynb
```

### Running Modular Scripts
```bash
# Run preprocessing and feature selection modules
python src/preprocessing.py
python src/feature_selection.py
```

---

## 13. Google Colab Link
* [Open in Google Colab](https://colab.research.google.com/github/Chetna015/ML_ASSIGNMENT/blob/main/notebooks/main_analysis.ipynb)
* **GitHub Repository:** [https://github.com/Chetna015/ML_ASSIGNMENT](https://github.com/Chetna015/ML_ASSIGNMENT)

---

## Repository Structure
```
ML-Preprocessing-Feature-Selection/
│
├── README.md
├── requirements.txt
├── Untitled1.ipynb
│
├── dataset/
│   ├── hr_attrition_assignment.csv
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── notebooks/
│   └── main_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   └── feature_selection.py
│
├── results/
│   ├── graphs/
│   │   ├── 01_histograms_distributions.png
│   │   ├── 02_boxplot_income_attrition.png
│   │   ├── 03_barchart_department_attrition.png
│   │   ├── 04_scatterplot_age_income_attrition.png
│   │   ├── 05_correlation_heatmap.png
│   │   └── 06_log_transformation_comparison.png
│   └── outputs/
│
└── report/
    └── final_summary.md
```
