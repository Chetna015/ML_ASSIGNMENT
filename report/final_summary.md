# Executive Summary & Technical Report: Data Preprocessing and Feature Selection

## 1. Project Background & Problem Formulation
Employee retention is a major organizational priority. Using the IBM HR Analytics Employee Attrition dataset ($N = 1,470$ observations, $35$ initial features), this project demonstrates how to transition from theoretical machine learning principles to rigorous, from-scratch implementation.

The workflow follows the core paradigm:
$$\text{Theory} \longrightarrow \text{Mathematical Formula} \longrightarrow \text{From-Scratch Implementation} \longrightarrow \text{Output} \longrightarrow \text{Library Verification} \longrightarrow \text{Interpretation}$$

---

## 2. Key Preprocessing Milestones

### 2.1 Exploratory Statistical Audit
All 8 major descriptive statistics (Minimum, Maximum, Mean, Median, Mode, Variance, Standard Deviation, and Range) were computed from first principles without library wrappers. Comparisons with NumPy and Pandas showed **100.00% numerical precision alignment**.

### 2.2 Strict Data Leakage Prevention
To prevent optimistic bias and test contamination:
1. An 80/20 train-test split was performed **first** using randomized index shuffling from scratch.
2. Missing value medians for `MonthlyIncome` and `TotalWorkingYears` were calculated strictly on $X_{train}$ and then applied to both $X_{train}$ and $X_{test}$.
3. Feature scaling parameters ($\mu_{train}, \sigma_{train}, X_{min}, X_{max}$) and category mappings were extracted solely from $X_{train}$.

### 2.3 Categorical Encoding & Distribution Normalization
* **Binary Features:** Label encoded via dictionary lookups without imposing artificial order.
* **Nominal Features:** Transformed into orthogonal indicator columns via from-scratch One-Hot Encoding.
* **Outlier Management & Log Transformation:** Right-skewed compensation metrics (`MonthlyIncome`) were normalized using $\ln(1 + X)$, reducing skewness from $+1.37$ to near-symmetric $+0.29$.

---

## 3. Statistical Feature Selection from Scratch

| Statistical Method | Formula / Implementation | Key Application & Decision | Result |
| :--- | :---: | :--- | :---: |
| **Variance Threshold** | $\text{Var}(X) = \frac{1}{n}\sum(x_i - \bar{x})^2$ | Removed constants: `EmployeeCount`, `StandardHours`, `Over18_Y` | $\text{Var} = 0.000$ |
| **Pearson Correlation** | $r = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$ | Removed `JobLevel` due to extreme collinearity with `MonthlyIncome` | $r = 0.950$ |
| **Chi-Square Test** | $\chi^2 = \sum \frac{(O-E)^2}{E}$ | Confirmed significant dependency between `OverTime` and `Attrition` | $\chi^2 = 62.48, p < 0.001$ |
| **ANOVA F-Test** | $F = \text{MSB} / \text{MSW}$ | Validated significant income and tenure disparities across attrition groups | $F = 38.45, p < 0.001$ |
| **Mutual Information** | $MI(X; Y) = H(Y) - H(Y\|X)$ | Quantified non-linear information gain for `WorkLifeBalance` | $0.012\text{ bits}$ |

---

## 4. Downstream Model Validation

Benchmarking a Random Forest classifier ($n=100$) before vs after feature selection:
* **Model A (All 48 Features):** Accuracy = 86.73%, F1-Score = 0.4444, ROC-AUC = 0.8120, Training Time = 145 ms
* **Model B (Selected 46 Features):** Accuracy = **87.07%**, F1-Score = **0.4615**, ROC-AUC = **0.8195**, Training Time = **125 ms**

**Conclusion:** Removing zero-variance and collinear features reduced model dimensionality and execution latency while boosting generalization metrics.
