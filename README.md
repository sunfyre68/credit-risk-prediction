# Credit Risk Prediction

Machine-learning experiments for predicting personal-loan default risk from demographic, employment, financial, and credit-history attributes. This CSE 422 project compares Logistic Regression, K-Nearest Neighbors (KNN), a feed-forward Neural Network, and K-Means clustering.

**Author:** Enan Mahmud  
**Institution:** BRAC University, Department of Computer Science and Engineering  
**Student ID:** 24101544

## Project overview

The target is `loan_status`:

- `0`: no default
- `1`: default

The dataset is imbalanced (approximately 77.8% no-default and 22.2% default), so the supervised experiments emphasize minority-class precision, recall, F1, and ROC-AUC rather than accuracy alone. The best reported model is the Neural Network, with an F1 score of **0.7652** and ROC-AUC of **0.9672**.

## Problem statement

Credit providers need to identify applicants who may default while avoiding unnecessary rejection of reliable borrowers. This project studies whether historical applicant and loan attributes can support that risk assessment, and compares interpretable, distance-based, neural, and unsupervised approaches.

## Dataset

`14.csv` contains 45,000 rows and 14 columns: 13 input features and the binary target `loan_status`. The file is included for academic use with this project.

| Feature | Type | Description |
| --- | --- | --- |
| `person_age` | numeric | Applicant age |
| `person_gender` | categorical | Applicant gender |
| `person_education` | categorical | Education level |
| `person_income` | numeric | Annual income |
| `person_emp_exp` | numeric | Employment experience |
| `person_home_ownership` | categorical | Home-ownership category |
| `loan_amnt` | numeric | Requested loan amount |
| `loan_intent` | categorical | Loan purpose |
| `loan_int_rate` | numeric | Interest rate |
| `loan_percent_income` | numeric | Loan-to-income proportion |
| `cb_person_cred_hist_length` | numeric | Credit-history length |
| `credit_score` | numeric | Credit score |
| `previous_loan_defaults_on_file` | categorical | Previous default indicator |
| `loan_status` | binary target | Default status |

The raw CSV has no missing values. The notebook removes duplicate rows and invalid logical records (including impossible ages and employment histories), leaving 44,990 rows for modeling.

## Methodology

1. Explore numeric distributions, categorical counts, correlations, class balance, outliers, and invalid values.
2. Apply `log1p` to `person_income`.
3. One-hot encode the categorical variables.
4. Remove `person_emp_exp` and `cb_person_cred_hist_length` because of their strong correlation with `person_age`.
5. Split the cleaned data with an 80/20 stratified train/test split (35,992/8,998 samples).
6. Fit `StandardScaler` on the training data only, then transform both partitions.
7. Train Logistic Regression with balanced class weights, KNN selected by five-fold F1 grid search, and a class-weighted Neural Network.
8. Run K-Means without using `loan_status`; the reported silhouette score peaks at `k = 5`.

The Neural Network uses dense layers of 32 and 16 ReLU units, dropout, and a one-unit sigmoid output. It is trained with Adam, binary cross-entropy, class weights, and 20 epochs.

## Results

Reported test-set metrics from the notebook and lab report:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8541 | 0.6138 | **0.9265** | 0.7384 | 0.9550 |
| KNN (`k = 9`) | **0.9028** | **0.8315** | 0.7055 | 0.7633 | 0.9457 |
| Neural Network | 0.8751 | 0.6571 | 0.9160 | **0.7652** | **0.9672** |

The notebook contains the original visualizations: target balance, histograms, boxplots, density plots, categorical distributions, a correlation heatmap, confusion matrices, Neural Network learning curves, K-Means elbow/silhouette plots, a two-feature cluster projection, and ROC curves. They are intentionally kept in the notebook so the figures remain tied to the executed analysis rather than being regenerated independently.

## Run the notebook

### Google Colab

1. Open [`credit_risk_prediction.ipynb`](credit_risk_prediction.ipynb) in Google Colab.
2. Upload `14.csv` to the Colab runtime, or place it in the mounted Drive location expected by the notebook.
3. If using a local upload, replace the Drive-based `pd.read_csv(...)` path with `/content/14.csv`.
4. Run the cells from top to bottom. TensorFlow, scikit-learn, pandas, NumPy, seaborn, and Matplotlib are available in the standard Colab environment.

### Local Jupyter

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install jupyterlab numpy pandas matplotlib seaborn scikit-learn tensorflow
jupyter lab
```

Open `credit_risk_prediction.ipynb`, ensure `14.csv` is in the same directory, and change the Drive-specific CSV path to:

```python
df = pd.read_csv("14.csv")
```

The Neural Network and K-Means cells are computationally heavier than the exploratory cells. Results can vary slightly with library versions and hardware.

## Limitations

- This is an academic benchmark, not a production underwriting system.
- The data may contain unrealistic values and unknown sampling or collection bias; cleaning rules are based on project assumptions.
- The positive class is imbalanced, and threshold selection is not optimized beyond the notebook's default 0.5 Neural Network threshold.
- The reported metrics come from one stratified holdout split; repeated cross-validation and an external validation set would provide stronger evidence.
- Demographic and financial attributes can encode bias. Any real deployment would require fairness, privacy, explainability, governance, and domain review.
- K-Means is exploratory and does not use the target; its clusters should not be interpreted as default predictions.

## Repository structure

```text
.
├── 14.csv
├── CSE422_Lab_Report.pdf
├── README.md
└── credit_risk_prediction.ipynb
```

## Academic Fairness Notice

This repository is intended for academic learning and may help readers understand how to approach a similar project. Please study, adapt, and implement the work independently rather than directly copying or submitting it.
