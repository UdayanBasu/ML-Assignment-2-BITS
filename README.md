# Machine Learning Assignment 2
## Online Shopper Purchase Intention Classification

## a. Problem Statement

The aim of this project is to predict whether an online shopping session will end in a purchase. Five machine-learning classification algorithms are implemented on the same dataset and compared using six evaluation metrics. A Streamlit application allows test-data upload, model selection, metric display, confusion-matrix viewing, classification-report viewing, and comparison of all models.

## b. Dataset Description

**Dataset:** Online Shoppers Purchasing Intention Dataset  
**Repository:** UCI Machine Learning Repository  
**Task:** Binary classification  
**Total instances:** 12,330  
**Feature size:** 17 input features  
**Target:** Revenue  
**Target classes:** FALSE = no purchase, TRUE = purchase  
**Missing values:** None reported by UCI

The dataset represents online-shopping sessions using page-visit information, durations, analytics measurements, visitor information, month, device/browser-related categories and other session attributes.

This dataset satisfies the assignment minimum of 500 instances and 12 features.

### Features

1. Administrative
2. Administrative_Duration
3. Informational
4. Informational_Duration
5. ProductRelated
6. ProductRelated_Duration
7. BounceRates
8. ExitRates
9. PageValues
10. SpecialDay
11. Month
12. OperatingSystems
13. Browser
14. Region
15. TrafficType
16. VisitorType
17. Weekend

**Target:** Revenue

## c. GitHub Repository Link

`<ADD_YOUR_GITHUB_REPOSITORY_LINK_HERE>`

## Live Streamlit App Link

`<ADD_YOUR_STREAMLIT_APP_LINK_HERE>`

## d. Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest Classifier

Categorical variables are one-hot encoded. Numeric variables are standardized for Logistic Regression, kNN and Gaussian Naive Bayes.

## Evaluation Metrics

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## Comparison Table

Run:

```powershell
python train_models.py
```

and copy the values printed by your BITS Virtual Lab execution into the table below.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| kNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest | | | | | | |

## Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Provides a useful linear baseline and produces probability estimates for purchase intention after preprocessing. |
| Decision Tree | Provides an interpretable nonlinear model but can be more sensitive to the training sample than ensemble methods. |
| kNN | Makes predictions using nearby sessions in the transformed feature space and benefits from numeric scaling. |
| Naive Bayes | Provides a fast probabilistic baseline, although the conditional-independence assumption may not hold for all browsing variables. |
| Random Forest | Combines many trees and is generally more robust than a single Decision Tree, especially for nonlinear relationships. |
| Overall Winner | Determine from the metric table generated in your BITS Virtual Lab run. |

## Streamlit Features

- CSV test-data upload
- Model-selection dropdown
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report
- Comparison table
- Automatic overall-winner display

## Repository Structure

```text
ml_assignment_2_online_shoppers/
├── app.py
├── data_utils.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── logistic_regression.py
    ├── decision_tree.py
    ├── knn.py
    ├── naive_bayes.py
    └── random_forest.py
```

## How to Run

```powershell
python -m pip install -r requirements.txt
python train_models.py
python -m streamlit run app.py
```

Upload the included `test_data.csv` in the Streamlit application.

## Important

The complete UCI dataset is fetched through `ucimlrepo` when the training script or Streamlit app starts, so an internet connection is required the first time the project runs.

Before submission:
- Add your own GitHub repository link.
- Add your deployed Streamlit Community Cloud link.
- Run the assignment in the BITS Virtual Lab.
- Add the required BITS Virtual Lab screenshot to the final PDF.
- Update the metric table using your actual execution.
- Customize the UI and written observations in your own words.
