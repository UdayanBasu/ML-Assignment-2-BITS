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

`https://github.com/UdayanBasu/ML-Assignment-2-BITS`

## Live Streamlit App Link

`https://ml-assignment-2-bits-nnbyugcnn7mz9aacj27caa.streamlit.app/`

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

The following results were obtained from the model-comparison output generated for the provided test data.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9620 | 0.9675 | 0.0000 | 0.0000 | 0.0000 | -0.0182 |
| Decision Tree | 0.9620 | 0.9740 | 0.0000 | 0.0000 | 0.0000 | -0.0182 |
| kNN | 0.9620 | 0.9481 | 0.0000 | 0.0000 | 0.0000 | -0.0182 |
| Naive Bayes | 0.9747 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Random Forest | 0.9747 | 0.9935 | 0.5000 | 1.0000 | 0.6667 | 0.6979 |
## Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved an accuracy of 0.9620 and AUC of 0.9675. However, Precision, Recall and F1 for the positive purchase class are all 0.0000, showing that it did not correctly identify purchase cases in this test sample. |
| Decision Tree | Achieved 0.9620 accuracy and a strong AUC of 0.9740, but its positive-class Precision, Recall and F1 are 0.0000. Its MCC is slightly negative (-0.0182), indicating weak classification agreement on this test sample. |
| kNN | Achieved 0.9620 accuracy and AUC of 0.9481. Like Logistic Regression and Decision Tree, it failed to identify positive purchase cases, resulting in 0.0000 Precision, Recall and F1. |
| Naive Bayes | Produced 0.9747 accuracy, but its AUC is 0.5000 and its positive-class Precision, Recall and F1 are 0.0000. The high accuracy therefore does not reflect useful discrimination between the two classes in this test sample. |
| Random Forest | Performed best overall. It achieved 0.9747 accuracy, the highest AUC of 0.9935, Precision of 0.5000, Recall of 1.0000, F1 of 0.6667 and MCC of 0.6979. It was the only model in this comparison that successfully detected the positive purchase cases. |
| Overall Winner | **Random Forest** is the overall winner for this test data because it has the strongest combination of AUC, Recall, F1 and MCC while maintaining high accuracy. |

