# Artificial Intelligence Portfolio - Weeks 3 to 9

This repository contains my practical coursework for **CN7023 2526 (T3) Artificial Intelligence (OC)**. The projects cover supervised learning, unsupervised learning, text classification, manual algorithm implementation, distributed data processing, and time-series forecasting using PySpark and Python.

**Student:** Yuvraj Chand  
**Student ID:** 3294044

## Portfolio Overview

| Week | Project | Main Methods | Key Result |
|---|---|---|---|
| 3 | House Price Prediction | Linear Regression, Lasso, Ridge | All models achieved approximately 0.994 R-squared; Ridge was marginally best |
| 4 | Titanic Survival Prediction | Weighted Logistic Regression, feature binning, Chi-square selection | 49.74% test accuracy and 0.4974 AUROC on the synthetic dataset |
| 5 | Customer Segmentation Classification | Multinomial Logistic Regression, Decision Tree | Both models achieved 100% test accuracy on the synthetic dataset |
| 6 | Clustering on Multiple Datasets | K-means, silhouette analysis | K = 3 was best for both datasets; Customer data achieved a 0.762 silhouette score |
| 7 | Spam Classification with Naive Bayes | Multinomial, Bernoulli, and Gaussian Naive Bayes | Multinomial NB was selected with 78.57% accuracy and 0.4000 spam F1-score |
| 8 | Manual Perceptron Spam Classification | PySpark preprocessing, NumPy Perceptron, pocket method | Pocket model achieved 85.46% accuracy and 0.6667 spam F1-score |
| 9 | Bitcoin Price Forecasting | Time-series features, Linear Regression, GBT, naive benchmark | Linear Regression achieved 0.9940 R-squared; the naive benchmark had the lowest RMSE |

## Week 3 - House Price Prediction

This project builds a distributed regression pipeline for predicting property prices from **Square Footage, Number of Bedrooms, Number of Bathrooms, Year Built, and Lot Size**.

- Loaded and validated a property dataset containing **1,000,000 records**.
- Checked the schema, descriptive statistics, missing values, and system resource usage.
- Combined the five predictors with `VectorAssembler` and standardised them with `StandardScaler`.
- Used a reproducible **70:30 train-test split**.
- Trained and compared standard Linear Regression, Lasso Regression, and Ridge Regression.
- Evaluated the models using MSE, RMSE, and R-squared.

All three models explained approximately **99.4% of the test-set price variation**. Ridge produced the best numerical score, although its advantage over standard Linear Regression and Lasso was negligible. The models achieved an RMSE of approximately **19,946.56**.

## Week 4 - Titanic Survival Prediction

This project implements a complete binary-classification pipeline for predicting survival in a **synthetic Titanic dataset**.

- Cleaned and validated approximately **1,000,000 passenger records**.
- Converted continuous Age and Fare values into categorical bins using `Bucketizer`.
- Encoded categorical passenger attributes and assembled the modelling features.
- Applied `ChiSqSelector` to the training data to select seven predictors without test-set leakage.
- Addressed class balance with a weighted Logistic Regression model.
- Evaluated the model with accuracy, AUROC, precision, recall, F1-score, a precision-recall curve, and a confusion matrix.

The model achieved **49.74% test accuracy**, **0.4974 AUROC**, **0.4980 precision**, **0.4767 recall**, and a **0.4871 F1-score**. These results indicate that the supplied synthetic predictors contained very little useful survival signal, while still demonstrating a complete and reproducible PySpark classification workflow.

## Week 5 - Customer Segmentation Classification

This project applies multiclass classification to assign customers to **Low Value, Medium Value, or High Value** segments.

- Processed a customer dataset containing **2,000,000 records**.
- Analysed the class distribution and excluded identifier columns from the predictors.
- Used Age, Annual Income, Spending Score, Years as a Customer, and Number of Purchases as features.
- Converted text labels to numerical classes with `StringIndexer`.
- Trained Multinomial Logistic Regression and Decision Tree classifiers using the same **70:30 split**.
- Compared accuracy, weighted F1-score, predictions, and confusion matrices.

Both models achieved **100% test accuracy**, while Multinomial Logistic Regression also achieved a weighted **F1-score of 1.000**. Because the data are synthetic and the segment labels may be closely determined by the predictor variables, the perfect results should be validated on independent real-world data.

## Week 6 - K-Means Clustering on Multiple Datasets

This project compares an unsupervised K-means workflow across two datasets.

### Grocery Dataset

- **440 records**.
- Clustering features: Fresh, Milk, and Grocery spending.
- Strong positive correlation of **0.77** between Milk and Grocery spending.
- Best tested configuration: **K = 3**, with a silhouette score of **0.658**.

### Customer Dataset

- **8,950 records**.
- Clustering features: Purchases Frequency, One-off Purchases Frequency, and Purchases Installments Frequency.
- Best tested configuration: **K = 3**, with a silhouette score of **0.762**.

The workflow includes exploratory analysis, correlation analysis, feature assembly, train-test splitting, silhouette-based K comparison, cluster counts, and 2D/3D visualisation. The Customer dataset formed more clearly separated clusters than the Grocery dataset.

## Week 7 - Spam Classification with Naive Bayes

This project compares three Naive Bayes algorithms for classifying email as **ham or spam**.

- Created reusable PySpark text-processing pipelines with tokenisation, stop-word removal, TF-IDF features, and label indexing.
- Trained Multinomial, Bernoulli, and Gaussian Naive Bayes models on the same split.
- Added binary word-presence features for Bernoulli Naive Bayes.
- Compared accuracy and spam-class precision, recall, F1-score, confusion matrices, and misclassified examples.

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1 |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 0.7857 | 0.5000 | 0.3333 | 0.4000 |
| Gaussian Naive Bayes | 0.5714 | 0.2857 | 0.6667 | 0.4000 |
| Bernoulli Naive Bayes | 0.7857 | 0.0000 | 0.0000 | 0.0000 |

Multinomial Naive Bayes was selected because it matched the highest spam F1-score while providing better accuracy and precision. A limitation is that only **69 cleaned records** remained from the uploaded file, and the cleaned message values appeared unusually short and numeric. The results should therefore be treated as a coursework demonstration rather than a production estimate.

## Week 8 - Manual Perceptron Spam Classification

This project combines distributed preprocessing in PySpark with a **Perceptron implemented manually in NumPy**.

- Cleaned and prepared **5,695 emails**, including 4,327 ham and 1,368 spam messages.
- Extracted **57 Spambase-style features**: 48 word-frequency, six character-frequency, and three capital-run measures.
- Split the data before scaling to prevent information leakage.
- Converted labels to `-1` for ham and `+1` for spam.
- Implemented the Perceptron update rule, bias updates, learning-curve tracking, and the pocket method.
- Analysed influential features, confusion matrices, classification metrics, and misclassified emails.

| Parameter Set | Accuracy | Spam Precision | Spam Recall | Spam F1 |
|---|---:|---:|---:|---:|
| Pocket model | 0.8546 | 0.6933 | 0.6420 | 0.6667 |
| Final-epoch model | 0.8639 | 0.7118 | 0.6708 | 0.6907 |

The model ran for **100 epochs** and did not reach zero training error, indicating that the numerical features were noisy or not perfectly linearly separable. The pocket parameters were used for the main error analysis because they represented the epoch with the fewest training errors.

## Week 9 - Bitcoin Price Forecasting

This project develops a distributed and leakage-aware forecasting pipeline for predicting the **next day's Bitcoin closing price**.

- Cleaned **7,679,564 one-minute OHLCV observations**.
- Aggregated the data into **5,334 daily records** covering 1 January 2012 to 8 August 2026.
- Created **23 lag, return, rolling-statistical, volume-history, and calendar features** using PySpark window functions.
- Used a chronological split of **4,255 training rows** and **1,064 future testing rows**.
- Compared Linear Regression, Gradient-Boosted Trees, and a naive current-close forecast.
- Evaluated forecasts using RMSE, MAE, R-squared, ordered prediction evidence, and forecast plots.

| Model | RMSE | MAE | R-squared |
|---|---:|---:|---:|
| Naive current-close benchmark | 1,855.7199 | 1,299.7194 | 0.9942 |
| Linear Regression | 1,884.7096 | 1,326.7075 | 0.9940 |
| Gradient-Boosted Trees | 28,858.8478 | 21,167.1809 | -0.4124 |

Linear Regression was the strongest trained model, but the naive benchmark achieved the lowest overall error. This result highlights the importance of chronological evaluation, realistic baselines, and caution when modelling a non-stationary financial time series.

## Technologies Used

- Python
- PySpark and Spark ML
- NumPy and pandas
- Matplotlib and Seaborn
- Google Colab
- Google Drive
- Jupyter notebooks

## Skills Demonstrated

- Distributed data loading, cleaning, and transformation
- Spark ML pipelines and reusable preprocessing
- Regression, binary classification, multiclass classification, and clustering
- Text feature extraction and imbalanced-class evaluation
- Manual implementation of a machine-learning algorithm
- Leakage-free feature engineering and model evaluation
- Time-aware splitting and financial forecasting
- Metric comparison, visualisation, error analysis, and reflective evaluation

## Running the Coursework

1. Open the required notebook in Google Colab or another Jupyter environment.
2. Install PySpark and the other required Python packages.
3. Mount Google Drive if the dataset is stored there.
4. Update the dataset path to match its location.
5. Run the notebook cells in order from top to bottom.

The results documented above are based on the submitted coursework runs. Exact outputs may vary if the datasets, random seeds, package versions, or computing environment are changed.

## Author

**Yuvraj Chand**  
Artificial Intelligence Portfolio, CN7023
