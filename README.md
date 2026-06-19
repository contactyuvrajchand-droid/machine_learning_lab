# ~linear Regression on big Data

# Problem Statement
<!--
The objective of this project was to develop and evaluate a scalable machine learning solution for predicting diabetes disease progression using patient clinical measurements. Traditional machine learning approaches may become inefficient when handling large datasets due to computational and memory constraints. Therefore, Apache Spark was utilized to build a distributed data processing and machine learning pipeline capable of efficiently managing data preprocessing, model training, and evaluation.

 The project aimed to:

 Load and analyze a diabetes dataset containing patient demographic and medical attributes.
 Perform data preprocessing, including categorical feature encoding and feature scaling.
 Train multiple regression models to predict the disease progression score.
 Compare the performance of Linear Regression, Lasso Regression, and Ridge Regression.
 Monitor system resource utilization (CPU and memory) during the execution process.
 Demonstrate the effectiveness of Spark ML pipelines for scalable predictive analytics.


# Solution / Methodology

 1. Environment Setup
Installed and configured PySpark.
Created a SparkSession with custom resource configurations:
.4 GB executor memory
.2 GB driver memory
.2 executor cores
Enabled Spark UI monitoring and tracked CPU and memory usage using the psutil library.

2. Data Loading and Exploration
Mounted Google Drive and loaded the diabetes dataset from a CSV file into a Spark DataFrame.
Conducted exploratory analysis by:
.Displaying sample records.
.Counting total records and partitions.
.Examining the dataset schema.
.Generating descriptive statistics.
.Checking for missing or null values.


3. Data Preprocessing

A preprocessing pipeline was developed to prepare the data for machine learning:
.Categorical Encoding: Converted the categorical variable sex into a numerical feature using StringIndexer.
.Feature Engineering: Combined predictor variables (age, gender, bmi, bp, tc, ldl, hdl, tch, ltg, glu) into a single feature vector using VectorAssembler.
.Feature Scaling: Applied StandardScaler to normalize feature values and improve model performance.

4. Model Development

Three regression models were implemented and compared:
Linear Regression
.Built a baseline regression model using standardized features.
.Estimated coefficients and intercept values to understand feature influence.
Lasso Regression (L1 Regularization)
.Applied L1 regularization (elasticNetParam = 1).
.Reduced the impact of less important features by shrinking some coefficients toward zero.
.Helped address potential overfitting and feature selection.
Ridge Regression (L2 Regularization)
.Applied L2 regularization (elasticNetParam = 0).
.Reduced model complexity by penalizing large coefficient values.
.Improved model generalization while retaining all features.

5. Pipeline Construction
   
A Spark ML Pipeline was created to automate the workflow:
.String Indexing
.Feature Assembly
.Standard Scaling
.Regression Model Training
The pipeline was also saved and loaded to demonstrate model persistence and reusability.

6. Model Training and Testing

   
Split the dataset into:
.70% Training Data
.30% Testing Data
Trained all three regression models on the training dataset.
Generated predictions on unseen test data.


7. Model Evaluation

 The models were evaluated using three standard regression metrics:
.Mean Squared Error (MSE): Measures average squared prediction error.
.Root Mean Squared Error (RMSE): Indicates prediction error in the original target units.
.R² Score: Measures the proportion of variance explained by the model.
The evaluation results were used to compare the predictive performance of Linear Regression, Lasso Regression, and Ridge Regression.

8. Visualization

A bar chart was generated using Matplotlib to visually compare:
.MSE
.RMSE
.R² Score
across all three regression models, allowing for easier identification of the best-performing approach.
-->
