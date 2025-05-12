# aerospace-data\_analytics

Propeller Performance Prediction Using Machine Learning

Project Overview:

This project aims to develop a machine learning pipeline for predicting the performance of aircraft propellers using data from the UIUC Propeller Database. We leveraged data preprocessing, feature engineering, hyperparameter tuning, and model deployment to create an efficient, real-time prediction API.

Data Sources:

* Experimental Data: Measurements of propeller performance (e.g., thrust, power, efficiency) across different test conditions.
* Geometric Data: Propeller blade dimensions and physical characteristics.
 Steps Involved:

1. Data Loading and Preprocessing:

   * Loaded experimental and geometric data from multiple volumes.
   * Merged these datasets based on common keys.
   * Cleaned and standardized the data to remove duplicates and handle missing values.

2. Feature Engineering:

   * Created physics-based features like blade loading, power coefficient, and log-transformed coefficients to capture key aerodynamic relationships.
   * Added derived features like solidity and advance ratio squared for better predictive power.

3. Exploratory Data Analysis (EDA):

   * Conducted scatter plots, correlation heatmaps, and pairplots to understand relationships between features and target variables.
   * Used statistical summaries to assess the quality and distribution of the data.

4. Model Training and Optimization:

   * Employed Gradient Boosting Regressor (GBR) for robust and interpretable predictions.
   * Performed hyperparameter tuning using Optuna to find the optimal model configuration.
   * Saved the trained model pipeline for deployment.

5. Deployment Preparation:

   * Developed a FastAPI application for real-time model predictions.
   * Implemented data validation using Pydantic to ensure input consistency.
   * Created structured endpoints for seamless integration with other systems.

6. Testing and Debugging:

   * Resolved common issues like file path mismatches, missing dependencies, and data type conflicts to ensure a smooth deployment.
Key Outcomes:

* Accurate Predictions: Achieved high model accuracy on test data, with optimized hyperparameters.
* Reusable Pipeline: Created a flexible pipeline that can be easily adapted to other propeller datasets.
* Real-time Inference: Built a FastAPI backend for efficient, scalable predictions.

Potential Extensions:

* Integration with a front-end application for real-time performance visualization.
* Deployment to a cloud environment for scalability.
* Additional model tuning to improve efficiency and reduce latency.
