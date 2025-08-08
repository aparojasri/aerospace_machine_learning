

# Propeller Performance Prediction Using Machine Learning

### Project Overview

This project presents a complete, end-to-end machine learning pipeline for predicting the aerodynamic performance of small-scale aircraft propellers. Traditional methods like wind-tunnel testing or CFD are expensive and time-consuming. This project builds a fast and accurate **data-driven surrogate model** using the canonical UIUC Propeller Database.

The pipeline integrates classical aerodynamic principles with modern machine learning techniques to forecast key performance metrics—**Thrust Coefficient (Cₜ), Power Coefficient (Cₚ), and Efficiency (η)**—directly from a propeller's geometry and operating conditions.

The final, trained model is deployed via a **REST API using FastAPI and Docker**, making it a scalable, real-time tool ready for integration into engineering design and optimization workflows.

---
### Key Features & Outcomes

* **High Predictive Accuracy:** Achieved an **R² > 0.98** and **MAE < 0.005** on the held-out test data using an Optuna-tuned Gradient Boosting Regressor. The model's low bias and uniform error were validated with Predicted vs. Actual and Residual plots.
* **Physics-Informed Feature Engineering:** Demonstrated the importance of domain knowledge by creating high-impact features like **Advance Ratio (J)**, **Solidity (σ)**, and **Blade Loading**, which were proven to be the most influential predictors through SHAP analysis.
* **Reproducible ML Pipeline:** The entire workflow—from data cleaning and feature scaling to polynomial expansion and modeling—is encapsulated in a single, professional `scikit-learn` pipeline, ensuring full reproducibility.
* **Scalable Deployment:** The trained model is serialized with `joblib` and served via a **FastAPI** application with Pydantic data validation. The entire application is containerized with **Docker** for portable, scalable, real-time inference.
* **In-depth Exploratory Data Analysis (EDA):** The project includes a comprehensive analysis of the UIUC dataset, using histograms, scatter plots, pairplots, and correlation heatmaps to uncover key aerodynamic trends and guide the modeling process.

---
### Technologies & Libraries Used

* **Programming Language:** Python 3.9+
* **Data Science Stack:** pandas, NumPy, scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Hyperparameter Tuning:** Optuna
* **Model Interpretation:** SHAP
* **API & Deployment:** FastAPI, Docker, joblib

---
### Future Work

* **Frontend Integration:** Develop a web-based front-end application (e.g., using Streamlit or React) to provide a user-friendly interface for real-time performance visualization.
* **Cloud Deployment:** Deploy the containerized application to a cloud service like AWS, Google Cloud, or Heroku for enhanced scalability and accessibility.
* **Advanced Modeling:** Explore more advanced models or ensemble techniques to further improve prediction accuracy and reduce inference latency.
