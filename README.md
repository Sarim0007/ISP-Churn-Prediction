# Internet Service Provider (ISP) Customer Churn Prediction

<img width="1060" height="861" alt="image" src="https://github.com/user-attachments/assets/92f55846-e942-4276-8a59-c01d35d38ca9" />

This project is an end-to-end data science application that predicts customer churn for an Internet Service Provider. It involves data cleaning, exploratory data analysis (EDA), model building, and deployment of the final model as an interactive web application using Streamlit.

---

##  Live Demo

**You can access the live web application here:** [Streamlit App URL](https://your-streamlit-app-url.streamlit.app/) <!-- Replace with your actual Streamlit Cloud URL -->

---

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [1. Data Cleaning & Preprocessing](#1-data-cleaning--preprocessing)
  - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [3. Feature Selection](#3-feature-selection)
  - [4. Model Building & Evaluation](#4-model-building--evaluation)
  - [5. Model Serialization](#5-model-serialization)
- [Technologies Used](#technologies-used)
- [Setup and Deployment](#setup-and-deployment)
  - [Prerequisites](#prerequisites)
  - [Running Locally](#running-locally)
  - [Deployment on Streamlit Cloud](#deployment-on-streamlit-cloud)
- [License](#license)

---

## Project Overview
The primary goal of this project is to build a machine learning model that can accurately predict whether a customer is likely to churn (cancel their subscription). By identifying customers at risk, an ISP can proactively offer incentives to retain them. The final output is a user-friendly web app where one can input customer details and get an instant churn prediction.

## Project Structure

customer_churn_project/
│
├── dataset/
│   └── internet_service_churn.csv      # The raw dataset
│
├── models/
│   ├── model.pkl                       # Serialized best-performing machine learning model
│   └── scaler.pkl                      # Serialized scaler object for data preprocessing
│
├── customer_churn.ipynb                # Jupyter Notebook with EDA and model building steps
├── app.py                              # The main Streamlit application script
└── requirements.txt                    # Python dependencies for deployment


---

## Methodology

### 1. Data Cleaning & Preprocessing
- **Handled Missing Values:**
  - `reamining_contract`: Filled missing values using the **median**.
  - `download_avg` & `upload_avg`: Filled missing values using the **mean**.
- **Dropped Irrelevant Columns:** The `id` column was removed as it does not contribute to the predictive power of the model.

### 2. Exploratory Data Analysis (EDA)
- Analyzed the distribution of churned vs. non-churned customers.
- Investigated relationships between features like `subscription_age`, `bill_avg`, and `churn`.
- Visualized correlations between all numerical features using a heatmap to identify multicollinearity and key relationships.

### 3. Feature Selection
- Used the `ExtraTreesClassifier` to determine the importance of each feature in predicting churn.
- Based on the results, the following less important features were dropped to improve model performance and reduce complexity:
  - `service_failure_count`
  - `download_over_limit`

### 4. Model Building & Evaluation
Several classification models were trained and evaluated to find the best performer:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- **Random Forest Classifier** (Best Performer)
- Support Vector Classifier (SVC)
- Gaussian Naive Bayes

The **Random Forest Classifier** was selected as the final model due to its superior performance, achieving an **accuracy of 93.45%**.

### 5. Model Serialization
- The trained `RandomForestClassifier` model was saved as `model.pkl`.
- The `StandardScaler` object, which was fitted on the training data, was saved as `scaler.pkl`.
- Using these serialized files (`.pkl`) allows the Streamlit app to make predictions on new data without needing to retrain the model every time it runs.

---

## Technologies Used
- **Python:** Core programming language
- **Pandas:** Data manipulation and analysis
- **NumPy:** Numerical operations
- **Scikit-learn:** Machine learning (model building, preprocessing, evaluation)
- **Matplotlib & Seaborn:** Data visualization
- **Jupyter Notebook:** For EDA and model experimentation
- **Streamlit:** For building and deploying the interactive web application
- **Pickle:** For model serialization

---

## Setup and Deployment

### Prerequisites
- Python 3.8+
- `pip` for package installation

### Running Locally
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/customer_churn_project.git](https://github.com/your-username/customer_churn_project.git)
    cd customer_churn_project
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
4.  **Run the training script (only needed once to generate the .pkl files):**
    Make sure `internet_service_churn.csv` is in the `dataset/` folder.
    ```bash
    python model_training_script.py 
    ```

5.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser.

### Deployment on Streamlit Cloud
This app is ready for deployment on Streamlit Cloud. Simply connect your GitHub repository to your Streamlit account. The `requirements.txt` file will ensure that all necessary libraries are automatically installed on the server.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
