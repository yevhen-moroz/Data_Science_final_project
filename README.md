# Telecom Churn Prediction

This project predicts the likelihood of customer churn in a telecommunications company based on historical data related to subscriptions, billing, contracts, service quality, and internet usage.
## Technologies Used

### Programming & Data Processing

* Python
* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Logistic Regression
* Decision Tree
* Random Forest

### Data Visualization

* Matplotlib
* Seaborn

### Web Application

* Streamlit

### Model Serialization

* Joblib

### Development & Deployment

* Jupyter Notebook
* Docker

### Version Control

* Git
* GitHub

## Project Structure

```text
telecom_churn_app/
│
├── data/
│   ├── raw/
│   │   └── internet_service_churn.csv
│   └── processed/
│       └── cleaned_churn_data.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── train_model.py
│   ├── predict.py
│   └── app.py
│
├── models/
│   ├── churn_model.pkl
│   └── metrics.json
│
├── assets/
│   ├── churn_distribution.png
│   ├── correlation_heatmap.png
│   ├── subscription_age_by_churn.png
│   ├── bill_avg_by_churn.png
│   └── eda_summary.txt
│
├── requirements.txt
├── Dockerfile
└── README.md
```

## Project Workflow

The project follows the pipeline below:

```text
notebooks/EDA.ipynb
        ↓
data/processed/cleaned_churn_data.csv
        ↓
src/train_model.py
        ↓
models/churn_model.pkl
        ↓
src/app.py
```

## What EDA.ipynb Does

After running the notebook, a cleaned dataset is created at `data/processed/cleaned_churn_data.csv`, which is then used for model training.

The notebook performs:

* loading the raw dataset from `data/raw/internet_service_churn.csv`;
* exploring the dataset structure;
* checking for missing values;
* analyzing the distribution of the target variable `churn`;
* generating visualizations;
* performing correlation analysis;
* cleaning the dataset;
* saving the processed dataset to `data/processed/cleaned_churn_data.csv`.

### Data Cleaning Rules

* the `id` column is removed;
* missing values in `reamining_contract` are filled with `0`;
* missing values in `download_avg` are filled with the median value;
* missing values in `upload_avg` are filled with the median value.

## Installing Dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Run the EDA Notebook

Open the file:

```text
notebooks/EDA.ipynb
```

and execute all cells. After completion, the following file should be created:

```text
data/processed/cleaned_churn_data.csv
```

### 2. Train the Model

```bash
python src/train_model.py
```

After training, the following files will be created or updated:

```text
models/churn_model.pkl
models/metrics.json
```

### 3. Run the Streamlit Application

```bash
streamlit run src/app.py
```

After launching, a web interface will open where you can enter information about a new customer and receive a churn prediction:

```text
High / Medium / Low Churn Risk
```

## Docker

Build the Docker image:

```bash
docker build -t telecom-churn-app .
```

Run the container:

```bash
docker run -p 8501:8501 telecom-churn-app
```

After startup, the application will be available in your browser at:

```text
http://localhost:8501
```

## Models

Several machine learning models are compared in `src/train_model.py`:

* Logistic Regression;
* Decision Tree;
* Random Forest.

The best-performing model is selected based on the F1-score and saved as `models/churn_model.pkl`.

## Evaluation Metrics

The following metrics are used to evaluate model performance:

* Accuracy;
* Precision;
* Recall;
* F1-score;
* Classification Report;
* Confusion Matrix.

For customer churn prediction, **Recall** and **F1-score** are particularly important because companies need to identify as many customers at risk of leaving as possible while maintaining a reasonable balance between false positives and false negatives.
