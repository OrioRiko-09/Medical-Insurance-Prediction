# MediPredict 🏥
### Medical Insurance Cost Prediction using Machine Learning

MediPredict is a Machine Learning-based web application that estimates medical insurance charges based on personal information such as age, BMI, number of children, smoking habits, sex, and region.

The project uses a Random Forest Regression model to predict estimated insurance costs and provides an interactive interface built with Streamlit.

## ✨ Features

- Predicts estimated medical insurance charges.
- **Interactive What-if Simulator ("Explore What-if Scenarios")** to test how altering Age, BMI, or Smoking status impacts insurance charges with real-time delta and percentage change comparisons.
- Interactive web interface using Streamlit with split 2-column layout and smooth card pop-in animations.
- Data exploration and visualization using Pandas, Matplotlib and Seaborn.
- Data preprocessing and categorical feature encoding.
- Comparison of multiple regression algorithms.
- Trained Random Forest model saved using Joblib.
- Custom UI with responsive styling and animations.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning |
| Joblib | Model serialization |
| Streamlit | Web application |

## 📊 Dataset

The project uses the Medical Cost Personal Dataset (`insurance.csv`).

The dataset contains information about individuals and their medical insurance charges.

Features:

- `age` – Age of the individual.
- `sex` – Sex of the individual.
- `bmi` – Body Mass Index.
- `children` – Number of dependent children.
- `smoker` – Smoking status.
- `region` – Residential region.
- `charges` – Medical insurance charges (target variable).

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset, identify patterns, and examine relationships between features.

Some observations:

- The original dataset contains 1,338 records and 7 columns.
- One duplicate record was identified and removed.
- No missing values were found.
- Insurance charges showed a right-skewed distribution.
- Smoking status showed a strong difference in average insurance charges.
- Age and BMI showed positive correlations with insurance charges.

## 🤖 Machine Learning Models

Three regression algorithms were trained and evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

### Model Evaluation

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 4177.05 | 5956.34 | 0.8069 |
| Decision Tree | 2787.72 | 5904.67 | 0.8103 |
| Random Forest | 2657.61 | 4724.02 | 0.8786 |

**Best Performing Model: Random Forest Regressor**

The Random Forest model achieved an R² score of approximately 0.8786 on the test dataset.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/OrioRiko-09/Medical-Insurance-Prediction
```

### 2. Navigate to the project directory

```bash
cd Medical-Insurance-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows:

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

`http://localhost:8501`

## 📁 Project Structure

```text
Medical-Insurance-Prediction/
│
├── dataset/
├── notebooks/
├── src/
├── models/
│   └── random_forest_model.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 📌 Future Improvements

- Deploy the application online.
- Improve prediction accuracy through additional model experimentation.
- Add more visualizations and model insights.
- Improve input validation and user experience.
- Explore additional relevant features for insurance cost estimation.

## ⚠️ Disclaimer

MediPredict is an educational Machine Learning project. Its predictions are estimates based on patterns in the training dataset and should not be considered professional medical or financial advice.

---

⭐ If you find this project interesting, consider giving the repository a star!
