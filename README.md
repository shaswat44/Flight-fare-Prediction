# ✈️ Flight Price Prediction

A machine learning project that predicts estimated flight ticket prices using a **Random Forest Regression model** and an interactive **Streamlit web application**.

## Overview

This project uses flight-related features to estimate ticket fares. Users can enter flight details through a web interface and receive a predicted price based on the trained model.

**Note:** Predictions are estimates based on the available training data. They are not live airline prices.

## Live Demo

Try the application here:
https://flight-fare-prediction-qb3btfdwzcnxafen8wttqj.streamlit.app/

## Features

- Interactive Streamlit web application
- Flight fare prediction based on airline and route
- Travel class selection (Economy or Business)
- Departure and arrival time inputs
- Journey duration and number of stops
- Model evaluation and baseline comparison
- Saved model loading for predictions

## Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, Random Forest Regressor
- **Web Application:** Streamlit
- **Model Serialization:** Joblib
- **Notebook:** Jupyter Notebook

## Model Performance

The existing test evaluation reported the following metrics:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | ₹1,434.21 |
| Root Mean Squared Error (RMSE) | ₹3,127.39 |
| R² Score | 0.9811 |

These results reflect performance on the evaluated test dataset. They do not guarantee the same accuracy for unseen real-world fares.

## Project Structure

```text
Flight-Price-Prediction/
├── app.py
├── data/
│   └── Freshly_cleaned.csv
├── models/
│   └── flight_price_model.pkl
├── notebook/
│   └── 01_data_understanding.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

The `data/` and `models/` folders are excluded from Git tracking. Make sure the required dataset and trained model are available locally before running the project.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/shaswat44/Flight-fare-Prediction.git
cd Flight-fare-Prediction
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add required local files

Place the cleaned dataset at:

`data/Freshly_cleaned.csv`

Place the trained model at:

`models/flight_price_model.pkl`

These files are not included in the Git repository.

### 5. Start the application

```bash
streamlit run app.py
```

## Dataset

The project uses a cleaned flight fare dataset containing flight details such as airline, source, destination, travel class, timing, journey duration, stops, and ticket price.

Check the dataset's license and redistribution permissions before sharing it publicly.

## Limitations

- Predictions depend on the training dataset and its coverage.
- The model does not fetch live airline prices.
- Results may differ for routes, dates, or fare conditions not adequately represented in the training data.

## Future Improvements

- Deploy the application online
- Add more recent flight data
- Improve model validation on unseen data
- Add model comparison and feature importance visualizations

## Author

**Shaswat Bhatnagar**

GitHub: [@shaswat44](https://github.com/shaswat44)
