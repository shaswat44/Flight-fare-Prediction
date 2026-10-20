
# Flight Price Prediction

A machine learning project that estimates flight fares using a Random Forest regression model and a Streamlit web application.

## Features
- Flight fare prediction through an interactive UI
- Airline, source, destination, travel class, date and flight timing inputs
- Model evaluation and prediction testing scripts

## Tech Stack
- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit

## Model Evaluation
- MAE: ₹1,434.21
- RMSE: ₹3,127.39
- R² Score: 0.9811

These metrics are from the existing test evaluation and are not a guarantee of real-world fare accuracy.

## Run Locally
1. Install the required Python packages.
2. Ensure the trained model is available at `models/flight_price_model.pkl`.
3. Run:

```bash
streamlit run app.py
```

## Dataset
The project uses `Freshly_cleaned.csv`. Check the dataset's license and redistribution permissions before publishing it.

## Note
Predictions are estimates based on the training data and are not live airline prices.
