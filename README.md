# Employee Turnover Prediction & Talent Retention Intelligence

Project 7 - HR Analytics. Predicts the probability that an employee resigns.

- Models: Linear Regression (linear probability model), Logistic Regression, XGBoost
- Data: built-in SYNTHETIC demo data, or upload your own CSV/Excel file in the app
- App: Streamlit dashboard (upload page + 5 dashboard pages + data/model-card page)

## Files (all in the top level of the repository)
- `app.py` - Streamlit app
- `core.py` - cleaning, features, models, metrics
- `requirements.txt` - libraries
- `sample_employee_data.csv` - synthetic file to practise uploading

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Using your own data
Open the app (it opens on the upload page), upload a CSV/Excel file, match your columns to the fields
(the app auto-guesses common names such as the IBM HR Attrition dataset), pick which value means
"left", and click "Validate & train models". Do not include gender, age, race, religion or marital status.

## Ethics
Scores support retention only. Never use them to penalize employees or automate adverse decisions.
Individual scores must not be shared with line managers.
