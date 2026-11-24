# Heart Disease Prediction

A machine learning project that predicts whether a patient is at risk of heart disease from their health measurements. It includes a deployed Streamlit app for instant predictions and a Power BI dashboard that explores the patterns in the data. It is built as a data science project, not as a medical tool.

## What it does

- **Predicts heart disease risk.** Takes a patient's details and returns whether potential heart disease risk is predicted.
- **Uses clinical measurements.** Inputs include age, sex, resting blood pressure, cholesterol, maximum heart rate, chest pain type, fasting blood sugar, resting ECG, exercise-induced angina, ST depression (oldpeak) and ST slope.
- **Compares models.** Logistic Regression, SVM and Decision Tree were trained and compared in the notebook. Logistic Regression is the deployed model.
- **Live app.** A Streamlit app applies the same encoding and scaling as training, then shows the prediction.
- **Dashboard.** A Power BI dashboard summarises how heart disease relates to age, sex, cholesterol and other factors.

## Keywords

Machine Learning · Healthcare Analytics · Classification · Predictive Modeling · Logistic Regression · SVM · Decision Tree · Feature Scaling · Streamlit · Power BI · Data Science · Python

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (label encoding, standard scaling, Logistic Regression, SVM, Decision Tree)
- [Streamlit](https://streamlit.io/) for the deployed app
- Power BI for the dashboard
- Jupyter Notebook for analysis and modeling

## Project Structure

| Folder | Purpose |
| :--- | :--- |
| `Heart_Disease_model and Data/` | Training notebook (Logistic Regression, SVM, Decision Tree) and the dataset |
| `Deployed_Heart_Disease_model/` | Streamlit app, trained Logistic Regression model, scaler and encoders |
| `Heart_Disease_Dashboard/` | Power BI dashboard (`.pbix`) and data |

## Author

**Amar Chaurasiya**, Data Scientist working as an AI Engineer. Open to full-time roles and freelance work.

- Email: [amarchaurasiya124@gmail.com](mailto:amarchaurasiya124@gmail.com)
- LinkedIn: [linkedin.com/in/amar-chaurasiya-163788262](https://www.linkedin.com/in/amar-chaurasiya-163788262)
- Portfolio: [amarchaurasiya.vercel.app](https://amarchaurasiya.vercel.app)
