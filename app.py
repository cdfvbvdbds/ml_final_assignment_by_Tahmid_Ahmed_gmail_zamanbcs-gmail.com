import pandas as pd
import gradio as gr
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
url = "https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv"
df = pd.read_csv(url)
df = df.drop(columns=['Loan_ID'])
df['Dependents'] = df['Dependents'].replace('3+', '3')
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']
numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Total_Income']
categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))])
model.fit(X, y)

def predict_loan(gender, married, dependents, education, self_employed, 
                 app_income, coapp_income, loan_amount, term, credit_history, property_area):

    # Re-create the dataframe feature 'Total_Income'
    total_income = app_income + coapp_income
    data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [app_income],
        'CoapplicantIncome': [coapp_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [term],
        'Credit_History': [float(credit_history)],
        'Property_Area': [property_area],
        'Total_Income': [total_income]
    })

    prediction = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    result = "Approved" if prediction == 1 else "Rejected"
    return f"{result} (Probability: {prob:.2%})"
iface = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Dropdown(["Yes", "No"], label="Married"),
        gr.Dropdown(["0", "1", "2", "3"], label="Dependents"),
        gr.Dropdown(["Graduate", "Not Graduate"], label="Education"),
        gr.Dropdown(["Yes", "No"], label="Self Employed"),
        gr.Number(label="Applicant Income", value=5000),
        gr.Number(label="Coapplicant Income", value=0),
        gr.Number(label="Loan Amount", value=100),
        gr.Number(label="Loan Amount Term", value=360),
        gr.Radio(["1.0", "0.0"], label="Credit History (1.0 = Good)"),
        gr.Dropdown(["Urban", "Semiurban", "Rural"], label="Property Area")
    ],
    outputs="text",
    title="Loan Approval Prediction System"
)

iface.launch()
