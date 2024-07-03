import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Automated Model Training and Evaluation')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    le = LabelEncoder()
    df['BmiClass'] = le.fit_transform(df['BmiClass'])

    st.subheader('Initial dataset')
    st.write(df)

    st.subheader('Set Parameters')
    split_ratio = st.slider('Data split ratio (% for Training Set)', min_value=10, max_value=90, value=80)

    X = df.drop(columns=['Bmi'])
    y = df['Bmi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-split_ratio)/100, random_state=42)

    st.subheader('Train split')
    st.write(X_train)
    st.write(y_train)

    st.subheader('Test split')
    st.write(X_test)
    st.write(y_test)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    st.subheader('Feature importance')
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    st.bar_chart(feature_importance.set_index('feature'))

    st.subheader('Prediction results')
    predictions = pd.DataFrame({
        'actual': y_train.append(y_test),
        'predicted': list(y_train_pred) + list(y_test_pred),
        'class': ['train'] * len(y_train) + ['test'] * len(y_test)
    }).reset_index(drop=True)
    st.write(predictions)

    st.subheader('Prediction results graph')
    fig, ax = plt.subplots()
    sns.scatterplot(x='actual', y='predicted', hue='class', data=predictions, ax=ax)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    st.pyplot(fig)

    # Additional feature to predict BMI category
    st.subheader('Predict BMI Category')
    age = st.number_input('Age', min_value=0, max_value=120, value=25)
    height = st.number_input('Height (in meters)', min_value=0.5, max_value=2.5, value=1.75)
    weight = st.number_input('Weight (in kg)', min_value=10, max_value=300, value=70)

    input_data = pd.DataFrame([[age, height, weight]], columns=['Age', 'Height', 'Weight'])
    
    X_class = df[['Age', 'Height', 'Weight']]
    y_class = df['BmiClass']
    model_class = RandomForestClassifier()
    model_class.fit(X_class, y_class)

    if st.button('Predict BMI Category'):
        bmi_category_encoded = model_class.predict(input_data)
        bmi_category = le.inverse_transform(bmi_category_encoded)
        st.write(f'The predicted BMI category is: {bmi_category[0]}')
