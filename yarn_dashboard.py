
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")
st.title("Yarn Processing Plant - Failure & Downtime Prediction")

# Load your dataset
df = pd.read_csv("Yarn_Processing_Dummy_Dataset.csv")  # CSV must be in same folder

# Define features and target
features = ["Temperature", "Humidity", "Pressure", "Vibration", "Noise", "Speed", "Torque", "Energy"]
X = df[features]
y = df[["Failure Probability", "Downtime (Hours)"]]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sidebar inputs
st.sidebar.header("Input Sensor Readings")
input_data = []
for f in features:
    input_data.append(
        st.sidebar.slider(
            f, float(df[f].min()), float(df[f].max()), float(df[f].mean())
        )
    )

# Predict button
if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([input_data], columns=features)
    prediction = model.predict(input_df)[0]
    
    st.subheader("Prediction Results")
    st.metric("Failure Probability", f"{prediction[0]:.2f}")
    st.metric("Downtime (Hours)", f"{prediction[1]:.2f}")
    
    # Plot
    fig, ax = plt.subplots()
    ax.bar(["Failure Probability", "Downtime (Hours)"], prediction, color=["tomato", "cornflowerblue"])
    st.pyplot(fig)

# Show raw data
st.subheader("Sample Dataset")
st.dataframe(df.head(10))
