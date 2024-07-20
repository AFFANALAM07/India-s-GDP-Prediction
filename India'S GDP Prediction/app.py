from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("India's_GDP.csv")


# Prepare the data
X = df['Year'].values.reshape(-1, 1)
y = df['GDP'].values.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        year_input = int(request.form['year'])
        if year_input <= 2025:
            return render_template('index.html', prediction_text="Please enter a year after 2025.")
        year = np.array([[year_input]])
        prediction = lin_reg.predict(year)
        return render_template('index.html', prediction_text=f"Predicted GDP for the year {year_input}: ${prediction[0][0]:,.2f} billion")
    except ValueError:
        return render_template('index.html', prediction_text="Invalid input. Please enter a valid year.")

if __name__ == "__main__":
    app.run(debug=True)
