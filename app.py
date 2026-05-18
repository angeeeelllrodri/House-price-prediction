from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model only
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:

        OverallQual = float(request.form['OverallQual'])
        GrLivArea = float(request.form['GrLivArea'])
        GarageCars = float(request.form['GarageCars'])
        GarageArea = float(request.form['GarageArea'])
        TotalBsmtSF = float(request.form['TotalBsmtSF'])
        FirstFlrSF = float(request.form['FirstFlrSF'])
        FullBath = float(request.form['FullBath'])
        TotRmsAbvGrd = float(request.form['TotRmsAbvGrd'])
        YearBuilt = float(request.form['YearBuilt'])

        # Create dataframe
        input_data = pd.DataFrame([[
            OverallQual,
            GrLivArea,
            GarageCars,
            GarageArea,
            TotalBsmtSF,
            FirstFlrSF,
            FullBath,
            TotRmsAbvGrd,
            YearBuilt
        ]], columns=[
            'OverallQual',
            'GrLivArea',
            'GarageCars',
            'GarageArea',
            'TotalBsmtSF',
            '1stFlrSF',
            'FullBath',
            'TotRmsAbvGrd',
            'YearBuilt'
        ])

        prediction = model.predict(input_data)[0]

        return render_template(
            'result.html',
            prediction=round(prediction, 2)
        )

    except Exception as e:
        return f"ERROR : {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)