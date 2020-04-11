from flask import Flask, render_template, request, jsonify, redirect
from TLOModelPack import timeseries, train_model
import pandas as pd
import test
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import date
from datetime import timedelta


app = Flask(__name__)

"""Creating a route and allowing for testing the api"""
@app.route('/api/home', methods=['POST'])
def index_api():
    data = request.get_json()
    print(type(data))
    day = data.get('day')

    model = timeseries.load_model()
    future_sales_list_14 = [0, 0, 0, 1, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0]
    days = pd.date_range('2020-03-23', '2020-04-05', freq='D')
    df_forecast14 = pd.DataFrame(
        data=future_sales_list_14, index=days, columns=['Sales_Time'])
    forecast_sarimax_exog = model.predict(
        447, 447+13, exog=df_forecast14[['Sales_Time']], typ='levels').rename("SARIMAX ForeCasts")
    prediction = str(round(forecast_sarimax_exog.iloc[int(day)], 0))

    response_dict = {
        'prediction': prediction
    }

    r = jsonify(response_dict)
    return r


"""Creating a route and allowing for both get and post requests"""
@app.route('/', methods=['GET', 'POST'])
def index():
    # Checking if the request is a post request
    if request.method == 'POST':
        # getting data from the form
        data = request.form.get('text')
        print(data)
        return render_template('index.html', name=data)
    else:
        return render_template('index.html', name='')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Checking if the request is a post request
    if request.method == 'POST':
        # getting data from the form
        data = 0
        if int(request.form.get('text')) in range(1, 10):
            try:
                data = int(request.form.get('text'))
            except ValueError:
                pass
        else:
            data = 0

        model = timeseries.load_model()
        future_sales_list_7 = [0, 0, 0, 1, 3, 2, 0]
        today = date.today()
        print("Today's date:", today)
        forecast_end_date = today + timedelta(days=6)
        days = pd.date_range(today, forecast_end_date, freq='D')
        df_forecast7 = pd.DataFrame(
            data=future_sales_list_7, index=days, columns=['Sales_Time'])

        if data != 0:
            forecast_sarimax_exog = model.predict(
                447, 447+6, exog=df_forecast7[['Sales_Time']], typ='levels').rename("SARIMAX ForeCasts")
            return render_template('predict.html', name=str(round(forecast_sarimax_exog.iloc[data-1], 2)))
        else:
            return redirect('/')
    else:
        return render_template('predict.html', name='')


@app.route('/train', methods=['GET', 'POST'])
def train():

    if request.method == 'POST':
        print('POSTPOSTPOST')
        button = request.form.get('mybutton')
        
        if "button_simple_model" in request.form:
            print("Simple")
            model, rmse = train_model.make_simple_model()
            string_rmse = f'The RMSE for the Simple Model is {str(int(rmse))}'
       
        if "button_sarima_model" in request.form:
            print("Sarima")
            model, rmse = train_model.make_sarimax_model()
            string_rmse = f'The RMSE for the SarimaX Model is {str(int(rmse))}'

    
        return render_template('train.html', data=string_rmse)
    
    else:
        print('GETGETGET')
        return render_template('train.html', data='')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
