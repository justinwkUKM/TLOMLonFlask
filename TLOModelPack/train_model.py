import pandas as pd
import numpy as np
from holistics import HolisticsAPI
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import pickle


def read_data():
    
    """Download report 87273 --- from Package Assigned By Date  ----- LAZADA DASHBOARD ON HOLISTICS"""

    obj = HolisticsAPI(api_key = 'TN66YjRr7FeABTLBO6Dh4Gz9jbPjzTLqZKIAgArHLsM=', url = 'https://secure.holistics.io')
    # my_dataframe = obj.export_data(report_id='146265')
    # print(my_dataframe.shape)
    # my_dataframe.to_csv('downloaded_data.csv') 
    df3 = pd.read_csv('downloaded_data.csv', index_col ='date', parse_dates=True)
    df3 = df3.rename(columns = {"count(id)": "Orders", 
                              "sum(case when package_status='delivered' then 1 else 0 end)":"Deliveries"})
    print(df3.index.max() - df3.index.min())
    df3.loc[((df3['Orders'] > 1600)), 'Sales_Time'] = 1
    df3.loc[((df3['Orders'] > 3500)), 'Sales_Time'] = 2
    df3.loc[((df3['Orders'] > 5000)), 'Sales_Time'] = 3
    df3['Sales_Time'] = df3['Sales_Time'].fillna(0)
    print(df3.columns)
    adf_test(df3['Orders'])
    
    train_data = df3.loc[:'2020-03-10']
    train_data.tail()
   
    #SHOULD MAKE THIS DYNAMIC
    test_data = df3.loc['2020-03-11':'2020-03-24']
    print('testshape ', test_data.shape)
    print(test_data.head())
    
    model = ExponentialSmoothing(train_data['Orders'], trend='add' , seasonal='add', seasonal_periods=7)
    fitted_model = model.fit()
    test_predictions = fitted_model.forecast(14)
    
    print('ROOT MEAN SQUARE',np.sqrt(mean_squared_error(test_data['Orders'], test_predictions)))
    final_model = ExponentialSmoothing(df3['Orders'], trend='add', seasonal='add', seasonal_periods=7).fit()
    forecast_predictions = final_model.forecast(7)
    print(forecast_predictions)
    
    auto_ar = auto_arima(df3['Orders'], exogenous=df3[['Sales_Time']], seasonal=True, m=7)
    seasonal_order = auto_ar.get_params()['seasonal_order']
    order = auto_ar.get_params()['order']
    model = SARIMAX(train_data['Orders'], exog=train_data[['Sales_Time']], order=order,seasonal_order=seasonal_order)
    resultant_sarimax_model = model.fit()
    start = len(train_data)
    end = len(train_data) + len(test_data) - 1
    
    predictions_sarimax = resultant_sarimax_model.predict(start,end, exog = test_data[['Sales_Time']], typ='levels').rename("SARIMA Predictions")

    print('ROOT MEAN SQUARE',np.sqrt(mean_squared_error(test_data['Orders'], predictions_sarimax)))

    final_model = SARIMAX(df3['Orders'], exog=df3[['Sales_Time']], order=order,seasonal_order=seasonal_order)
    result= final_model.fit()
    
        # save the model to disk
    filename = 'resources/models/finalized_timeseries_model2.sav'
    pickle.dump(result, open(filename, 'wb'))

    # # some time later...

    # # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string()) 
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")