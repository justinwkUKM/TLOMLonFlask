from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle

def intro():
    return('Intro to Timeseries')

def load_model():
    filename = 'resources/models/finalized_timeseries_model.sav'
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    print(loaded_model)
    return  loaded_model