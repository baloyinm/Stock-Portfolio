## imports
from sqlalchemy import create_engine
import pandas as pd
import yfinance as yf
import json
import requests
from bs4 import BeautifulSoup




############################
## Yahoo Finance Part
###########################

import json

# According to the DB the following stock types exist
# 1    "Stock"
# 2    "World Index"
# 3    "ETF"
# 4    "Crypto"

# Offset is the number that determines when to move to the next page on a table for web scraping
offset = 0

Stock_url = ""
World_index_url = "https://finance.yahoo.com/world-indices"
ETF_url = f"https://finance.yahoo.com/etfs/?offset={offset}&count=25"
Crypto_url = f"https://finance.yahoo.com/crypto/?count=100&offset={offset}"

Url_list = {
    "Stock": Stock_url,
    "World Index": World_index_url,
    "ETF": ETF_url,
    "Crypto": Crypto_url
}

# Create JSON string for Scraping data list
Scraping_data_list = json.dumps([
    {
        "id": 1,
        "name": "Stock",
        "base_url": Url_list["Stock"],
        "url_type": "single_table"
    },
    {
        "id": 2,
        "name": "World Index",
        "base_url": Url_list["World Index"],
        "url_type": "single_table"
    },
    {
        "id": 3,
        "name": "ETF",
        "base_url": Url_list["ETF"],
        "url_type": "multi_table"
    },
    {
        "id": 4,
        "name": "Crypto",
        "base_url": Url_list["Crypto"],
        "url_type": "multi_table"
    }
])

# Parse the JSON string into a Python list of dictionaries
data = json.loads(Scraping_data_list)

# Iterate through the list
for item in data:
    print("Item:")
    # Iterate through the keys and values of each dictionary
    for key, value in item.items():
        print(f"{key}: {value}")
    print()  # Empty line for better readability between items


# Code for creating type number  in given dataframe


def type_creator(df,data_type):
    if data_type == "Stock":
        data_type = 1
    elif data_type == "World Index":
        data_type = 2
    elif data_type == "ETF":
        data_type = 3
    elif data_type == "Crypto":
        data_type = 4
    else:
        data_type = 5

    df['type'] = data_type

    return df






import pandas as pd
import requests
from bs4 import BeautifulSoup

import pandas as pd
import requests
from bs4 import BeautifulSoup
# function to scrape data for single table thing
def scrape_single_table(url,data_type):
      response = requests.get(url,data_type)


      # According to the DB the following stock types exist
# 1    "Stock"
# 2    "World Index"
# 3    "ETF"
# 4    "Crypto"
# 5    "Unclassified"

#Creating the type id for the stock on creation

# This can be commented out for now as another function has replaced it 
 
      if  data_type== "Stock":
        data_type =1
      elif  data_type== "World Index":
        data_type =2

      elif  data_type== "ETF":
        data_type =3

      elif data_type== "Crypto":
        data_type =4
      else: data_type= 5


      soup = BeautifulSoup(response.text, 'html.parser')
      tables = soup.find_all('table')
      if tables:
          # Access the first element of the list and convert it to a DataFrame
          df = pd.DataFrame(pd.read_html(str(tables[0]))[0])


            #Copying first 5 coulumns then dropping the ones with empty data

          df= df.iloc[:, :5].copy()
          df=df.dropna()




          # Changing the column names
          # Assuming df is your DataFrame
          new_column_names = ['symbol', 'name','intraday_price' ,'change', 'percent_change' ]
          df.columns = new_column_names

          # Remove '+' and '%' signs and convert to numeric
          df['percent_change'] = pd.to_numeric(df['percent_change'].str.replace('[+,%]', '', regex=True))
          df["type"]=data_type
          #Adding   a type coloumn
          df.reset_index(drop=True, inplace=True)
              # Adding import_date column with current datetime with GMT+2
          gmt_time = datetime.utcnow() + timedelta(hours=2)
          df['import_date'] = gmt_time

          #df=df.dropna
          return df
      else:
          print("No tables found on the webpage.")
          return None

# Test the function
url = Url_list["World Index"]
df = scrape_single_table(Url_list["World Index"], "World Index")




############################
## Other crypto data


import pandas as pd
import requests
from bs4 import BeautifulSoup






##urls =['https://finance.yahoo.com/world-indices/']
##url2 =  ['https://finance.yahoo.com/etfs/?offset=25&count=25']   # etfs

##https://finance.yahoo.com/commodities/



import pandas as pd
import requests
from bs4 import BeautifulSoup

def scrape_multi_table(base_url, data_type, num_tables=50, step=100):
    """
    Scrape multiple tables from Yahoo Finance pages based on the data type.

    Parameters:
    - base_url (str): The base URL for the Yahoo Finance page.
    - data_type (str): The type of data to scrape ('Stock', 'World Index', 'ETF', 'Crypto').
    - num_tables (int): Number of tables to scrape (default is 20).
    - step (int): Step for offset (default is 100).

    Returns:
    - final_df (DataFrame): Concatenated DataFrame of all tables scraped.
    """
    if data_type == "Stock":
        data_type = 1
    elif data_type == "World Index":
        data_type = 2
    elif data_type == "ETF":
        data_type = 3
    elif data_type == "Crypto":
        data_type = 4
    else:
        data_type = 5

    # List to hold DataFrames for all tables
    all_combined_dfs = []

    # Iterate through tables with different offsets
    for i in range(num_tables):
        offset = i * step
        url = f"{base_url}{offset}"
        print("Scraping table", i+1, "with offset", offset)
        table_df = scrape_table(url)
        all_combined_dfs.append(table_df)

    # Concatenate DataFrames for all tables
    final_df = pd.concat(all_combined_dfs, ignore_index=True)
    return final_df


def scrape_table(url):
    """
    Scrape table data from a Yahoo Finance page.

    Parameters:
    - url (str): The URL of the Yahoo Finance page.

    Returns:
    - combined_df (DataFrame): Combined DataFrame of all tables scraped.
    """
    # Send a GET request to the URL
    response = requests.get(url)
    # Parse the HTML content of the webpage
    soup = BeautifulSoup(response.text, 'html.parser')
    # Find all tables on the webpage
    tables = soup.find_all('table')
    # List to hold DataFrame for each table
    all_dfs = []

    # Iterate through each table
    for table in tables:
        # Convert the table to a pandas DataFrame
        df = pd.read_html(str(table))[0]  # Assuming there's only one table per page
        all_dfs.append(df)
    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

# Example usage
base_url = 'https://finance.yahoo.com/crypto/?count=25&offset='
final_df = scrape_multi_table(base_url, "Crypto")
final_df



# Cleaning the crypto data
import pandas as pd
from datetime import datetime, timedelta

def clean_and_format_data(final_df):
    """
    Clean up and format the data from the scraped DataFrame.

    Parameters:
    - final_df (DataFrame): DataFrame containing the scraped data.

    Returns:
    - new_df (DataFrame): Cleaned and formatted DataFrame.
    """
    # Copying the first 10 columns of the DataFrame
    new_df = final_df.iloc[:, :10].copy()

    # Deleting one of the volume columns
    new_df = new_df.drop(['Volume in Currency (Since 0:00 UTC)', 'Volume in Currency (24Hr)'], axis=1)

    # Adding a type column to identify it as crypto (type_id = 4)
    new_df['type_id'] = 4

    # Changing the column names
    new_column_names = ['symbol', 'name', 'intraday_price', 'change', 'percent_change',
                        'market_cap', 'volume', 'circulating_supply', 'type_id']
    new_df.columns = new_column_names

    # Remove '+' and '%' signs and convert to numeric
    new_df['percent_change'] = pd.to_numeric(new_df['percent_change'].str.replace('[+,%]', '', regex=True))

    # Function to convert string values with B, M, or T suffix to numeric values
    def convert_to_numeric(value_str):
        suffixes = {'B': 1e9, 'M': 1e6, 'T': 1e12}
        for suffix, factor in suffixes.items():
            if value_str.endswith(suffix):
                return float(value_str[:-1]) * factor
        return float(value_str)  # If no suffix, return the original value as float

    # Apply the conversion function to the 'Volume' column
    new_df['volume'] = new_df['volume'].apply(convert_to_numeric)
    new_df['circulating_supply'] = new_df['circulating_supply'].apply(convert_to_numeric)
    new_df['market_cap'] = new_df['market_cap'].apply(convert_to_numeric)
    
    # Adding import_date column with current datetime with GMT+2
    gmt_time = datetime.utcnow() + timedelta(hours=2)
    new_df['import_date'] = gmt_time
    
    return new_df

# Example usage:
new_df = clean_and_format_data(final_df)
new_df
























## Halving dates 
##Mining rigs
#mining profitability
###########################


############################
## Data science things
###########################

import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, RidgeCV, LassoCV, ElasticNetCV, BayesianRidgeCV, RANSACRegressor, HuberRegressor, PassiveAggressiveRegressor, TweedieRegressor, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, VotingRegressor, StackingRegressor, IsolationForest, HistGradientBoostingRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern
from prophet import Prophet
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from keras.layers import LSTM, GRU
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.isotonic import IsotonicRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor

class StockPricePredictor:
    def __init__(self, models):
        self.models = models
        self.predictions = {}

    def train(self, X, y):
        for model_name, model in self.models.items():
            model.fit(X, y)

    def predict(self, X):
        for model_name, model in self.models.items():
            self.predictions[model_name] = model.predict(X)

    def get_predictions_df(self, stock_id):
        predictions_df = pd.DataFrame(self.predictions)
        predictions_df['Stock_ID'] = stock_id
        predictions_df['Date'] = datetime.now().strftime("%Y-%m-%d")
        return predictions_df

    def predict_with_ids(self, X, stock_id):
        predictions = {}
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict(X)
        predictions_df = pd.DataFrame(predictions)
        predictions_df['Stock_ID'] = stock_id
        predictions_df['Date'] = datetime.now().strftime("%Y-%m-%d")
        return predictions_df

class MultiFeatureStockPricePredictor(StockPricePredictor):
    def __init__(self, models):
        super().__init__(models)

# Fetch data from Yahoo Finance
def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data['Close'].reset_index()

# Sample usage for single feature
symbol = 'AAPL'
start_date = '2023-01-01'
end_date = '2023-12-31'
data = fetch_data(symbol, start_date, end_date)
X_single = data[['Close']].values[:-1]
y_single = data['Close'].values[1:]

# Linker models for single feature
linker_models_single = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Bayesian Ridge': BayesianRidge(),
    'RidgeCV': RidgeCV(),
    'LassoCV': LassoCV(),
    'ElasticNetCV': ElasticNetCV(),
    'Bayesian Ridge CV': BayesianRidgeCV(),
    'SVR_linear': SVR(kernel='linear'),
    'SVR_poly': SVR(kernel='poly'),
    'SVR_sigmoid': SVR(kernel='sigmoid'),
    'KNN': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'RANSAC': RANSACRegressor(),
    'Huber': HuberRegressor(),
    'Passive Aggressive': PassiveAggressiveRegressor(),
    'Tweedie': TweedieRegressor(),
    'SGD': SGDRegressor(),
    'NuSVR': NuSVR(),
    'Linear SVR': LinearSVR(),
}

# Ensemble models for single feature
ensemble_models_single = {
    'Random Forest': RandomForestRegressor(),
    'Extra Trees': ExtraTreesRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'HistGradientBoosting': HistGradientBoostingRegressor(),
    'Bagging': BaggingRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'Voting': VotingRegressor(estimators=[('lr', LinearRegression()), ('rf', RandomForestRegressor()), ('gb', GradientBoostingRegressor())]),
    'Stacking': StackingRegressor(estimators=[('lr', LinearRegression()), ('rf', RandomForestRegressor()), ('gb', GradientBoostingRegressor())]),
    'IsolationForest': IsolationForest(),
}

# Other models for single feature
other_models_single = {
    'ARIMA': ARIMA(),
    'Prophet': Prophet(),
    'XGBoost': xgb.XGBRegressor(),
    'LightGBM': LGBMRegressor(),
    'CatBoost': CatBoostRegressor(verbose=0),
    'MLP': MLPRegressor(),
    'PCA_LR': make_pipeline(PCA(n_components=1), LinearRegression()),
    'CNN': Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_single.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ]),
    'Gaussian Process_RBF': GaussianProcessRegressor(kernel=RBF()),
    'Gaussian Process_DotProduct': GaussianProcessRegressor(kernel=DotProduct()),
    'Gaussian Process_Matern': GaussianProcessRegressor(kernel=Matern()),
    'RadiusNeighbors': RadiusNeighborsRegressor(),
    'ExtraTree': ExtraTreeRegressor(),
    'LSTM': Sequential([
        LSTM(50, activation='relu', input_shape=(X_single.shape[1], 1)),
        Dense(1)
    ]),
    'GRU': Sequential([
        GRU(50, activation='relu', input_shape=(X_single.shape[1], 1)),
        Dense(1)
    ]),
    'SVR_rbf': SVR(kernel='rbf'),
    'SVR_poly_degree_3': SVR(kernel='poly', degree=3),
    'SVR_poly_degree_4': SVR(kernel='poly', degree=4),
    'SVR_poly_degree_5': SVR(kernel='poly', degree=5),
    'Kernel Ridge': KernelRidge(),
    'Linear SVR': LinearSVR(),
    'NuSVR': NuSVR(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Extra Trees': ExtraTreesRegressor(),
    'Bagging': BaggingRegressor(),
    'HistGradientBoosting': HistGradientBoostingRegressor(),
    'Passive Aggressive': PassiveAggressiveRegressor(),
    'Tweedie': TweedieRegressor(),
    'SGD': SGDRegressor(),
    'Bayesian Ridge': BayesianRidge(),
    'Huber': HuberRegressor(),
    'RANSAC': RANSACRegressor(),
    'KNN': KNeighborsRegressor(),
    'Radius Neighbors': RadiusNeighborsRegressor(),
    'Isotonic Regression': IsotonicRegression(),
    'PLS': PLSRegression(),
    'Voting': VotingRegressor(estimators=[('lr', LinearRegression()), ('rf', RandomForestRegressor()), ('gb', GradientBoostingRegressor())]),
    'Stacking': StackingRegressor(estimators=[('lr', LinearRegression()), ('rf', RandomForestRegressor()), ('gb', GradientBoostingRegressor())]),
    'Dummy': DummyRegressor(),
}

# Combine all models for single feature
models_single = {**linker_models_single, **ensemble_models_single, **other_models_single}

# Create single feature predictor
single_feature_predictor = StockPricePredictor(models_single)
single_feature_predictor.train(X_single, y_single)

# Predict with specific stock ID for single feature
stock_id = 'AAPL'
predictions_df_single = single_feature_predictor.predict_with_ids([[data['Close'].iloc[-1]]], stock_id)
print("Single Feature Predictions:\n", predictions_df_single)

# Sample usage for multi feature
X_multi = data[['Close', 'Open', 'High', 'Low', 'Volume']].values[:-1]  # Using multiple features
y_multi = data['Close'].values[1:]

# Models that can handle multiple features
multi_feature_models = {
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'HistGradientBoosting': HistGradientBoostingRegressor(),
    'Extra Trees': ExtraTreesRegressor(),
    'Bagging': BaggingRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'LightGBM': LGBMRegressor(),
    'CatBoost': CatBoostRegressor(verbose=0),
    'MLP': MLPRegressor(),
    'LSTM': Sequential([
        LSTM(50, activation='relu', input_shape=(X_multi.shape[1], 1)),
        Dense(1)
    ]),
    'GRU': Sequential([
        GRU(50, activation='relu', input_shape=(X_multi.shape[1], 1)),
        Dense(1)
    ]),
    'SVR_rbf': SVR(kernel='rbf'),
    'SVR_poly_degree_3': SVR(kernel='poly', degree=3),
    'SVR_poly_degree_4': SVR(kernel='poly', degree=4),
    'SVR_poly_degree_5': SVR(kernel='poly', degree=5),
    'Kernel Ridge': KernelRidge(),
    'Linear SVR': LinearSVR(),
    'NuSVR': NuSVR(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Extra Trees': ExtraTreesRegressor(),
    'Bagging': BaggingRegressor(),
    'HistGradientBoosting': HistGradientBoostingRegressor(),
    'Passive Aggressive': PassiveAggressiveRegressor(),
    'Tweedie': TweedieRegressor(),
    'SGD': SGDRegressor(),
    'Bayesian Ridge': BayesianRidge(),
    'Huber': HuberRegressor(),
    'RANSAC': RANSACRegressor(),
    'KNN': KNeighborsRegressor(),
    'Radius Neighbors': RadiusNeighborsRegressor(),
    'Isotonic Regression': IsotonicRegression(),
    'PLS': PLSRegression(),
    'Voting': VotingRegressor(estimators=[('lr', LinearRegression()), ('rf', RandomForestRegressor()), ('gb', GradientBoostingRegressor())]),
    'Stacking': StackingRegressor(estimators=[('lr', LinearRegression()), ('rf', RandomForestRegressor()), ('gb', GradientBoostingRegressor())]),
    'Dummy': DummyRegressor(),
}

# Create multi feature predictor
multi_feature_predictor = MultiFeatureStockPricePredictor(multi_feature_models)
multi_feature_predictor.train(X_multi, y_multi)






# Select Stock and amount then show growth ober time 
#  Select model then predict with percentage  of confidence
#select percent then give one stock select stocks 
# select  risk  then stock
# Select risk then stocks  and allocation 
# Select stocks then optimize portfolio
    # Minimize risk 
    # Maximise profit
#Select 



############################
### Database connections and operations
############################


############################
## Data Engineering
###########################

# historical imports -> All historical data until the current day
    # Historical predictions (connect to data science )
    # 1 year's worth of data then daily predictions  
    # All data then daily predictions 
    # All data then monthly predictions on a monthly bassis  (before the predictions month)
# Daily predictions  all data 
# Daily predictions -> historical data 



############################
## API development
###########################