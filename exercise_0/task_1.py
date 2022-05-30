import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# constants
PATH: str = "input/exercise_0/"

def load_data(filename: str) -> pd.DataFrame:
    """Loads the bikesharing data into a Dataframe"""
    csv = pd.read_csv(f"{filename}.csv", delimiter=",", decimal=".")
    return csv


def transform_day_table(table: pd.DataFrame) -> pd.DataFrame: 
    """Transforms the dataframe as specified in task 1 
    """
    table = table.drop(['instant', 'registered', 'casual', 'atemp'], axis=1)
    table['dteday'] = pd.to_datetime(table['dteday'])
    table['dteday'] = table['dteday'].dt.day_of_year

    return table
    

def plot_day_table(table: pd.DataFrame):
    """ Plots every column in the given dataframe with respect to the cnt collumn as a scatter plot"""
    col_names = list(table.drop('cnt', axis=1))
    for col in col_names:
        table.plot(x=col, y='cnt', kind='scatter')
        plt.show()


def train_rfr_model(train_set: pd.DataFrame) -> RandomForestRegressor: 
    """Trains and Returns an RandomForesRegressor"""
    model = RandomForestRegressor(n_estimators=100)
    
    x_train, y_train = split_set(train_set)
    model = model.fit(x_train, y_train)
    
    return model


def split_set(data_set: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Splits the given dataset into its x and y component"""
    col_names = list(data_set.drop('cnt', axis=1))
    x_data = data_set[col_names]
    y_data = data_set['cnt']
        
    return x_data, y_data


def rmse(prediction: pd.DataFrame, y_true: pd.DataFrame) -> float:
    """"Returns the root mean squared error of prediction and y_true"""
    return np.sqrt(mean_squared_error(y_true, prediction))


if __name__ == "__main__": 
    day_table = load_data(PATH + "day")
    day_table = transform_day_table(day_table)
 
    # plot_day_table(day_table)

    # I split the dataset in two different ways. 
    # The first time i am just cutting of 20 percent of the dataset at the end
    # the rmse of that model should be more true to unseen data, because new data 
    # is usually not inside the learned timeframe
    train_set, test_set = train_test_split(day_table, test_size=0.2, shuffle=False)

    test_x, test_y = split_set(test_set)
    model = train_rfr_model(train_set) 
    prediction = model.predict(test_x)
    print(rmse(prediction, test_y))

    # Random Split as a reference
    train_set, test_set = train_test_split(day_table, test_size=0.2, shuffle=True)

    test_x, test_y = split_set(test_set)
    model = train_rfr_model(train_set) 
    prediction = model.predict(test_x)
    print(rmse(prediction, test_y))
