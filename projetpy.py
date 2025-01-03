import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

from data_fetcher import DataFetcher

def best_corr(data_nasdaq, data_local, n):

    data_local['date'] = pd.to_datetime(data_local['date'])
    data_nasdaq['date'] = pd.to_datetime(data_nasdaq['date'])

    #Ajout du prix annuel moyen Shifter de -&
    data_local['year'] = data_local['date'].dt.year
    
    annual_coffee_avg = data_local.groupby('year')['coffee_nearby'].mean()
    annual_coffee_avg_next = annual_coffee_avg.shift(-1)
    data_nasdaq['year'] = data_nasdaq['date'].dt.year
    data_nasdaq['next_year_coffee_avg'] = data_nasdaq['year'].map(annual_coffee_avg_next)
    data_nasdaq.drop(columns=['year'], inplace=True)

    #Matrice de correlation
    data_nasdaq_corr = data_nasdaq.drop('date', axis=1)
    correlation_matrix = data_nasdaq_corr.corr()

    coffee_correlations = correlation_matrix['next_year_coffee_avg'].drop('next_year_coffee_avg')
    most_correlated = coffee_correlations.abs().sort_values(ascending=False)

    top_correlations = most_correlated.head(n)

    print("Les paramètres les plus corrélés avec le prix du café :")
    print(top_correlations)
        
    return top_correlations.index.tolist()


def nomr_and_plot(data_nasdaq):
    columns_to_normalize = data_nasdaq.columns[1:]

    scaler = MinMaxScaler()

    data_nasdaq[columns_to_normalize] = scaler.fit_transform(data_nasdaq[columns_to_normalize])

    plt.figure(figsize=(12, 6))
    plt.plot(data_nasdaq['date'], data_nasdaq['next_year_coffee_avg'], label='Prix du café', color='brown', linewidth=2)

    for indice in data_nasdaq.columns[2:]:
        plt.plot(data_nasdaq['date'], data_nasdaq[indice], label=indice, linestyle=':')


    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Valeur', fontsize=12)
    plt.title('Prix du café et indices économiques', fontsize=14)
    plt.legend(loc='upper left')

    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()


def ml_short(data_nasdaq):
    X = data_nasdaq 
    y = data_nasdaq['next_year_coffee_avg'] 


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    '''scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.drop('date', axis=1))
    X_test_scaled = scaler.transform(X_test.drop('date', axis=1))'''

    '''scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)'''
    '''model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)'''

    model = LinearRegression()
    model.fit(X_train.drop('date', axis=1), y_train)
    y_pred = model.predict(X_test.drop('date', axis=1))

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")

    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0, len(y_test)-1, len(y_test)), y_test, label='Valeurs réelles', color='blue')
    plt.plot(np.linspace(0, len(y_test)-1, len(y_test)), y_pred, label='Valeurs prédites', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Prix du café')
    plt.title('Prix du café - Réel vs Prédit')
    plt.legend()
    plt.grid(True)
    plt.show()


from DataMonitor import DataMonitor 

dataMonitor = DataMonitor()

#Already extracted
dataMonitor.collectData('dxy.txt', 'escp_msf_exercise.sqlite')
#dataMonitor.saveDataMainAs('data_output.csv')'''
#dataMonitor.openData1('data_output.csv')

#Already extracted
dataMonitor.collectDataFromNasdaq()
#dataMonitor.saveDataNasdaqAs('fetched_nasdaq.csv')'''
#dataMonitor.openData2('fetched_nasdaq.csv')

dataMonitor.mergeAll()
print(dataMonitor._data_all)
dataMonitor.fillData('all', 'ffill')
dataMonitor.filterAfterDate("2006-04-30")
dataMonitor.describe()
dataMonitor.tocsv()


best_index = best_corr(dataMonitor._data_nasdaq, dataMonitor._data_prof, 5)

filtered_columns = ['date', 'next_year_coffee_avg'] + best_index 
print(best_index)

data_nasdaq = dataMonitor._data_nasdaq[filtered_columns]
data_nasdaq = data_nasdaq.drop(data_nasdaq.index[-1])
data_nasdaq.to_csv("nasdaq_output.csv", index=False)

ml_short(data_nasdaq)












