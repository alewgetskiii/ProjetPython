import pandas as pd
import sqlite3
import nasdaqdatalink
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np




dxy = pd.read_csv('dxy.txt', sep=",", parse_dates=["date"], skipinitialspace=True)
dxy['date'] = pd.to_datetime(dxy['date'], errors='coerce')

conn = sqlite3.connect('escp_msf_exercise.sqlite')

data = pd.read_sql_query("SELECT * FROM data", conn)
variables_reference = pd.read_sql_query("SELECT * FROM variables_reference", conn)

conn.close()

data = pd.merge(data, variables_reference, on="id_variable")
data = data.pivot(index="date", columns="variable", values="value")
data.reset_index(inplace=True)

data['date'] = pd.to_datetime(data['date'], errors='coerce')
data = pd.merge(data, dxy, on="date", how="left")

data.rename(columns={'value': 'dxy_value'}, inplace=True)

data.to_csv("data_output.csv", index=False)

def fetch_and_filter():
    #Fetch des data
    nasdaqdatalink.ApiConfig.api_key = "6JGsdynvBPRj-yNcyJm3"
    data_nasdaq = nasdaqdatalink.get_table("QDL/ODA", paginate=True)
    data_nasdaq = data_nasdaq[data_nasdaq['indicator'].str.startswith(("USA", "CHN", "COL"))] 

    #ajout de la date et choix de la période
    data_nasdaq = data_nasdaq.pivot(index="date", columns="indicator", values="value")
    data_nasdaq.reset_index(inplace=True)
    data_nasdaq = data_nasdaq[(data_nasdaq['date'] >= "1990-01-01") & (data_nasdaq['date'] <= "2020-12-31")]

    #Suppression des colones avec beaucoup de donné manquante
    threshold = len(data_nasdaq) * 0.9
    data_nasdaq = data_nasdaq.dropna(axis=1, thresh=threshold)

    #Filtre de indicateur pertinant
    terms_to_keep = [
        "TXG_RPCH", "NGDP_FY", "NGDP_D", "NGDP_RPCH", "PPPEX", 
        "TMG_RPCH", "PCPIPCH", "PCPIEPCH", "PCPI", "PCPIE", "NGAP_NPGDP"
    ]
    filtered_columns = ['date'] + [col for col in data_nasdaq.columns if any(col.endswith(term) for term in terms_to_keep)]
    data_nasdaq = data_nasdaq[filtered_columns]  

    data_nasdaq = data_nasdaq.drop(['CHN_NGDP_FY','CHN_NGDP_D','CHN_NGDP_RPCH'], axis=1)
    data_nasdaq.to_csv('fetched_nasdaq.csv', index=False)
    return data_nasdaq

def best_corr(data_nasdaq, n):

    #Formatage des dates
    data['date'] = pd.to_datetime(data['date'])
    data_nasdaq['date'] = pd.to_datetime(data_nasdaq['date'])

    #Ajout du prix annuel moyen Shifter de -&
    data['year'] = data['date'].dt.year
    annual_coffee_avg = data.groupby('year')['coffee_nearby'].mean()
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


#fetched_data = fetch_and_filter()
   


data_nasdaq = pd.read_csv('fetched_nasdaq.csv')

best_index = best_corr(data_nasdaq, 5)

filtered_columns = ['date', 'next_year_coffee_avg'] + best_index 
print(best_index)

data_nasdaq = data_nasdaq[filtered_columns]
data_nasdaq = data_nasdaq.drop(data_nasdaq.index[-1])
data_nasdaq.to_csv("nasdaq_output.csv", index=False)





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

ml_short(data_nasdaq)












