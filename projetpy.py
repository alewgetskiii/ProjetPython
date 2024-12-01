import pandas as pd
import sqlite3
import nasdaqdatalink

nasdaqdatalink.ApiConfig.api_key = "6JGsdynvBPRj-yNcyJm3"


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


data_nasdaq = nasdaqdatalink.get_table("QDL/ODA", paginate=True)
data_nasdaq = data_nasdaq[data_nasdaq['indicator'].str.startswith(("USA", "CHN", "COL"))]

#data_nasdaq = pd.read_csv('output.csv')
#2020-02-11
data_nasdaq = data_nasdaq.pivot(index="date", columns="indicator", values="value")
data_nasdaq.reset_index(inplace=True)
data_nasdaq = data_nasdaq[(data_nasdaq['date'] >= "1990-01-01") & (data_nasdaq['date'] <= "2020-12-31")]
data_nasdaq.to_csv("nasdaq_output.csv", index=False)


print(data)







