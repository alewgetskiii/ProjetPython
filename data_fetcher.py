import pandas as pd
import sqlite3
import nasdaqdatalink


class DataFetcher:
    def fetch_local_data(self):
        #Fetch depuis le txt
        dxy = pd.read_csv('dxy.txt', sep=",", parse_dates=["date"], skipinitialspace=True)
        dxy['date'] = pd.to_datetime(dxy['date'], errors='coerce')

        #Fetch depuis SQL
        conn = sqlite3.connect('escp_msf_exercise.sqlite')

        data = pd.read_sql_query("SELECT * FROM data", conn)
        variables_reference = pd.read_sql_query("SELECT * FROM variables_reference", conn)

        conn.close()

        #Reorganise les variables par colonnes
        data = pd.merge(data, variables_reference, on="id_variable")
        data = data.pivot(index="date", columns="variable", values="value")
        data.reset_index(inplace=True)

        #Enleve 'china_gdp_yoy_forecast' car sur Nasdaq elle est plus compléte
        data = data.drop('china_gdp_yoy_forecast', axis=1)

        #Merge les deux data
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data = pd.merge(data, dxy, on="date", how="left")
        data.rename(columns={'value': 'dxy_value'}, inplace=True)

        #Filtre les valeur abérantes du à des oublie de virgules
        for col in data.select_dtypes(include=['number']).columns:
            if col in data.columns:
                mean = data[col].mean()
                std = data[col].std()
                outliers = data[col] > (mean + 3 * std)
            
                data.loc[outliers, col] = data.loc[outliers, col] / 10

        data.to_csv("data_output.csv", index=False)
        data['date'] = pd.to_datetime(data['date'])
    
        return data

    def fetch_web_data(self):
        #Fetch des data
        nasdaqdatalink.ApiConfig.api_key = "6JGsdynvBPRj-yNcyJm3"
        data_nasdaq = nasdaqdatalink.get_table("QDL/ODA", paginate=True)
        data_nasdaq = data_nasdaq[data_nasdaq['indicator'].str.startswith(("USA", "CHN", "COL"))] 

        #ajout de la date et choix de la période
        data_nasdaq = data_nasdaq.pivot(index="date", columns="indicator", values="value")
        data_nasdaq.reset_index(inplace=True)
        data_nasdaq = data_nasdaq[(data_nasdaq['date'] >= "1990-01-01") & (data_nasdaq['date'] <= "2020-12-31")]

        #Suppression des colones avec beaucoup de donné manquante
        #threshold = len(data_nasdaq) * 0.9
        #data_nasdaq = data_nasdaq.dropna(axis=1, thresh=threshold)

        #Filtre de indicateur pertinant
        terms_to_keep = [
            "TXG_RPCH", "NGDP_RPCH", "PPPEX", "TMG_RPCH", "PCPI"
        ]
        filtered_columns = ['date'] + [col for col in data_nasdaq.columns if any(col.endswith(term) for term in terms_to_keep)]
        data_nasdaq = data_nasdaq[filtered_columns]  

        data_nasdaq = data_nasdaq.drop('USA_PPPEX', axis = 1)
        data_nasdaq['date'] = pd.to_datetime(data_nasdaq['date'])

        data_nasdaq.to_csv('fetched_nasdaq.csv', index=False)
        return data_nasdaq

