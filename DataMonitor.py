import pandas as pd
import sqlite3
import nasdaqdatalink



class DataMonitor:

    _data_prof = None
    _data_nasdaq = None
    _data_all = None

    def __init__(self):
        pass

    'if data already extracted, avoid doing same process'
    def openData1(self, file_to_open):
        self._data_prof = pd.read_csv(file_to_open)
    
    def openData2(self, file_to_open):
        self._data_nasdaq = pd.read_csv(file_to_open)

    def collectData(self, path_txt, path_sql):
        'Collect data from .txt and shape date format'
        data_txt = pd.read_csv(path_txt)
        data_txt['date'] = pd.to_datetime(data_txt['date'], errors='coerce')

        'Collect data from sql and merge on id'
        root_sql = sqlite3.connect(path_sql)
        data_sql = pd.read_sql_query("SELECT * FROM data", root_sql)
        variables_reference = pd.read_sql_query("SELECT * FROM variables_reference", root_sql)
        root_sql.close()    
        data_sql = pd.merge(data_sql, variables_reference, on="id_variable")
        data_sql = data_sql.pivot(index="date", columns="variable", values="value")
        data_sql.reset_index(inplace=True)
        data_sql['date'] = pd.to_datetime(data_sql['date'], errors='coerce')

        'Merge all data'
        data = pd.merge(data_sql, data_txt, on="date", how="left")
        data.rename(columns={'value': 'dxy_value'}, inplace=True)

        self._data_prof = data
    

    def saveDataMainAs(self, filename):
        self._data_prof.to_csv(filename, index=False)

    def saveDataNasdaqAs(self, filename):
        self._data_nasdaq.to_csv(filename, index=False)

    def collectDataFromNasdaq(self):
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
        "TXG_RPCH", "NGDP_RPCH", "PPPEX", "TMG_RPCH", "PCPI"
    ]
        filtered_columns = ['date'] + [col for col in data_nasdaq.columns if any(col.endswith(term) for term in terms_to_keep)]
        data_nasdaq = data_nasdaq[filtered_columns]  

        #data_nasdaq = data_nasdaq.drop(['CHN_NGDP_FY','CHN_NGDP_D','CHN_NGDP_RPCH',], axis=1)
        data_nasdaq.to_csv('fetched_nasdaq.csv', index=False)

        self._data_nasdaq = data_nasdaq

    def mergeAll(self):
        self._data_all = pd.merge(self._data_prof, self._data_nasdaq, on='date', how='left')
        self._data_all = self._data_all.set_index('date')

    def fillData(self, variables, method):
        if variables == 'all':
            self._data_all = self._data_all.ffill()
        else:
            self._data_all[variables] = self._data_all[variables].fillna(method=method)
    
    
    def filterAfterDate(self, date):
        if not isinstance(self._data_all.index, pd.DatetimeIndex):
            self._data_all.index = pd.to_datetime(self._data_all.index)

        # Appliquer le filtre
        self._data_all = self._data_all[self._data_all.index > pd.Timestamp(date)]

    def tocsv(self):
        self._data_all.to_csv("data_from_class.csv", index=False)
    
    def describe(self):
        print(self._data_all.describe())
        