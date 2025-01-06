import pandas as pd
import sqlite3
import nasdaqdatalink
import numpy as np



class DataMonitor:

    _data_prof = None
    _data_nasdaq = None
    _data_all = None
    _return_prof = None
    _return_nasdaq = None
    _return_all = None

    def __init__(self):
        pass

    'if data already extracted, avoid doing same process'
    def openDataProf(self, file_to_open):
        self._data_prof = pd.read_csv(file_to_open)
    
    def openDataNasdaq(self, file_to_open):
        self._data_nasdaq = pd.read_csv(file_to_open)

    def openReturnProf(self, file_to_open):
        self._return_prof = pd.read_csv(file_to_open)
    
    def openReturnNasdaq(self, file_to_open):
        self._data_nasdaq = pd.read_csv(file_to_open)

    ''' Collect Data and Returns from prof's files '''
    def collectDataProf(self, path_txt, path_sql):
        'Collect data from .txt and shape date format'
        data_txt = pd.read_csv(path_txt)
        data_txt['date'] = pd.to_datetime(data_txt['date'], errors='coerce')
        data_txt.rename(columns={' value': 'coffee'}, inplace=True)
        self.addReturns(data_txt)

        'Collect data from sql and merge on id'
        root_sql = sqlite3.connect(path_sql)
        data_sql = pd.read_sql_query("SELECT * FROM data", root_sql)
        variables_reference = pd.read_sql_query("SELECT * FROM variables_reference", root_sql)
        root_sql.close()

        '''Compute and add returns for each variable before pivot'''
        returns = []
        for i in range(1, len(variables_reference)+1):
            returns.extend(self.computeReturns(data_sql[data_sql['id_variable']==i]['value']))
        data_sql['return'] = returns

        data_sql = pd.merge(data_sql, variables_reference, on="id_variable")
        ''' data (prices) '''
        price_sql = data_sql.pivot(index="date", columns="variable", values="value")
        price_sql.reset_index(inplace=True)
        price_sql['date'] = pd.to_datetime(price_sql['date'], errors='coerce')
        ''' returns '''
        return_sql = data_sql.pivot(index="date", columns="variable", values="return")
        return_sql.reset_index(inplace=True)
        return_sql['date'] = pd.to_datetime(return_sql['date'], errors='coerce')
        return_sql.columns = ['date'] + ['r_' + col for col in return_sql.columns[1:]]

        'Merge all data, how=outer to include all data'
        data_prof = pd.merge(data_txt.drop(columns=['r_coffee'], inplace=False), price_sql, on="date", how="outer")
        data_prof.sort_values(by='date', ascending=True, inplace=True)
        return_prof = pd.merge(data_txt.drop(columns=['coffee'], inplace=False), return_sql, on="date", how="outer")
        return_prof.sort_values(by='date', ascending=True, inplace=True)

        data_prof.to_csv('data_prof.csv', index=False)
        return_prof.to_csv('return_prof.csv', index=False)

        self.setDataProf(data_prof)
        self.setReturnProf(return_prof)

    def collectDataProfInitial(self, path_txt, path_sql):
        'Collect data from .txt and shape date format'
        data_txt = pd.read_csv(path_txt)
        data_txt['date'] = pd.to_datetime(data_txt['date'], errors='coerce')
        data_txt.rename(columns={' value': 'coffee'}, inplace=True)

        'Collect data from sql and merge on id'
        root_sql = sqlite3.connect(path_sql)
        data_sql = pd.read_sql_query("SELECT * FROM data", root_sql)
        variables_reference = pd.read_sql_query("SELECT * FROM variables_reference", root_sql)
        root_sql.close()    
        data_sql = pd.merge(data_sql, variables_reference, on="id_variable")
        data_sql = data_sql.pivot(index="date", columns="variable", values="value")
        data_sql.reset_index(inplace=True)
        data_sql['date'] = pd.to_datetime(data_sql['date'], errors='coerce')

        'Merge all data, how=outer to include all data'
        data = pd.merge(data_txt, data_sql, on="date", how="outer")

        self.setDataProf(data)


    def collectDataFromNasdaqInitial(self):
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
        data_nasdaq['date'] = pd.to_datetime(data_nasdaq['date'], errors='coerce')


        #data_nasdaq = data_nasdaq.drop(['CHN_NGDP_FY','CHN_NGDP_D','CHN_NGDP_RPCH',], axis=1)
        data_nasdaq.to_csv('data_nasdaq.csv', index=False)

        self._data_nasdaq = data_nasdaq

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
        threshold = int(len(data_nasdaq) * 0.9)
        data_nasdaq = data_nasdaq.dropna(axis=1, thresh=threshold)

        #Filtre de indicateur pertinant
        terms_to_keep = [
        "TXG_RPCH", "NGDP_RPCH", "PPPEX", "TMG_RPCH", "PCPI"
    ]
        filtered_columns = ['date'] + [col for col in data_nasdaq.columns if any(col.endswith(term) for term in terms_to_keep)]
        data_nasdaq = data_nasdaq[filtered_columns]
        data_nasdaq['date'] = pd.to_datetime(data_nasdaq['date'], errors='coerce')

        ''' dataframe of returns '''
        return_nasdaq = data_nasdaq.copy()
        '''     Add returns for each variable'''
        self.addReturns(return_nasdaq)
        '''     Remove non-return data'''
        columns_returns = ['date'] + [col for col in return_nasdaq.columns[1:] if col[:2]=='r_']
        return_nasdaq = return_nasdaq[columns_returns]

        data_nasdaq.to_csv('data_nasdaq.csv', index=False)
        return_nasdaq.to_csv('return_nasdaq.csv', index=False)
        
        self._data_nasdaq = data_nasdaq
        self._return_nasdaq = return_nasdaq

    def addReturns(self, df):
        for col in df.columns:
            if col != 'date':
                df['r_'+col] = (df[col]/df[col].shift(1))-1
    
    def computeReturns(self, col_df):
        return np.array((col_df/col_df.shift(1))-1)
    
    def mergeData(self):
        self._data_all = pd.merge(self._data_prof, self._data_nasdaq, on='date', how='outer')
        self._data_all.sort_values(by='date', ascending=True, inplace=True)
        self._data_all = self._data_all.set_index('date')
        ''' fill values of coffee as we merged outer '''
        self._data_all['coffee'] = self._data_all['coffee'].fillna(method='ffill')
    
    def mergeReturns(self):
        self._return_all = pd.merge(self._return_prof, self._return_nasdaq, on='date', how='outer')
        self._return_all.sort_values(by='date', ascending=True, inplace=True)
        self._return_all = self._return_all.set_index('date')
        ''' fill values of coffee as we merged outer '''
        self._return_all['r_coffee'] = self._return_all['r_coffee'].fillna(0) #donnees constantes -> return 0

        
    def fillData(self, variables, method):
        if variables == 'all':
            self._data_all = self._data_all.fillna(method=method, axis=0)
        else:
            self._data_all[variables] = self._data_all[variables].fillna(method=method)
    
    
    def filterAfterDate(self, date):
        if not isinstance(self._data_all.index, pd.DatetimeIndex):
            self._data_all.index = pd.to_datetime(self._data_all.index)

        # Appliquer le filtre
        self._data_all = self._data_all[self._data_all.index > pd.Timestamp(date)]

    def tocsv(self,  filename):
        self._data_all.to_csv(filename, index=True)
    
    def describe(self):
        print(self._data_all.describe())
    
    def setDataAll(self, data):
        self._data_all = data
    
    def getDataAll(self):
        return self._data_all
    
    def setDataProf(self, data):
        self._data_prof = data
    
    def setReturnProf(self, data):
        self._return_prof = data
    
    def getDataProf(self):
        return self._data_prof
    
    def saveDataProfAs(self, filename):
        self._data_prof.to_csv(filename, index=False)

    def saveDataNasdaqAs(self, filename):
        self._data_nasdaq.to_csv(filename, index=False)

    def saveReturnProfAs(self, filename):
        self._return_prof.to_csv(filename, index=False)

    def saveReturnNasdaqAs(self, filename):
        self._return_nasdaq.to_csv(filename, index=False)