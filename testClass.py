from DataMonitor import DataMonitor 
from DataAnalyser import DataAnalyser

dataMonitor = DataMonitor()

#Already extracted
'''dataMonitor.collectData('dxy.txt', 'escp_msf_exercise.sqlite')
dataMonitor.saveDataProfAs('data_prof.csv')'''
dataMonitor.openData1('data_prof.csv')

#Already extracted
'''dataMonitor.collectDataFromNasdaq()
dataMonitor.saveDataNasdaqAs('data_nasdaq.csv')'''
dataMonitor.openData2('data_nasdaq.csv')

dataMonitor.mergeAll()

print(dataMonitor._data_all)
dataMonitor.fillData('all', 'ffill')

''' on enlève les lignes où y'a pas de donnees
pour faire la matrice de correl et regression
sans trous
'''
dataMonitor.setDataAll(dataMonitor.getDataAll().dropna())
dataMonitor.describe()
#dataMonitor.tocsv("data_all.csv")
print(dataMonitor.getDataAll().columns)

'''On commence l'analyse de donnees'''
dataAnalyser = DataAnalyser(dataMonitor.getDataAll())
print(dataAnalyser.getCorrelMatrix())
best_features = dataAnalyser.getBestCorrel(10)
print(best_features)
#variables = ['china_gdp_yoy_forecast', 'coffee_nearby']
#beta, intercept, r_squared_adj  = dataAnalyser.linearRegCoef(variables, with_constant=True, display=True)

best_results = dataAnalyser.optimizerRegression(best_features, max_features=3, top_best=3)
print(best_results)

for res in best_results:
    dataAnalyser.displayRegression(res[0].split(','), res[1][0], res[1][1])