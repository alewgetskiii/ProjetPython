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
best_coef = dataAnalyser.getBestCorrel(10)
print(best_coef)
variables = ['china_gdp_yoy_forecast', 'coffee_nearby']
beta, intercept = dataAnalyser.linearRegCoef(variables, with_constant=True)
dataAnalyser.displayRegression(variables, beta, intercept, True)