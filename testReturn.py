from DataMonitor import DataMonitor 
from DataAnalyser import DataAnalyser

dataMonitor = DataMonitor()

#if not already extracted and saved
#dataMonitor.collectDataProf('dxy.txt', 'escp_msf_exercise.sqlite')
dataMonitor.openDataProf('data_prof.csv')
dataMonitor.openReturnProf('return_prof.csv')

#if not already extracted and saved
#dataMonitor.collectDataFromNasdaq()
dataMonitor.openDataNasdaq('data_nasdaq.csv')
dataMonitor.openReturnNasdaq('return_nasdaq.csv')


dataMonitor.mergeData()
dataMonitor.mergeReturns()
print(dataMonitor._data_all)
print(dataMonitor._return_all)


dataAnalyser = DataAnalyser(dataMonitor.getDataAll(), dataMonitor.getReturnAll())
print(dataAnalyser.getCorrelMatrix())

'''

results_effect = {}
for col in dataAnalyser.getData().columns[2:]:
    results_effect[col] = dataAnalyser.getEffects(col, shift_max=3)
top_res = list(sorted(results_effect.items(), key=lambda x:abs(x[1][1]), reverse=True))[:10]
print(top_res)
#dataAnalyser.plotEffects(top_res[0][0])

freq = [dataAnalyser.getFrequency(col) for col in dataAnalyser._data.columns[2:]]
annual_variables = [col for col in dataAnalyser._data.columns[2:] if dataAnalyser.getFrequency(col)==365]
monthly_variables = [col for col in dataAnalyser._data.columns[2:] if dataAnalyser.getFrequency(col)==31]
weekly_variables = [col for col in dataAnalyser._data.columns[2:] if dataAnalyser.getFrequency(col)==7]
daily_variables = [col for col in dataAnalyser._data.columns[2:] if dataAnalyser.getFrequency(col)==1]

_, _, _ = dataAnalyser.linearRegCoef(daily_variables, True, True)'''