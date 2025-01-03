from DataMonitor import DataMonitor 

dataMonitor = DataMonitor()

#Already extracted
'''dataMonitor.collectData('dxy.txt', 'escp_msf_exercise.sqlite')
dataMonitor.saveDataMainAs('data_output.csv')'''
dataMonitor.openData1('data_output.csv')

#Already extracted
'''dataMonitor.collectDataFromNasdaq()
dataMonitor.saveDataNasdaqAs('fetched_nasdaq.csv')'''
dataMonitor.openData2('fetched_nasdaq.csv')

dataMonitor.mergeAll()
print(dataMonitor._data_all)
dataMonitor.fillData('all', 'ffill')
dataMonitor.filterAfterDate("2006-04-30")
dataMonitor.describe()
dataMonitor.tocsv()

