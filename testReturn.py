from DataMonitor import DataMonitor 
from DataAnalyser import DataAnalyser
import numpy as np


dataMonitor = DataMonitor()

#if not already extracted and saved
#dataMonitor.collectDataProf('dxy.txt', 'escp_msf_exercise.sqlite')
dataMonitor.openDataProf('data_prof.csv')
dataMonitor.openReturnProf('return_prof.csv')

#if not already extracted and saved
#dataMonitor.collectDataFromNasdaq()
dataMonitor.openDataNasdaq('data_nasdaq.csv')
dataMonitor.openReturnNasdaq('return_nasdaq.csv')

print(dataMonitor._return_nasdaq)
print(dataMonitor._return_prof)



dataMonitor.mergeData()
dataMonitor.mergeReturns()
print(dataMonitor._data_all)
print(dataMonitor._return_all)


dataAnalyser = DataAnalyser(dataMonitor.getDataAll(), dataMonitor.getReturnAll())
print(dataAnalyser.getCorrelMatrix(dataAnalyser._data))
print(dataAnalyser.getCorrelMatrix(dataAnalyser._returns))


'''results_effect = {}
for col in dataAnalyser.getReturns().columns[1:]:
    results_effect[col] = dataAnalyser.getEffectsReturns(col, shift_max=3)
top_res = list(sorted(results_effect.items(), key=lambda x:abs(x[1][1]), reverse=True))[:10]
print(top_res)'''

freq = [dataAnalyser.getFrequencyReturns(col) for col in dataAnalyser._returns.columns[2:]]
annual_variables = [col for col in dataAnalyser._returns.columns[2:] if dataAnalyser.getFrequencyReturns(col)==365]
monthly_variables = [col for col in dataAnalyser._returns.columns[2:] if dataAnalyser.getFrequencyReturns(col)==31]
weekly_variables = [col for col in dataAnalyser._returns.columns[2:] if dataAnalyser.getFrequencyReturns(col)==7]
daily_variables = [col for col in dataAnalyser._returns.columns[2:] if dataAnalyser.getFrequencyReturns(col)==1]


'''for col in annual_variables:
    annual_price_coffee = [dataAnalyser._data['coffee'][date] for date in dataAnalyser.getDatesData(col[2:])]
    annual_return_coffee = np.array([(annual_price_coffee[i+1]/annual_price_coffee[i])-1 for i in range(len(annual_price_coffee)-1)])
    annual_returns_col = np.array(dataAnalyser.getColReturns(col))
    print('correlation : ' +col + ' : '+str(dataAnalyser.getCorrel(annual_return_coffee, annual_returns_col, lag=0)))'''

annual_correl = dataAnalyser.getCorrelationByFrequency(annual_variables, range_lag=2, top_best=10)
print(annual_correl)
monthly_correl = dataAnalyser.getCorrelationByFrequency(monthly_variables, range_lag=2, top_best=10)
#print(monthly_correl)
weekly_correl = dataAnalyser.getCorrelationByFrequency(weekly_variables, range_lag=2, top_best=10)
#print(weekly_correl)
daily_correl = dataAnalyser.getCorrelationByFrequency(daily_variables, range_lag=2, top_best=10)
#print(daily_correl)


for varAndLag, _ in annual_correl.items():
    beta, intercept, score, score_adj = dataAnalyser.linearRegByFrequency(varAndLag[:-6], int(varAndLag[-1]))
    print(score)

print('Meilleur causalite p<0.05 : ')
causalities = dataAnalyser.causal(annual_variables, 5, 0.1)
print(causalities)


''' Regression multiple from result causalities'''
optimal_lags = [int(causalities[col].idxmin()[-1]) for col in causalities.columns]
beta, intercept, score, score_adj = dataAnalyser.linearRegWithLagsByFrequency(causalities.columns, optimal_lags)
print(score)

#dataAnalyser.RandomForest(annual_variables, daily_variables)

''' On fait une regression linéaire multiple sur chaque frequence SEPAREMMENT: annuelle, mensuelle, weekkly, daily
puis on aggrege les modelisations sur une fréquence choisie '''
variables_by_freq = [annual_variables,weekly_variables,daily_variables] 
#montlhy have no enough impact -> lags vide, on enleve
#weekly : 1 variable
#daily : 2 variables
betas_by_freq = []
intercept_by_freq = []
optimal_lags_by_freq = []
for variables in variables_by_freq:
    causalities = dataAnalyser.causal(variables, max_lag=5, filter_p_value=0.20)
    optimal_lags = [int(causalities[col].idxmin()[-1]) for col in causalities.columns]
    betas, intercept, score, score_adj = dataAnalyser.linearRegWithLagsByFrequency(causalities.columns, optimal_lags)
    betas_by_freq.append(betas)
    intercept_by_freq.append(intercept)
    optimal_lags_by_freq.append(optimal_lags)
