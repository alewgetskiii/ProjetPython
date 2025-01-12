from DataMonitor import DataMonitor 
from DataAnalyser import DataAnalyser
import pandas as pd


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


dataAnalyser = DataAnalyser(dataMonitor.getDataAll(), dataMonitor.getReturnAll())

#dataAnalyser.plotVariable('coffee')

'''results_effect = {}
for col in dataAnalyser.getReturns().columns[1:]:
    results_effect[col] = dataAnalyser.getEffectsReturns(col, shift_max=3)
top_res = list(sorted(results_effect.items(), key=lambda x:abs(x[1][1]), reverse=True))[:10]
print(top_res)'''

freq = [dataAnalyser.getFrequencyReturns(col) for col in dataAnalyser._returns.columns[1:]]
'''on remarque qu'on a seulement des donnees:
annuelles, mensuelles, hebdomadaires et journalieres
qu'on filtre, on veut qu'elles commencent au moins en 1990-12-31 pour avoir assez de donnees pour regresser
vu qu'on raisonne sur les donnees annuelles'''
annual_variables = [col for col in dataAnalyser._returns.columns[1:] if dataAnalyser.getFrequencyReturns(col)==365 and int(dataAnalyser.getColReturns(col).index[0][:4]) <= 1991]
monthly_variables = [col for col in dataAnalyser._returns.columns[1:] if dataAnalyser.getFrequencyReturns(col)==31 and int(dataAnalyser.getColReturns(col).index[0][:4]) <= 1991]
weekly_variables = [col for col in dataAnalyser._returns.columns[1:] if dataAnalyser.getFrequencyReturns(col)==7 and int(dataAnalyser.getColReturns(col).index[0][:4]) <= 1991]
daily_variables = [col for col in dataAnalyser._returns.columns[1:] if dataAnalyser.getFrequencyReturns(col)==1 and int(dataAnalyser.getColReturns(col).index[0][:4]) <= 1991]

'''On remarque que les donnees monthly et weekly sont vides apres le filtre
on ne travaille donc que sur les donnees annual et daily'''

'Annualisons les donnees daily'
for var in daily_variables:
    df = dataAnalyser._data[var[2:]]
    df.index = pd.to_datetime(df.index)
    df = pd.DataFrame(df.resample('Y').last())
    dataAnalyser._data[var] = df.rename(columns={var: var[2:]})
    df = df.pct_change().dropna()
    dataAnalyser._returns[var] = df

annual_variables = annual_variables + daily_variables

''' On split les donnees '''
split_date = '2010-12-31'


annual_correl = dataAnalyser.getCorrelationByFrequency(annual_variables, range_lag=6, 
                                                       top_best=10, split_date=split_date)
print(annual_correl)


'''Causalites'''
causalities = dataAnalyser.causal(annual_variables, max_lag=6, p_value_limit=0.10)
print(causalities)


''' On fait une regression linéaire multiple sur les donnees annuelles '''
'''based on causalites'''
variables_selected = []
lags = []
for col, lag, p_value, _ in causalities[:10]:
    variables_selected.append(col)
    lags.append(lag)

betas, intercept, score, score_adj = dataAnalyser.linearRegWithLagsByFrequency(variables=variables_selected, lags=lags, split_year=split_date[:4], displayPred=True)
print('selected variables : ' +str(variables_selected))
print('lags : ' +str(lags))
print('r²: '+str(score))
print('r² adj: '+str(score_adj))
''' On voit que notre modèle surreagit les baisses
cela est surement du a un probleme de colinearite de nos variables '''

VIF_df = dataAnalyser.getMultiColinearity(variables_selected, lags, split_date[:4])
''' on enleve les variables avec un VIF > 10'''
variables_selected = [var[:-6] for var in VIF_df[VIF_df['VIF']<10]['Variable']]
lags = [int(var[-1]) for var in VIF_df[VIF_df['VIF']<10]['Variable']]
betas, intercept, score, score_adj = dataAnalyser.linearRegWithLagsByFrequency(variables=variables_selected, lags=lags, split_year=split_date[:4], displayPred=True)
print('selected variables : ' +str(variables_selected))
print('lags : ' +str(lags))
print('r²: '+str(score))
print('r² adj: '+str(score_adj))


'''based on correlations'''

'''variables_selected = []
lags = []
for varAndLag, _ in annual_correl.items():
    variables_selected.append(varAndLag[:-6])
    lags.append(int(varAndLag[-1]))

betas, intercept, score, score_adj = dataAnalyser.linearRegWithLagsByFrequency(variables=variables_selected, lags=lags, split_year=split_date[:4], displayPred=True)
print('selected variables : ' +str(variables_selected))
print('lags : ' +str(lags))
print('r²: '+str(score))
print('r² adj: '+str(score_adj))

VIF_df = dataAnalyser.getMultiColinearity(variables_selected, lags, split_date[:4])
on enleve les variables avec un VIF > 10
variables_selected = [var[:-6] for var in VIF_df[VIF_df['VIF']<10]['Variable']]
lags = [int(var[-1]) for var in VIF_df[VIF_df['VIF']<10]['Variable']]
betas, intercept, score, score_adj = dataAnalyser.linearRegWithLagsByFrequency(variables=variables_selected, lags=lags, split_year=split_date[:4], displayPred=True)
print('selected variables : ' +str(variables_selected))
print('lags : ' +str(lags))
print('r²: '+str(score))
print('r² adj: '+str(score_adj))
'''