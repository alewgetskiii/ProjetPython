import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import pandas as pd

class DataAnalyser:    

    _data = None
    _returns = None

    def __init__(self, data, returns):
        self.setData(data)
        self.setReturns(returns)

    def getFrequency(self, col):
        ''' in days '''
        return pd.to_datetime(self._data[col].dropna().index.to_series()).diff().value_counts().index[0].days

    def getDates(self, col):
        return np.array(self._data[self._data[col].notna()].index)
    
    def getEffects(self, col, shift_max):
        dates = self.getDates(col)
        pos_dates = [self._data.index.get_loc(date) for date in dates]
        _shift_max = min(len(self._data)-pos_dates[-1]-1, shift_max)
        returns_col = [np.array(self._data[col])[pos] for pos in pos_dates]
        res = [0, 0]
        if np.std(returns_col) == 0:
                return res
        for shift in range(_shift_max+1):
            returns_coffee = [np.array(self._data['r_coffee'])[pos+shift] for pos in pos_dates]
            res = [shift, np.corrcoef(returns_coffee, returns_col)[0, 1]] if abs(np.corrcoef(returns_coffee, returns_col)[0, 1])>abs(res[1]) else res
        return res

    
    def plotEffects(self, col):
        dates = self.getDates(col)
        pos_dates = [self._data.index.get_loc(date) for date in dates]
        pos_up = [pos for pos in pos_dates if np.array(self._data[col])[pos]>0]
        pos_down = [pos for pos in pos_dates if np.array(self._data[col])[pos]<0]
        up = [date for date in dates if self._data.loc[date, col]>0]
        down = [date for date in dates if self._data.loc[date, col]<0]

        fig, (ax1) = plt.subplots(1, 1, figsize=(12, 12))
        ax1.plot(self._data.index , self.getColValue('coffee'), alpha = 0.5, label='Coffee Price', color='blue')
        #ax1.plot(up, [self.getValueByIndex(date, 'coffee') for date in up], '^', 'green', label='Increase of '+col)
        ax1.plot(up, [np.array(self._data['coffee'])[pos] for pos in pos_up], '^', color='green', label='Increase of '+col)
        #ax1.plot(down, [self.getValueByIndex(date, 'coffee') for date in down], 'v', 'red', label='Decrease of '+col)
        ax1.plot(down, [np.array(self._data['coffee'])[pos] for pos in pos_down], 'v', color='red', label='Decrease of '+col)
        ax1.set_title('Coffee price and '+col + ' release')
        ax1.legend()

        plt.tight_layout()
        plt.show()

    def getCorrelMatrix(self):
        return self._data.corr()
    
    def getBestCorrel(self, n):
        return self.getCorrelMatrix()['coffee'].drop('coffee').abs().sort_values(ascending=False).head(n).index.tolist()
    


    def linearRegCoef(self, variables, with_constant, display):
        data = self.getColValue(variables).shift(-1)
        data['r_coffee'] = self._data['r_coffee']
        data = data.dropna()
        x = sm.add_constant(data[variables]) if with_constant else self.getColValue(variables)
        ols = sm.OLS(data['r_coffee'], x)
        ols_result = ols.fit()
        print(ols_result.summary())
        if with_constant:
            #var_beta = ols_result.bse[1]**2
            #var_intercept = ols_result.bse[0]**2
            beta = list(ols_result.params)[1:]
            intercept = ols_result.params[0]
        else:
            #var_beta = ols_result.bse[0]**2
            beta = ols_result.params
            intercept = 0
        if display:
                self.displayRegression(variables, beta, intercept)
        return beta, intercept, ols_result.rsquared_adj
    
    def displayRegression(self, variables, hedge_ratios, intercept):
        data = self.getColValue(['r_coffee']+variables).dropna()
        residuals = np.add(data['r_coffee'],-1*np.add(intercept, np.sum(np.multiply(np.array(hedge_ratios), np.array(data[variables])), axis=1)))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        ax = [i for i in range(len(residuals))]
        ax1.plot(data.index , data['r_coffee'], 'r--', label='Coffee Price', color='black')
        ax1.plot(ax, np.add(intercept, np.sum(np.multiply(np.array(hedge_ratios), np.array(data[variables])), axis=1)), color='purple', label='Prediction: '+str(variables))
        ax1.set_title('Actual vs Prediction')
        ax1.legend()

        ax2.plot(data.index, residuals, label='Residuals', color='blue')
        ax2.set_title('Residuals')
        ax2.legend()

        plt.tight_layout()
        plt.show()
        
    def optimizerRegression(self, variables, max_features, top_best):
        if variables == 'all':
            variables = self._data.columns[1:] #don't take coffee as a feature
        results = {}
        for sub_variables in [list(x) for i in range(1, max_features + 1) for x in combinations(variables, i)]:
            key = ''
            for var in sub_variables:
                key += var + ','
            key = key[:-1]
            results[key] = self.linearRegCoef(sub_variables, True, False)
        return list(sorted(results.items(), key=lambda x:x[1][-1], reverse=True))[:top_best]

    def getLabelValue(self):
        return self._data[self._data['r_coffee'].notna()]['r_coffee']
    
    def getColValue(self, col):
        return self._data[self._data[col].notna()][col]

    
    def getValueByIndex(self, index, col):
        return self._data.at[index, col]

    def getData(self):
        return self._data
    
    def setData(self, data):
        self._data = data

    def setReturns(self, data):
        self._returns = data