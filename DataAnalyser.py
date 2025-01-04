import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

class DataAnalyser:    

    _data = None

    def __init__(self, data):
        self.setData(data)
        
    def getCorrelMatrix(self):
        return self._data.corr()
    
    def getBestCorrel(self, n):
        return self.getCorrelMatrix()['coffee_price'].drop('coffee_price').abs().sort_values(ascending=False).head(n).index.tolist()
    
    def linearRegCoef(self, variables, with_constant):
        x = sm.add_constant(self.getData()[variables]) if with_constant else self.getData[variables]
        ols = sm.OLS(self.getLabelValue(), x)
        ols_result = ols.fit()
        ols_result.summary()
        if with_constant:
            var_beta = ols_result.bse[1]**2
            var_intercept = ols_result.bse[0]**2
            beta = ols_result.params[1]
            intercept = ols_result.params[0]
            return beta, intercept, var_beta, var_intercept
        else:
            var_beta = ols_result.bse[0]**2
            beta = ols_result.params[0]
            return beta, var_beta
    
    def displayRegression(self, variables, hedge_ratios, intercepts, isDisplayed=False):
        residuals = np.add(self.getLabelValue(),-1*np.add(intercepts, np.multiply(hedge_ratios, self.getData()[variables])))

        if isDisplayed:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
            ax = [i for i in range(len(residuals))]
            ax1.plot(self._data.index , self.getLabelValue(), 'r--', label='Returns Test', color='black')
            ax1.plot(ax, np.add(intercepts, np.multiply(hedge_ratios, self.getData()[variables])), color='purple', label='Prediction')
            ax1.set_title('Return vs Prediction')
            ax1.legend()

            ax2.plot(self._data.index, residuals, label='Residuals', color='blue')
            ax2.set_title('Residuals')
            ax2.legend()

            plt.tight_layout()
            plt.show()
    
    def getLabelValue(self):
        return self._data['coffee_price']

    def getData(self):
        return self._data
    
    def setData(self, data):
        self._data = data