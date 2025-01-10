from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  
from sklearn.model_selection import GridSearchCV, train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore", category=FutureWarning)

class DataAnalyser:    

    _data = None
    _returns = None

    def __init__(self, data, returns):
        self.setData(data)
        self.setReturns(returns)
    
    def getFrequencyReturns(self, col):
        ''' in days '''
        return pd.to_datetime(self._returns[col].dropna().index.to_series()).diff().value_counts().index[0].days

    def getDatesData(self, col):
        return np.array(self._data[self._data[col].notna()].index)
    
    def getDates(self, col):
        data = self._data if col in self._data.columns else self._returns
        return np.array(data[data[col].notna()].index)
    
    def getDatesReturns(self, col):
        return np.array(self._returns[self._returns[col].notna()].index)
    
    def getCorrelationByFrequency(self, variables, range_lag, top_best, split_date):
        correlations = {}
        for var in variables:
            returns_var = self.getColReturns(var)[self.getColReturns(var).index >= '1991-12-31']
            returns_var = returns_var[returns_var.index <= split_date]
            for lag in range (range_lag+1):
                correlations[var+' lag '+str(lag)] = self.getCorrel(self.getAnnualCoffeeReturn()[:len(returns_var)], returns_var, lag)
        return dict(list(dict(sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)).items())[:min(top_best, len(variables)*range_lag+1)])
    
    def linearRegWithLagsByFrequency(self, variables, lags, split_year, displayPred):
        max_lag = max(lags)
        x = []
        for i in range(len(variables)):
            if lags[i] == 0:
                x.append(list(self.getColReturns(variables[i])[self.getColReturns(variables[i]).index >= '1991-12-31'][max_lag-lags[i]:]))
            else:
                x.append(list(self.getColReturns(variables[i])[self.getColReturns(variables[i]).index >= '1991-12-31'][max_lag-lags[i]:-lags[i]]))
        x = list(zip(*x))
        x = np.array([list(obs) for obs in x])

        return_coffee = self.getAnnualCoffeeReturn()[max_lag:]
        X = sm.add_constant(x)

        'Split train and test'
        X_train, X_test, y_train, y_test = self.split_year(X, return_coffee, year=split_year)

        ols = sm.OLS(y_train, X_train)
        ols_result = ols.fit()
        y_pred = ols_result.predict(X_test)
        beta = list(ols_result.params)[1:]
        intercept = ols_result.params[0]

        if displayPred:
            self.plotPrediction(y_test, y_pred)
        
        return beta, intercept, ols_result.rsquared ,ols_result.rsquared_adj

    def plotPrediction(self, y, y_pred):
        fig, (ax1) = plt.subplots(1, 1, figsize=(12, 12))
        ax1.plot(y.index , y, label='Coffee returns', color='blue')
        ax1.plot(y.index , y_pred, label='Prediction', color='green')
        ax1.set_title('Coffee Return vs Prediction')
        ax1.legend()
        plt.show()

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

    def causal(self, variables, max_lag, p_value_limit):
        results = []
        lag_matrix = {var: [None] * max_lag for var in variables}  # Dictionnaire pour stocker les p-values par lag

        ''' on teste la causalite des autres variables sur les returns du coffee'''
        for var in variables:
            returns_var = np.array(self.getColReturns(var)[self.getColReturns(var).index >= '1991-12-31'])
            data = pd.DataFrame({'y': self.getAnnualCoffeeReturn()[:len(returns_var)], 'x': returns_var})
            try:
                test_result = grangercausalitytests(data[['y', 'x']], max_lag, verbose=False)
                p_values = [test_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
            
                # Filtrage des p-values inférieures à la limite
                valid_p_values = [(var, lag, p_value) for lag, p_value in enumerate(p_values, start=1) if p_value < p_value_limit]
                for lag, p_value in enumerate(p_values, start=1):
                    lag_matrix[var][lag-1] = p_value

                if valid_p_values:
                    # Ajouter les p-values valides dans la liste des résultats
                    results.extend(valid_p_values)
        
            except Exception as e:
                print(f"Erreur avec la colonne {var}: {e}")
        
        '''Trier les résultats par p_value (croissant)'''
        sorted_results = sorted(results, key=lambda x: x[2])
        seen_first_values = set()

        '''Parcours des triplets'''
        unique_results = []
        for triplet in sorted_results:
            first_value = triplet[0]
            if first_value not in seen_first_values:
                unique_results.append(triplet)
                seen_first_values.add(first_value)

        # Convertir le dictionnaire en DataFrame
        lag_df = pd.DataFrame(lag_matrix)
        # Afficher la matrice des p-values pour chaque variable et lag
        print("Matrice filtree des p-values pour chaque variable et lag :")
        print(lag_df)

        return unique_results

    def RandomForest(self, X_daily, X_annual):
        valid_columns = [col for col in X_daily + X_annual if col in self._returns.columns]
        X_combined = self._returns[valid_columns]  
        X_combined = X_combined.ffill()  
        X_combined = X_combined.fillna(0)
        print(X_combined)
        y = self._returns['r_coffee']
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }

        #grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
        #grid_search.fit(X_train, y_train)

        #best_model = grid_search.best_estimator_
        #print("Best parameters:", grid_search.best_params_)
        model = RandomForestRegressor(max_depth= 10, max_features= 'sqrt', min_samples_leaf= 4, min_samples_split=2, n_estimators=300, random_state=42)
        #model = XGBRegressor(max_depth= 10, max_features= 'sqrt', min_samples_leaf= 4, min_samples_split=2, n_estimators=300, random_state=42)
        #model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test) 

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)


        # Affichage des métriques
        
        print(f"R-squared: {r2:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7, color='b', label='Prédictions')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', linestyle='--', label='Idéal')
        plt.xlabel("Valeurs Réelles")
        plt.ylabel("Valeurs Prédites")
        plt.title("Prédictions vs Valeurs Réelles")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Retourner éventuellement les scores pour utilisation ultérieure
        return r2, mae, mse, rmse

    
    def plotVariable(self, col):
        data = self.getValue(col)
        fig, (ax1) = plt.subplots(1, 1, figsize=(12, 12))
        ax1.plot(data.index , data, label=col, color='blue')
        ax1.set_title(col)
        ax1.legend()
        plt.tight_layout()
        plt.show()

    def getAnnualCoffeeReturn(self):
        df = self._data['coffee']
        df.index = pd.to_datetime(df.index)
        df = pd.DataFrame(df.resample('Y').last())
        return df.pct_change().dropna()['coffee']

    

    def getCorrelMatrix(self, data):
        return data.corr()
    
    def getBestCorrel(self, n):
        return self.getCorrelMatrix()['coffee'].drop('coffee').abs().sort_values(ascending=False).head(n).index.tolist()
    
    def getCorrel(self, values1, values2, lag):
        if lag > 0:
            return np.corrcoef(values1[lag:], values2[:-lag])[0, 1]
        else:
            return np.corrcoef(values1, values2)[0, 1]

    def linearRegCoef(self, variables, with_constant, display):
        data = self.getColValue(variables)
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
        
    def linearRegByFrequency(self, variable, lag):
        price_coffee = [self._data['coffee'][date] for date in self.getDatesData(variable[2:])]
        return_coffee = np.array([(price_coffee[i+1]/price_coffee[i])-1 for i in range(len(price_coffee)-1)])
        returns_var = np.array(self.getColReturns(variable))
        if lag>0:
            return_coffee, returns_var = return_coffee[lag:], returns_var[:-lag]
        x = sm.add_constant(returns_var)
        ols = sm.OLS(return_coffee, x)
        ols_result = ols.fit()
        beta = list(ols_result.params)[1:]
        intercept = ols_result.params[0]
        return beta, intercept, ols_result.rsquared ,ols_result.rsquared_adj

    
    
    

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

    
    def getColValue(self, col):
        return self._data[self._data[col].notna()][col]

    
    def getValue(self, col):
        data = self._data if col in self._data.columns else self._returns
        return data[data[col].notna()][col]
    
    def getColReturns(self, col):
            return self._returns[self._returns[col].notna()][col]

    def getValueByIndex(self, index, col):
        return self._data.at[index, col]

    def getData(self):
        return self._data
    
    def getReturns(self):
        return self._returns
    
    def setData(self, data):
        self._data = data

    def setReturns(self, data):
        self._returns = data

    def split_year(self, X, y, year):
        cutoff_date = year+"-12-31"
        ''' y is df with date as index whereas X is array'''
        y_train = y[y.index <= cutoff_date]
        y_test = y[y.index > cutoff_date]
        X_train = X[:len(y_train)]
        X_test = X[len(y_train):]
    
        return X_train, X_test, y_train, y_test