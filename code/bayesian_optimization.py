import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from skopt.space import Real
from sklearn.utils import parallel_backend
from skopt import BayesSearchCV
import data_handler
import matplotlib.pyplot as plt

class Optimization:
    __company_name = None
    __company_data = None
    __prices_data = None
    __quarters = None
    __max_days = None
    __opt = None
    __alpha = None

    def __init__(self, company_name: str):
        self.__company_name = company_name
        self.__company_data = data_handler.CsvHandler(company_name)
        self.__prices_data = self.__company_data.get_equal_length_prices()
        self.__quarters = self.__company_data.quarters
        self.__years = self.__company_data.years
        self.__max_days = self.__company_data.max_days


    def bayesian_optimization(self, X, Y):
        kernel = C() * RBF()
        search_spaces = {
            'alpha': Real(1e-10, 1e-1, prior='log-uniform'),
            'kernel__k2__length_scale': Real(1e-1, 1e1, prior='log-uniform'),
            'kernel__k1__constant_value': Real(1.0, 1e2, prior='log-uniform')
        }

        history = []  # Used to store the history of optimization

        # This is our callback
        def on_step(optim_result):
            # The optimizer is stored in the 'x' attribute of the result
            score = -optim_result.fun
            history.append(score)
            print("Best score: %s" % score)

        opt = BayesSearchCV(
            GaussianProcessRegressor(kernel=kernel, normalize_y=False),
            search_spaces,
            n_iter=50,  # Number of optimization iterations
            cv=5,  # Cross-validation folds
            n_jobs=-1  # Number of cores to use
        )

        with parallel_backend('threading', n_jobs=1):
            opt.fit(X, Y, callback=on_step)  # Pass the callback

        self.__opt = opt

        # Print the final results
        print("Val. score: %s" % opt.best_score_)
        print("Test score: %s" % opt.score(X, Y))
        print("Best params: %s" % str(opt.best_params_))

        # Plot the optimization history
        plt.figure(figsize=(10, 6))
        plt.plot(history)
        plt.xlabel("Iteration")
        plt.ylabel("Best Score")
        plt.title("Optimization History")
        plt.show()

    def get_eval_model(self, start_year: int, end_year: int):
        years_quarters = list(range(start_year, end_year + 1)) + ['Quarter']
        training_years = years_quarters[:-2]
        df_prices = self.__prices_data[self.__prices_data.columns.intersection(years_quarters)]

        possible_days = list(df_prices.index.values)
        X = np.empty([1, 2], dtype=int)
        Y = np.empty([1], dtype=float)

        first_year_prices = df_prices[start_year]
        if start_year == self.__company_data.years[0]:
            first_year_prices = (first_year_prices[first_year_prices.iloc[:] != 0])
            first_year_prices = (pd.Series([0.0], index=[first_year_prices.index[0] - 1]))._append(first_year_prices)

        first_year_days = list(first_year_prices.index.values)
        first_year_X = np.array([[start_year, day] for day in first_year_days])

        X = first_year_X
        Y = np.array(first_year_prices)
        for current_year in training_years[1:]:
            current_year_prices = list(df_prices.loc[:, current_year])
            current_year_X = np.array([[current_year, day] for day in possible_days])
            X = np.append(X, current_year_X, axis=0)
            Y = np.append(Y, current_year_prices)

        last_year_prices = df_prices[end_year]
        last_year_prices = last_year_prices[last_year_prices.iloc[:].notnull()]

        last_year_days = list(last_year_prices.index.values)
        last_year_X = np.array([[end_year, day] for day in last_year_days])

        X = np.append(X, last_year_X, axis=0)
        Y = np.append(Y, last_year_prices)
        return X, Y

if __name__ == '__main__':
    o = Optimization("APPL")
    X, Y = o.get_eval_model(2019, 2023)
    o.bayesian_optimization(X, Y)
'''    
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from skopt.space import Real
from sklearn.utils import parallel_backend
from skopt import BayesSearchCV
import data_handler
import matplotlib.pyplot as plt

class Optimization:
    __company_name = None
    __company_data = None
    __prices_data = None
    __quarters = None
    __max_days = None
    __opt = None
    __alpha = None
    __start_year = None
    __end_year = None

    def __init__(self, company_name: str):
        self.__company_name = company_name
        self.__company_data = data_handler.CsvHandler(company_name)
        self.__prices_data = self.__company_data.get_equal_length_prices()
        self.__quarters = self.__company_data.quarters
        self.__years = self.__company_data.years
        self.__max_days = self.__company_data.max_days
        self.__start_year = 2019
        self.__end_year = 2023


    def bayesian_optimization(self):
        X, Y = self.get_eval_model()
        kernel = C() * RBF()
        search_spaces = {
            'alpha': Real(1e-10, 1e-1, prior='log-uniform'),
            'kernel__k2__length_scale': Real(1e-1, 1e1, prior='log-uniform'),
            'kernel__k1__constant_value': Real(1.0, 1e2, prior='log-uniform')
        }

        history = []  # Used to store the history of optimization

        # This is our callback
        def on_step(optim_result):
            # The optimizer is stored in the 'x' attribute of the result
            score = -optim_result.fun
            history.append(score)
            print("Best score: %s" % score)

        opt = BayesSearchCV(
            GaussianProcessRegressor(kernel=kernel, normalize_y=False),
            search_spaces,
            n_iter=50,  # Number of optimization iterations
            cv=5,  # Cross-validation folds
            n_jobs=-1  # Number of cores to use
        )

        with parallel_backend('threading', n_jobs=1):
            opt.fit(X, Y, callback=on_step)  # Pass the callback

        self.__opt = opt

        # Print the final results
        print("Val. score: %s" % opt.best_score_)
        print("Test score: %s" % opt.score(X, Y))
        print("Best params: %s" % str(opt.best_params_))

        # Plot the optimization history
        plt.figure(figsize=(10, 6))
        plt.plot(history)
        plt.xlabel("Iteration")
        plt.ylabel("Best Score")
        plt.title("Optimization History")
        plt.show()

    def get_eval_model(self):
        start_year = self.__start_year
        end_year = self.__end_year
        years_quarters = list(range(start_year, end_year + 1)) + ['Quarter']
        training_years = years_quarters[:-2]
        df_prices = self.__prices_data[self.__prices_data.columns.intersection(years_quarters)]

        possible_days = list(df_prices.index.values)
        X = np.empty([1, 2], dtype=int)
        Y = np.empty([1], dtype=float)

        first_year_prices = df_prices[start_year]
        if start_year == self.__company_data.years[0]:
            first_year_prices = (first_year_prices[first_year_prices.iloc[:] != 0])
            first_year_prices = (pd.Series([0.0], index=[first_year_prices.index[0] - 1]))._append(first_year_prices)

        first_year_days = list(first_year_prices.index.values)
        first_year_X = np.array([[start_year, day] for day in first_year_days])

        X = first_year_X
        Y = np.array(first_year_prices)
        for current_year in training_years[1:]:
            current_year_prices = list(df_prices.loc[:, current_year])
            current_year_X = np.array([[current_year, day] for day in possible_days])
            X = np.append(X, current_year_X, axis=0)
            Y = np.append(Y, current_year_prices)

        last_year_prices = df_prices[end_year]
        last_year_prices = last_year_prices[last_year_prices.iloc[:].notnull()]

        last_year_days = list(last_year_prices.index.values)
        last_year_X = np.array([[end_year, day] for day in last_year_days])

        X = np.append(X, last_year_X, axis=0)
        Y = np.append(Y, last_year_prices)
        return X, Y

    def get_best_para(self):
        self.bayesian_optimization()
        print(self.__opt.best_params_)
        kernel = self.__opt.best_params_["kernel__k2__length_scale"] * RBF(length_scale=self.__opt.best_params_["kernel__k2__length_scale"])
        self.__alpha = self.__opt.best_params_["alpha"]
        kernel = 63 * RBF(length_scale=1)
        self.__alpha = 1e-10
        return kernel, self.__alpha



if __name__ == '__main__':
    o = Optimization("APPL")
    o.get_best_para()
'''
