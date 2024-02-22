from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt

class MLModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def linear_regression(self):
        reg = LinearRegression().fit(self.X_train, self.y_train['target_t1'])
        p_train = reg.predict(self.X_train)
        p_test = reg.predict(self.X_test)

        RMSE_train = np.sqrt(mean_squared_error(self.y_train['target_t1'], p_train))
        RMSE_test = np.sqrt(mean_squared_error(self.y_test['target_t1'], p_test))

        print('Train RMSE: {:.3f}\nTest RMSE: {:.3f}'.format(RMSE_train, RMSE_test))

    def random_forest_regressor(self):
        splits = TimeSeriesSplit(n_splits=3, max_train_size=365 * 2)
        rfr = RandomForestRegressor()

        rfr_grid = {'n_estimators': [500],
                    'max_depth': [3, 5, 10, 20, 30],
                    'max_features': [4, 8, 16, 32, 59]}

        grid_search = GridSearchCV(estimator=rfr, param_grid=rfr_grid, cv=splits, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train['target_t1'])

        best_params = grid_search.best_params_
        best_score = np.sqrt(np.abs(grid_search.best_score_))

        print('Best parameters:', best_params)
        print('Best RMSE (CV): {:.3f}'.format(best_score))

        rfr.set_params(**best_params)
        rfr.fit(self.X_train, self.y_train['target_t1'])
        p_train = rfr.predict(self.X_train)
        p_test = rfr.predict(self.X_test)

        RMSE_train = np.sqrt(mean_squared_error(self.y_train['target_t1'], p_train))
        RMSE_test = np.sqrt(mean_squared_error(self.y_test['target_t1'], p_test))

        print('Train RMSE: {:.3f}\nTest RMSE: {:.3f}'.format(RMSE_train, RMSE_test))

        test_resid_1step = self.y_test['target_t1'] - p_test
        test_MAPE = np.mean(np.abs(test_resid_1step / self.y_test['target_t1'])) * 100
        print('1-step ahead forecasting MAPE:', test_MAPE)

        test_df[['target_t1', 'pred_t1']].plot()
        plt.title('1-period ahead Forecasting')
        plt.ylabel('(MWh)')
        plt.legend()
        plt.show()

        plt.scatter(y=self.y_train['target_t1'], x=p_train, label='train')
        plt.scatter(y=self.y_test['target_t1'], x=p_test, label='test')
        plt.title('1-period ahead Actual vs forecasting ')
        plt.ylabel('Actual')
        plt.xlabel('Forecast')
        plt.legend()
        plt.show()

        self.residual_analysis(test_resid_1step)

    def residual_analysis(self, test_resid_1step):
        test_resid_1step.plot.hist(bins=10, title='Test 1-step ahead residuals distribution')
        plt.xlabel('Residuals')
        plt.show()

        test_resid_1step.plot(title='Test 1-step ahead residuals time series')
        plt.ylabel('Residuals')
        plt.show()

        plt.scatter(x=self.y_test['target_t1'].values, y=test_resid_1step.values)
        plt.title('Test 1-step ahead residuals vs Actual values')
        plt.ylabel('Residuals')
        plt.xlabel('Actual values')
        plt.show()