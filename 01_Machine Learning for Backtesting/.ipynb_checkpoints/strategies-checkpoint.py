from backtesting import Strategy
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

class SimpleClassificationUD(Strategy):
    
    def init(self):
        self.model = DecisionTreeClassifier(max_depth=15, random_state=42)
        self.already_bought = False

    def next(self):
        explanatory_today = self.data.df.iloc[-1:, :]
        forecast_tomorrow = self.model.predict(explanatory_today)[0]
        
        # conditions to sell or buy
        if forecast_tomorrow == 'UP' and self.already_bought == False:
            self.buy()
            self.already_bought = True
        elif forecast_tomorrow == 'DOWN' and self.already_bought == True:
            self.sell()
            self.already_bought = False
        else:
            pass


class SimpleRegression(Strategy):
    # model = DecisionTreeRegressor(max_depth=15, random_state=42)
    model = None
    
    limit_buy = 1
    limit_sell = -5

    def init(self):
        self.already_bought = False

    def next(self):
        explanatory_today = self.data.df.iloc[-1:, :]
        forecast_tomorrow = self.model.predict(explanatory_today)[0]

        # conditions to sell or buy
        if forecast_tomorrow > self.limit_buy and self.already_bought == False:
            self.buy()
            self.already_bought = True
        elif forecast_tomorrow < self.limit_sell and self.already_bought == True:
            self.sell()
            self.already_bought = False
        else:
            pass
        
        
class Regression(Strategy):

    limit_buy = 1
    limit_sell = -5

    N_TRAIN = 600

    def init(self):
        self.model = DecisionTreeRegressor(max_depth=15, random_state=42)
        self.already_bought = False
        
        X_train = self.data.df.iloc[:self.N_TRAIN, :-1]
        y_train = self.data.df.iloc[:self.N_TRAIN, -1]
        
        self.model.fit(X_train, y_train)

    def next(self):
        explanatory_today = self.data.df.iloc[[-1], :-1]
        forecast_tomorrow = self.model.predict(explanatory_today)[0]
        
        # conditions to sell or buy
        if forecast_tomorrow > self.limit_buy and self.already_bought == False:
            self.buy()
            self.already_bought = True
        elif forecast_tomorrow < self.limit_sell and self.already_bought == True:
            self.sell()
            self.already_bought = False
        else:
            pass
        
        
class WalkForward(Regression):
    def next(self):

        if len(self.data) < self.N_TRAIN:
            return # we don't take any action and move on to the following day
        
        if len(self.data) % 200 != 0:
            return super().next()
        
        X_train = self.data.df.iloc[-self.N_TRAIN:, :-1]
        y_train = self.data.df.iloc[-self.N_TRAIN:, -1]

        self.model.fit(X_train, y_train)

        super().next()
        
        
class WalkForwardAnchored(Regression):
    def next(self):

        if len(self.data) < self.N_TRAIN:
            return # we don't take any action and move on to the following day
        
        if len(self.data) % 200 != 0:
            return super().next()
        
        X_train = self.data.df.iloc[:, :-1]
        y_train = self.data.df.iloc[:, -1]

        self.model.fit(X_train, y_train)

        super().next()
        
        
class RegressionAggresiveStopLoss(Strategy):
    model = RandomForestRegressor(max_depth=15, random_state=42)

    limit_buy = 1
    limit_sell = -5

    N_TRAIN = 600
    price_delta = .004
    
    n_days_stop_loss = 2
    size_trades = .2
    
    def init(self):
        
        X_train = self.data.df.iloc[:self.N_TRAIN, :-1]
        y_train = self.data.df.iloc[:self.N_TRAIN, -1]

        self.model.fit(X_train, y_train)

    def next(self):
        explanatory_today = self.data.df.iloc[[-1], :-1]
        forecast_tomorrow = self.model.predict(explanatory_today)[0]

        # Proceed only with out-of-sample data. Prepare some variables
        high, low, close = self.data.High, self.data.Low, self.data.Close
        current_time = self.data.index[-1]

        upper, lower = close[-1] * (1 + np.r_[1, -1]*self.price_delta)

        # conditions to sell or buy
        if forecast_tomorrow > self.limit_buy and not self.position.is_long:
            self.buy(size=self.size_trades, tp=upper, sl=lower)
        elif forecast_tomorrow < self.limit_sell and not self.position.is_short:
            self.sell(size=self.size_trades, tp=lower, sl=upper)
        else:
            pass
        
        
        # Additionally, set aggressive stop-loss on trades that have been open 
        # for more than two days
        for trade in self.trades:
            if current_time - trade.entry_time > pd.Timedelta(f'{self.n_days_stop_loss} days'):
                if trade.is_long:
                    trade.sl = max(trade.sl, low)
                else:
                    trade.sl = min(trade.sl, high)
                    
                    
class WalkForwardAggresiveStopLoss(RegressionAggresiveStopLoss):
    def next(self):

        if len(self.data) < self.N_TRAIN:
            return # we don't take any action and move on to the following day
        
        if len(self.data) % 200 != 0:
            return super().next()
        
        X_train = self.data.df.iloc[-self.N_TRAIN:, :-1]
        y_train = self.data.df.iloc[-self.N_TRAIN:, -1]

        self.model.fit(X_train, y_train)

        super().next()