from backtesting import Strategy
from sklearn.tree import DecisionTreeRegressor

class Regression(Strategy):
    limit_buy = 1
    limit_sell = -5
    
    n_train = 600
    
    def init(self):
        self.model = DecisionTreeRegressor(max_depth=15, random_state=42)
        self.already_bought = False
        
        X_train = self.data.df.iloc[self.n_train,:-1]
        y_train = self.data.df.iloc[self.n_train,-1]
        
        self.model.fit(X=X_train, y=y_train)

    def next(self):
        explanatory_today = self.data.df.iloc[[-1], :]
        forecast_tomorrow = self.model.predict(explanatory_today)[0]
        
        if forecast_tomorrow > self.limit_buy and self.already_bought == False:
            self.buy()
            self.already_bought = True
        elif forecast_tomorrow < self.limit_sell and self.already_bought == True:
            self.sell()
            self.already_bought = False
        else:
            pass

class WalkForwardUnanchored(Regression):
    def next(self):

        if len(self.data) < self.n_train: # 600
            return # we don't take any action and move on to the following day
        
        if len(self.data) % 200 != 0:
            return super().next()
        
        # 800
        X_train = self.data.df.iloc[:, :-1]
        y_train = self.data.df.iloc[:, -1]

        self.model.fit(X_train, y_train)

        super().next()