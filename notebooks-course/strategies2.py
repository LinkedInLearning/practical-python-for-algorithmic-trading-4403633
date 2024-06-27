from backtesting import Strategy
from sklearn.ensemble import RandomForestRegressor

class Regression(Strategy):
  limit_buy = 3
  limit_sell = -3
  already_bought = False
  n_train = 600

  def init(self):
    self.model = RandomForestRegressor(random_state = 42, max_depth = 15)

    self.coef_retrain = 200

    X_train = self.data.df.iloc[:self.n_train, :-1]
    y_train = self.data.df.iloc[:self.n_train, -1]

    self.model.fit(X_train, y_train)

  def next(self):
    explanatory_today = self.data.df.iloc[[-1], :-1]
    forecast_tomorrow = self.model.predict(explanatory_today)[0]

    if(forecast_tomorrow > self.limit_buy and self.already_bought ==False):
      self.buy()
      self.already_bought = True

    elif(forecast_tomorrow < self.limit_sell and self.already_bought ==True):
      self.sell()
      self.already_bought = False
  
    else:
      pass


  
class WalkForwardUnanchored(Regression):
  def next(self):
    if len(self.data) < self.n_train:
      return
    
    if len(self.data) % self.coef_retrain == 0:

      X_train = self.data.df.iloc[-self.n_train:, :-1]
      y_train = self.data.df.iloc[-self.n_train:, -1]

      self.model.fit(X_train, y_train)

      super().next()

    else:
      super().next()