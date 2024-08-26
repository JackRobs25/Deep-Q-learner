import argparse
import glob, os, re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tech_ind import bollinger_bands
from tech_ind import relative_strength_index
from tech_ind import macd
from sklearn.preprocessing import MinMaxScaler
import math
from DeepQLearner import DeepQLearner




class StockEnvironment:

  """
  Anything you need.  Suggestions:  __init__, train_learner, test_learner.

  I wrote train and test to just receive a learner and a dataframe as
  parameters, and train/test that learner on that data.

  You might want to print or accumulate some data as you train/test,
  so you can better understand what is happening.

  Ultimately, what you do is up to you!
  """
  def __init__ (self, fixed = None, floating = None, cash = None, shares = None):
    self.shares = shares
    self.fixed_cost = fixed
    self.floating_cost = floating
    self.starting_cash = cash
    self.learner = None
    self.losses = []

  def prepare_world(self, data, symbol):
    """
    Read the relevant price data
    Return a DataFrame containing everything you need.
    """
    # get a dataframe of time and price from self.df
    world = pd.DataFrame(index=data.index, columns=['Price']) 
    world['Price'] = data['Price']
    min_price = data['Price'].min()
    max_price = data['Price'].max()
    world['Price'] = (world['Price'] - min_price) / (max_price - min_price)
    world['AS1'] = data['AS1']
    world['BS1'] = data['BS1']
    world['AS2'] = data['AS2']
    world['BS2'] = data['BS2']
    world['AS3'] = data['AS3']
    world['BS3'] = data['BS3']
    world['AS4'] = data['AS4']
    world['BS4'] = data['BS4']
    world['AS5'] = data['AS5']
    world['BS5'] = data['BS5']
    world['AS6'] = data['AS6']
    world['BS6'] = data['BS6']
    world['AS7'] = data['AS7']
    world['BS7'] = data['BS7']
    world['AS8'] = data['AS8']
    world['BS8'] = data['BS8']
    world['AS9'] = data['AS9']
    world['BS9'] = data['BS9']
    world['AS10'] = data['AS10']
    world['BS10'] = data['BS10']
    world = world[~world.index.duplicated()]

    return world

  def calc_state (self, world, interval, holdings):
    row = world.loc[interval]
    # ratio = row['BS1']/row['AS1']
    # return np.array([ratio, holdings]).reshape(1,2)
    return np.array([row['Price'], row['AS1'], row['BS1'], row['AS2'], row['BS2'],row['AS3'], row['BS3'],
                     row['AS4'], row['BS4'], row['AS5'], row['BS5'], row['AS6'], row['BS6'],
                     row['AS7'], row['BS7'], row['AS8'], row['BS8'], row['AS9'], row['BS9'],
                     row['AS10'], row['BS10'], int(holdings)]).reshape(1, 22)
  

  def train_learner(self, data, symbol, cutoff, learner, trips=1, dyna = 0,
                    eps = 0.0, eps_decay = 0.0):
    world = self.prepare_world(data, symbol)
    times = data.index[:cutoff]
    trades = pd.DataFrame(index=times)
    trades['Trade'] = 0
    portfolio_value = pd.DataFrame(index=times)
    portfolio_value['Value'] = 0
    cash = self.starting_cash
    holdings = 0
    prev_value = cash
    first = True
    for index, row in world.iterrows(): # for each day in training data
      price = world.loc[index, 'Price']
      curr_value = holdings*price + cash
      curr_state = self.calc_state(world, index, holdings)
      # compute the reward for the previous action
      reward = curr_value - prev_value
      # query learner to get an action
      # first action of every trip selected by test 
      if first:
        action = learner.test(curr_state)
      else:
        action = learner.train(curr_state, reward)
      first = False
      # long = 2
      # short = 1
      # flat = 0
      trade = 0
      share_limit = self.shares
      if action == 2: # buy
        if holdings > 0:
          trade = 0
        elif holdings == 0:
          trade = share_limit
        else:
          trade = share_limit*2
      elif action == 1: # sell
        if holdings > 0:
          trade = -share_limit*2
        elif holdings == 0:
          trade = -share_limit
        else:
          trade = 0
      elif action == 0: # flat
        if holdings > 0:
          trade = -share_limit
        elif holdings == 0:
          trade = 0
        else:
          trade = share_limit

      stock_value = abs(trade * price)
      if(not math.isnan(stock_value)):
        if trade > 0:
          fee = self.fixed_cost + (self.floating_cost * stock_value)
          cash -= fee
          cash -= stock_value
          holdings += trade
          
        if trade < 0:
          fee = self.fixed_cost + (self.floating_cost * stock_value)
          cash -= fee
          cash += stock_value
          holdings += trade

      trades.at[index, 'Trade'] = trade
      portfolio_value.at[index, 'Value'] = curr_value.astype('int64')
      prev_value = curr_value

    self.learner = learner
    self.losses = learner.losses
    # print("len: ", len(self.losses))
    return self.losses
  
  def test_learner(self, data, symbol, cutoff, learner, oos=False):
    """
    Evaluate a trained Q-Learner on a particular stock trading task.

    Only if print_results is True, print a summary result of what happened
    during the test.  Print nothing if print_results is False.

    If printing, feel free to include portfolio stats or other information, but AT LEAST:

    Test trip, net result: $31710.00
    Benchmark result: $6690.0000

    Return a tuple of the test trip net result and the benchmark result, in that order.
    """
    world = self.prepare_world(data, symbol)

    if oos:
        times = data.index[cutoff:]
    else: 
      times = data.index[:cutoff]
    trade_df = pd.DataFrame(index=times, columns=['Trade'])
    spy_df = pd.read_csv('data/SPY.csv', index_col='Date', parse_dates=True, usecols=['Date', 'Adj Close'])
    spy_df.rename(columns={'Adj Close': 'SPY'}, inplace=True)
    learned_trades = trade_df.join(spy_df, how='inner')
    learned_trades.drop('SPY', axis=1, inplace=True)
    learned_trades['Trade'] = 0

    portfolio_value = pd.DataFrame(index=times)
    portfolio_value['Value'] = 0

    cash = self.starting_cash
    holdings = 0


    for index, row in world.iterrows():
      price = world.loc[index, 'Price']
      curr_value = holdings*price + cash
      curr_state = self.calc_state(world, index, holdings)
      action = self.learner.test(curr_state)

      trade = 0
      share_limit = self.shares
      if action == 2: # buy
        if holdings > 0:
          trade = 0
        elif holdings == 0:
          trade = share_limit
        else:
          trade = share_limit*2
      elif action == 1: # sell
        if holdings > 0:
          trade = -share_limit*2
        elif holdings == 0:
          trade = -share_limit
        else:
          trade = 0
      elif action == 0: # flat
        if holdings > 0:
          trade = -share_limit
        elif holdings == 0:
          trade = 0
        else:
          trade = share_limit
      # print("trade: ", trade)

      stock_value = abs(trade * price)
      if(not math.isnan(stock_value)):
        if trade > 0:
          fee = self.fixed_cost + (self.floating_cost * stock_value)
          cash -= fee
          cash -= stock_value
          holdings += trade
          
        if trade < 0:
          fee = self.fixed_cost + (self.floating_cost * stock_value)
          cash -= fee
          cash += stock_value
          holdings += trade
              
      learned_trades.at[index, 'Trade'] = trade
      portfolio_value.at[index, 'Value'] = curr_value.astype('int64')
    test_trip_net_result = portfolio_value.tail(1)

    
    return test_trip_net_result
      


if __name__ == '__main__':
  # Train one Q-Learning agent for each stock in the data directory.
  # Each one will use all days in ascending order, with a train and
  # test period each day.  Each agent is NOT reset between days.
  # It is totally fine to just use one stock, and just one agent.
  # Or one agent to trade all the stocks.  Or whatever, really.

  ### Command line argument parsing.

  parser = argparse.ArgumentParser(description='Stock environment for Deep Q-Learning.')

  sim_args = parser.add_argument_group('simulation arguments')
  sim_args.add_argument('--cash', default=10000000, type=float, help='Starting cash for the agent.')
  sim_args.add_argument('--fixed', default=0.00, type=float, help='Fixed transaction cost.')
  sim_args.add_argument('--floating', default='0.00', type=float, help='Floating transaction cost.')
  sim_args.add_argument('--shares', default=1000, type=int, help='Number of shares to trade (also position limit).')
  sim_args.add_argument('--trials', default=1, type=int, help='Number of complete experimental trials.')
  sim_args.add_argument('--trips', default=1, type=int, help='Number of training trips per stock-day.')

  args = parser.parse_args()

  # Store the final in-sample and out-of-sample result of each trial.
  is_results = []
  oos_results = []

  ### HFT data file reading.

  # Read the data only once.  It's big!
  csv_files = glob.glob(os.path.join(".", "data", "hft_data", "INTC", "INTC_2024-03-04*_message_*.csv"))
  # csv_files = glob.glob(os.path.join(".", "data", "hft_data", "*", "*_message_*.csv"))
  date_str = re.compile(r'_(\d{4}-\d{2}-\d{2})_')
  stock_str = re.compile(r'([A-Z]+)_\d{4}-\d{2}-\d{2}_')

  df_list = []
  day_list = []
  sym_list = []

  for csv_file in sorted(csv_files):
    date = date_str.search(csv_file)
    date = date.group(1)
    day_list.append(date)

    symbol = stock_str.search(csv_file)
    symbol = symbol.group(1)
    sym_list.append(symbol)

    # Find the order book file that matches this message file.
    book_file = csv_file.replace("message", "orderbook")

    # Read the message file and index by timestamp.
    df = pd.read_csv(csv_file, names=['Time','EventType','OrderID','Size','Price','Direction'])
    df['Time'] = pd.to_datetime(date) + pd.to_timedelta(df['Time'], unit='s')

    # print(df.head())
    names = [f"{x}{i}" for i in range(1,11) for x in ["AP","AS","BP","BS"]]
    # print(pd.read_csv(book_file,names=names).head())
    # Read the order book file and merge it with the messages.
    names = [f"{x}{i}" for i in range(1,11) for x in ["AP","AS","BP","BS"]]
    df = df.join(pd.read_csv(book_file, names=names, nrows=10000), how='inner') # remove nrows when testing full thing
    df = df.set_index(['Time'])

    BBID_COL = df.columns.get_loc("BP1")       # ADD THESE TWO LINES AT THE END OF THE DATA LOADING.
    BASK_COL = df.columns.get_loc("AP1")       # THEY GET THE COLUMN NUMBER FOR BEST BID/ASK BY NAME.

    print (f"Read {df.shape[0]} unique order book shapshots from {csv_file}")

    df_list.append(df)

  days = len(day_list)
#   print("df: ", df_list)
#   print("day: ", day_list)
#   print("sym: ", sym_list)

  ### Benchmark computation.

  # Compute once per day for later use.
  is_brets = []   # IS  period benchmark return
  oos_brets = []  # OOS period benchmark return

  # Prepare to receive IS and OOS cumulative returns per day,
  # potentially multiple trials.
  is_cr = [ [] for i in range(days) ]
  oos_cr = [ [] for i in range(days) ]


  ### The big learning loop.

  # Run potentially many experiments.
  for trial in range(args.trials):

    # Create an instance of the environment class.
    env = StockEnvironment(fixed = args.fixed, floating = args.floating, cash = args.cash, shares = args.shares)   # TO DO: your parameters.
    # My approach: train and test on a part of each day.
    # You do what you want to!
    for day in range(days):
      cutoff_row = int(df_list[day].shape[0]*0.5)
      data = df_list[day]
      symbol = sym_list[day]

      # You might need to make a learner around here...
      # But again, it depends on your approach and strategy.

      if (len(is_brets) <= day):
        # Compute benchmark cumulative returns once per day only.
        # This assumes you defined some "cutoff row" where the training
        # stops for the day and the testing starts.  If you did something
        # different, you will need to alter this.

        is_start_mid = (data.iloc[0,BASK_COL] + data.iloc[0,BBID_COL]) / 2
        oos_start_mid = (data.iloc[cutoff_row,BASK_COL] + data.iloc[cutoff_row,BBID_COL]) / 2
        oos_end_mid = (data.iloc[-1,BASK_COL] + data.iloc[-1,BBID_COL]) / 2

        is_brets.append((oos_start_mid / is_start_mid) - 1.0)
        oos_brets.append((oos_end_mid / oos_start_mid) - 1.0)

      learner = DeepQLearner(update_interval=100, num_features=22)

      # We might do multiple trips through the data for training one stock-day.
      # I did them all at once.  Up to you what you want to do!
      for trip in range(args.trips):

        print (f"Training {symbol}, {day_list[day]}: Trip {trip}")

        # Probably call your env.train_learner around here to train on one day for one trip.
        # Up to you.
        losses = env.train_learner(data, symbol, cutoff_row, learner, trips=args.trips) 


        # Here, I drew a nice, updating plot of the loss per "trip" to ensure my learner
        # was moving in the right direction.  This required exposing/returning some
        # accumulated loss values from my learner object.
        # print("losses: ", losses)

      #   plt.figure(1)
      #   plt.clf()
      #   plt.plot(losses)
      #   plt.show()                
      #   # plt.pause(0.01)

      # plt.figure(1)
      # plt.clf()
      # plt.plot(losses)
      # plt.show()                
      # Now test the learned policy and see how it does.

      # In sample.
      print (f"In-sample {symbol}: {day_list[day]}")
      is_cr[day].append(env.test_learner(data, symbol, cutoff_row, learner))   # Call your env.test_learner on the in-sample data you trained on.

      # Out of sample.
      print (f"Out-of-sample {symbol}: {day_list[day]}")
      oos_cr[day].append(env.test_learner(data, symbol, cutoff_row, learner, oos=True))  # Call your env.test_learner on some out-of-sample data.


  ### Print final summary stats.

  is_cr = ((np.array(is_cr)) / args.cash) - 1
  oos_cr = ((np.array(oos_cr)) / args.cash) - 1
  # is_cr = np.array(is_cr)
  # oos_cr = np.array(oos_cr)
  


  # Print summary results.
  print ()
  print (f"In-sample per-symbol per-day min, median, mean, max results across all {args.trials} trials")
  for day in range(days):
    print(f"IS {sym_list[day]} {day_list[day]}: {np.min(is_cr[day]):.4f}, {np.median(is_cr[day]):.4f}, \
          {np.mean(is_cr[day]):.4f}, {np.max(is_cr[day]):.4f} vs long benchmark {is_brets[day]:.4f}")

  print ()
  print (f"Out-of-sample per-symbol per-day min, median, mean, max results across all {args.trials} trials")
  for day in range(days):
    print(f"OOS {sym_list[day]} {day_list[day]}: {np.min(oos_cr[day]):.4f}, {np.median(oos_cr[day]):.4f}, \
          {np.mean(oos_cr[day]):.4f}, {np.max(oos_cr[day]):.4f} vs long benchmark {oos_brets[day]:.4f}")


