"""

Deep Reinforcement Learning Quantra Module

Copyright: Quantra by QuantInsti

"""

import pandas as pd
import numpy as np
from datetime import timedelta
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import talib
import pickle
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
""" REWARD FUNCTIONS"""


def get_pnl(entry, curr, pos):
    # Transaction cost and commissions
    tc = 0.001
    return (curr*(1-tc) - entry*(1+tc))/entry*(1+tc)*pos


def reward_pos_log_pnl(entry, curr, pos):
    """positive log categorical"""
    pnl = get_pnl(entry, curr, pos)
    if pnl >= 0:
        return np.ceil(np.log(pnl*100+1))
    else:
        return 0


def reward_pure_pnl(entry, curr, pos):
    '''pure pnl'''
    return get_pnl(entry, curr, pos)


def reward_positive_pnl(entry, curr, pos):
    '''positive pnl, zero otherwise'''
    pnl = get_pnl(entry, curr, pos)
    if pnl >= 0:
        return pnl
    else:
        return 0


def reward_categorical_pnl(entry, curr, pos):
    '''Sign of pnl'''
    return np.sign(get_pnl(entry, curr, pos))


def reward_positive_categorical_pnl(entry, curr, pos):
    '''1 for win, 0 for loss'''
    pnl = get_pnl(entry, curr, pos)
    if pnl >= 0:
        return 1
    else:
        return 0


def reward_exponential_pnl(entry, curr, pos):
    '''exp pure pnl'''
    return np.exp(get_pnl(entry, curr, pos))


class Game(object):

    def __init__(self, bars5m, bars1d, bars1h, reward_function, lkbk=20, init_idx=None):
        self.bars5m = bars5m
        self.lkbk = lkbk
        self.trade_len = 0
        self.stop_pnl = None
        self.bars1d = bars1d
        self.bars1h = bars1h
        self.is_over = False
        self.reward = 0
        self.pnl_sum = 0
        self.init_idx = init_idx
        self.reward_function = reward_function
        self.reset()

    def _update_position(self, action):
        '''This is where we update our position'''
        if action == 0:
            pass

        elif action == 2:
            """---Enter a long or exit a short position---"""

            # If current position (buy) same as the action (buy), do nothing
            if self.position == 1:
                pass

            # If there is no current position, we update the position to indicate buy
            elif self.position == 0:
                self.position = 1
                self.entry = self.curr_price
                self.start_idx = self.curr_idx

            # If action is different than current position, we end the game and get rewards & trade duration
            elif self.position == -1:
                self.is_over = True

        elif action == 1:
            """---Enter a short or exit a long position---"""
            if self.position == -1:
                pass

            elif self.position == 0:
                self.position = -1
                self.entry = self.curr_price
                self.start_idx = self.curr_idx

            elif self.position == 1:
                self.is_over = True

    def _assemble_state(self):
        '''Here we can add secondary features such as indicators and times to our current state.
        First, we create candlesticks for different bar sizes of 5mins, 1hr and 1d.
        We then add some state variables such as time of day, day of week and position.
        Next, several indicators are added and subsequently z-scored.
        '''

        """---Initializing State Variables---"""
        self.state = np.array([])

        self._get_last_N_timebars()

        """"---Adding Normalised Candlesticks---"""

        def _get_normalised_bars_array(bars):
            bars = bars.iloc[-10:, :-1].values.flatten()
            """Normalizing candlesticks"""
            bars = (bars-np.mean(bars))/np.std(bars)
            return bars

        self.state = np.append(
            self.state, _get_normalised_bars_array(self.last5m))
        self.state = np.append(
            self.state, _get_normalised_bars_array(self.last1h))
        self.state = np.append(
            self.state, _get_normalised_bars_array(self.last1d))

        """---Adding Techincal Indicators---"""

        def _get_technical_indicators(bars):
            # Create an array to store the value of indicators
            tech_ind = np.array([])

            """Relative difference two moving averages"""
            sma1 = talib.SMA(bars['close'], self.lkbk-1)[-1]
            sma2 = talib.SMA(bars['close'], self.lkbk-8)[-1]
            tech_ind = np.append(tech_ind, (sma1-sma2)/sma2)

            """Relative Strength Index"""
            tech_ind = np.append(tech_ind, talib.RSI(
                bars['close'], self.lkbk-1)[-1])

            """Momentum"""
            tech_ind = np.append(tech_ind, talib.MOM(
                bars['close'], self.lkbk-1)[-1])

            """Balance of Power"""
            tech_ind = np.append(tech_ind, talib.BOP(bars['open'],
                                                     bars['high'],
                                                     bars['low'],
                                                     bars['close'])[-1])

            """Aroon Oscillator"""
            tech_ind = np.append(tech_ind, talib.AROONOSC(bars['high'],
                                                          bars['low'],
                                                          self.lkbk-3)[-1])
            return tech_ind

        self.state = np.append(
            self.state, _get_technical_indicators(self.last5m))
        self.state = np.append(
            self.state, _get_technical_indicators(self.last1h))
        self.state = np.append(
            self.state, _get_technical_indicators(self.last1d))

        """---Adding Time Signature---"""
        self.curr_time = self.bars5m.index[self.curr_idx]
        tm_lst = list(map(float, str(self.curr_time.time()).split(':')[:2]))
        self._time_of_day = (tm_lst[0]*60 + tm_lst[1])/(24*60)
        self._day_of_week = self.curr_time.weekday()/6
        self.state = np.append(self.state, self._time_of_day)
        self.state = np.append(self.state, self._day_of_week)

        """---Adding Position---"""
        self.state = np.append(self.state, self.position)

    def _get_last_N_timebars(self):
        '''This function gets the timebars for the 5m, 1hr and 1d resolution based
        on the lookback we've specified.
        '''
        wdw5m = 9
        wdw1h = np.ceil(self.lkbk*15/24.)
        wdw1d = np.ceil(self.lkbk*15)

        """---Getting candlesticks before current time---"""
        self.last5m = self.bars5m[self.curr_time -
                                  timedelta(wdw5m):self.curr_time].iloc[-self.lkbk:]
        self.last1h = self.bars1h[self.curr_time -
                                  timedelta(wdw1h):self.curr_time].iloc[-self.lkbk:]
        self.last1d = self.bars1d[self.curr_time -
                                  timedelta(wdw1d):self.curr_time].iloc[-self.lkbk:]

    def _get_reward(self):
        """Here we calculate the reward when the game is finished.
        Reward function design is very difficult and can significantly
        impact the performance of our algo.
        In this case, we use a simple pnl reward but it is conceivable to use
        other metrics such as Sharpe ratio, average return, etc.
        """
        if self.is_over:
            self.reward = self.reward_function(
                self.entry, self.curr_price, self.position)

    def get_state(self):
        """This function returns the state of the system.
        Returns:
            self.state: the state including indicators, position and times.
        """
        # Assemble new state
        self._assemble_state()
        return np.array([self.state])

    def act(self, action):
        """This function updates the state based on an action
        that was calculated by the NN.
        This is the point where the game interacts with the trading
        algo.
        """

        self.curr_time = self.bars5m.index[self.curr_idx]
        self.curr_price = self.bars5m['close'][self.curr_idx]

        self._update_position(action)

        # Unrealized or realized pnl. This is different from pnl in reward method which is only realized pnl.
        self.pnl = (-self.entry + self.curr_price)*self.position/self.entry

        self._get_reward()
        if self.is_over:
            self.trade_len = self.curr_idx - self.start_idx

        return self.reward, self.is_over

    def reset(self):
        """Resetting the system for each new trading game.
        Here, we also resample the bars for 1h and 1d.
        Ideally, we should do this on every update but this will take very long.
        """
        self.pnl = 0
        self.entry = 0
        self._time_of_day = 0
        self._day_of_week = 0
        self.curr_idx = self.init_idx
        self.t_in_secs = (
            self.bars5m.index[-1]-self.bars5m.index[0]).total_seconds()
        self.start_idx = self.curr_idx
        self.curr_time = self.bars5m.index[self.curr_idx]
        self._get_last_N_timebars()
        self.position = 0
        self.act(0)
        self.state = []
        self._assemble_state()


def init_net(env, rl_config):
    """
    This initialises the RL run by
    creating two new predictive neural network
    Args:
        env:
    Returns:
        modelQ: the neural network
        modelR: the neural network

    """
    hidden_size = len(env.state)*rl_config['HIDDEN_MULT']
    modelQ = Sequential()
    modelQ.add(Dense(len(env.state), input_shape=(
        len(env.state),), activation=rl_config['ACTIVATION_FUN']))
    modelQ.add(Dense(hidden_size, activation=rl_config['ACTIVATION_FUN']))
    modelQ.add(Dense(rl_config['NUM_ACTIONS'], activation='softmax'))
    modelQ.compile(SGD(lr=rl_config['LEARNING_RATE']), loss=rl_config['LOSS_FUNCTION'])

    modelR = Sequential()
    modelR.add(Dense(len(env.state), input_shape=(
        len(env.state),), activation=rl_config['ACTIVATION_FUN']))
    modelR.add(Dense(hidden_size, activation=rl_config['ACTIVATION_FUN']))
    modelR.add(Dense(rl_config['NUM_ACTIONS'], activation='softmax'))
    modelR.compile(SGD(lr=rl_config['LEARNING_RATE']), loss=rl_config['LOSS_FUNCTION'])

    return modelQ, modelR


class ExperienceReplay(object):
    '''This class calculates the Q-Table.
    It gathers memory from previous experience and 
    creates a Q-Table with states and rewards for each
    action using the NN. At the end of the game the reward
    is calculated from the reward function. 
    The weights in the NN are constantly updated with each new
    batch of experience. 
    This is the heart of the RL algorithm.
    Args:
        state_tp1: the state at time t+1
        state_t: the state at time t
        action_t: int {0..2} hold, sell, buy taken at state_t 
        Q_sa: float, the reward for state_tp1
        reward_t: the reward for state_t
        self.memory: list of state_t, action_t and reward_t at time t as well as state_tp1
        targets: array(float) Nx2, weight of each action
        inputs: an array with scrambled states at different times
        targets: Nx3 array of weights for each action for scrambled input states
    '''

    def __init__(self, max_memory, discount):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        '''Add states to time t and t+1 as well as  to memory'''
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def process(self, modelQ, modelR, batch_size=10):
        len_memory = len(self.memory)
        num_actions = modelQ.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]

        """---Initialise input and target arrays---"""
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))

        """Step randomly through different places in the memory
        and scramble them into a new input array (inputs) with the
        length of the pre-defined batch size"""

        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            """Obtain the parameters for Bellman from memory,
            S.A.R.S: state, action, reward, new state."""
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]
            inputs[i] = state_t

            """---Calculate the targets for the state at time t---"""
            targets[i] = modelR.predict(state_t)[0]

            """---Calculate the reward at time t+1 for action at time t---"""
            Q_sa = np.max(modelQ.predict(state_tp1)[0])

            if game_over:
                """---When game is over we have a definite reward---"""
                targets[i, action_t] = reward_t
            else:
                """
                ---Update the part of the target for which action_t occured to new value---
                Q_new(s,a) = reward_t + gamma * max_a' Q(s', a')
                """
                targets[i, action_t] = reward_t + self.discount * Q_sa

        return inputs, targets


def run(bars5m, rl_config):
    """
    Function to run the RL model on the passed price data
    """
    
    pnls = []
    trade_logs = pd.DataFrame()
    episode = 0

    ohlcv_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    bars1h = bars5m.resample('1H', label='left', closed='right').agg(ohlcv_dict).dropna()
    bars1d = bars1h.resample('1D', label='left', closed='right').agg(ohlcv_dict).dropna()

    """---Initialise a NN and a set up initial game parameters---"""
    env = Game(bars5m, bars1d, bars1h, rl_config['RF'],
               lkbk=rl_config['LKBK'], init_idx=rl_config['START_IDX'])
    q_network, r_network = init_net(env, rl_config)
    exp_replay = ExperienceReplay(max_memory=rl_config['MAX_MEM'], discount=rl_config['DISCOUNT_RATE'])

    """---Preloading the model weights---"""
    if rl_config['PRELOAD']:
        q_network.load_weights(rl_config['WEIGHTS_FILE'])
        r_network.load_weights(rl_config['WEIGHTS_FILE'])
        exp_replay.memory = pickle.load(open(rl_config['REPLAY_FILE'], 'rb'))

    r_network.set_weights(q_network.get_weights())

    """---Loop that steps through one trade (game) at a time---"""
    while True:
        """---Stop the algo when end is near to avoid exception---"""
        if env.curr_idx >= len(bars5m)-1:
            break

        episode += 1

        """---Initialise a new game---"""
        env = Game(bars5m, bars1d, bars1h, rl_config['RF'],
                   lkbk=rl_config['LKBK'], init_idx=env.curr_idx)
        state_tp1 = env.get_state()

        """---Calculate epsilon for exploration vs exploitation random action generator---"""
        epsilon = rl_config['EPSILON']**(np.log10(episode))+rl_config['EPS_MIN']

        game_over = False
        cnt = 0

        """---Walk through time steps starting from the end of the last game---"""
        while not game_over:
        
            if env.curr_idx >= len(bars5m)-1:
                break

            cnt += 1
            state_t = state_tp1

            """---Generate a random action or through q_network---"""
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, 3, size=1)[0]

            else:
                q = q_network.predict(state_t)
                action = np.argmax(q[0])

            """---Updating the Game---"""
            reward, game_over = env.act(action)

            """---Updating trade/position logs---"""
            tl = [[env.curr_time, env.position, episode]]
            if game_over:
                tl = [[env.curr_time, 0, episode]]
            trade_logs = trade_logs.append(tl)

            """---Move to next time step---"""
            env.curr_idx += 1
            state_tp1 = env.get_state()

            """---Adding state to memory---"""
            exp_replay.remember(
                [state_t, action, reward, state_tp1], game_over)

            """---Creating a new Q-Table---"""
            inputs, targets = exp_replay.process(
                q_network, r_network, batch_size=rl_config['BATCH_SIZE'])
            env.pnl_sum = sum(pnls)

            """---Update the NN model with a new Q-Table"""
            q_network.train_on_batch(inputs, targets)

            if game_over and rl_config['UPDATE_QR']:
                r_network.set_weights(q_network.get_weights())

        pnls.append(env.pnl)

        print("Trade {:03d} | pos {} | len {} | approx cum ret {:,.2f}% | trade ret {:,.2f}% | eps {:,.4f} | {} | {}".format(
            episode, env.position, env.trade_len, sum(pnls)*100, env.pnl*100, epsilon, env.curr_time, env.curr_idx))

        if not episode % 10:
            print('----saving weights, trade logs and replay buffer-----')
            r_network.save_weights(rl_config['WEIGHTS_FILE'], overwrite=True)
            trade_logs.to_pickle(rl_config['TRADE_FILE'])
            pickle.dump(exp_replay.memory, open(rl_config['REPLAY_FILE'], 'wb'))

        if not episode % 7 and rl_config['TEST_MODE']:
            print('\n**********************************************\nTest mode is on due to resource constraints and therefore stopped after 7 trades. \nYou can trade on full dataset on your local computer and set TEST_MODE flag to False in rl_config dictionary. \nThe full code file, quantra_reinforemcent_learning module and data file is available in last unit of the course.\n**********************************************\n')
            break

    if not rl_config['TEST_MODE']:
        print('----saving weights, trade logs and replay buffer-----')
        r_network.save_weights(rl_config['WEIGHTS_FILE'], overwrite=True)
        trade_logs.to_pickle(rl_config['TRADE_FILE'])
        pickle.dump(exp_replay.memory, open(rl_config['REPLAY_FILE'], 'wb'))

    print('***FINISHED***')


def drawdown_metrics(trade_analytics, chart_title):
    """
    The drawdown metrics and plotting is performed here
    """
    
    # Calculate the cumulative returns
    cum_rets = trade_analytics['rl_strategy_returns_cum']

    # Calculate the running maximum
    running_max = np.maximum.accumulate(cum_rets.dropna())

    # Ensure the value never drops below 1
    running_max[running_max < 1] = 1

    # Calculate the percentage drawdown
    drawdown = (cum_rets)/running_max - 1

    # Calculate the maximum drawdown
    max_dd = drawdown.min()*100

    """
    This part plots the drawdown
    """

    # Define the figure size for the mixed wave plot
    drawdown.plot(figsize=(10, 7), color='r')

    # Fill in-between the drawdown
    plt.fill_between(drawdown.index, drawdown.values, color='red')

    # Add grid to the plot
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.2)

    # Add labels
    plt.ylabel('Drawdown', fontsize=14)
    plt.xlabel('Time', fontsize=14)
    plt.title('Drawdown for '+chart_title, fontsize=16)

    # Define the tick size for x-axis and y-axis
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Display the plot
    plt.show()
    
    # Return the max drawdown
    return max_dd


def trade_analytics(trade_logs_file, price_data, chart_title):
    """
    The analysis of trade performance is done in this function
    """
    
    # Read the trade logs file where we have already run the RL model on the dataset
    trade_logs = pd.read_pickle(trade_logs_file)

    # Rename the column names
    trade_logs.columns = ['Time', 'position', 'trade_num']

    # Set the index to Time
    trade_logs = trade_logs.set_index('Time')

    # Join the trade logs and the price dataframe
    trade_analytics = trade_logs.join(price_data)

    # Calculate the percentage change
    trade_analytics['percent_change'] = trade_analytics.close.pct_change()

    # Calculate the strategy returns
    trade_analytics['rl_strategy_returns'] = trade_analytics['percent_change'] * \
        trade_analytics['position'].shift(1)

    # Calculate the cumulative strategy returns
    trade_analytics['rl_strategy_returns_cum'] = (
        trade_analytics['rl_strategy_returns']+1).cumprod()

    # Define the figure size for the cumulative returns plot
    trade_analytics['rl_strategy_returns_cum'].plot(figsize=(10,7))

    # Add legend to the axis
    plt.legend()

    # Add labels
    plt.ylabel('Cumulative Returns', fontsize=14)
    plt.xlabel('Time', fontsize=14)
    plt.title("Cumulative Returns for "+chart_title, fontsize=16)

    # Define the tick size for x-axis and y-axis
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Display the cumulative returns plot
    plt.show()

    # Calls the drawdown_metrics() to calculate and plot the drawdown
    max_dd = drawdown_metrics(trade_analytics, chart_title)

    # Portfolio returns
    pf_returns = trade_analytics['rl_strategy_returns_cum'].tail(1)-1
    pf_returns_mean = trade_analytics['rl_strategy_returns'].mean()
    portfolio_vol = trade_analytics['rl_strategy_returns'].std()

    # Calculate Sharpe Ratio (78 since we are dealing with 5 minute time step)
    sharpe_ratio = pf_returns_mean/portfolio_vol*((252*78)**0.5)

    # Prints the portfolio returns
    print("The final portfolio return is %.2f" % (pf_returns*100)+"%")

    # Prints the drawdown
    print("The maximum drawdown is %.2f" % max_dd+"%")

    # Prints the Sharpe ratio
    print("The Sharpe ratio is %.2f" % sharpe_ratio)
    
    # Return the trade analytics DataFrame
    return trade_analytics