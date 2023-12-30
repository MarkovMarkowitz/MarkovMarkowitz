import warnings
import time
from quantra_reinforcement_learning import reward_exponential_pnl
from quantra_reinforcement_learning import Game
from quantra_reinforcement_learning import init_net
from quantra_reinforcement_learning import ExperienceReplay
import numpy as np
import pickle
from datetime import datetime
warnings.filterwarnings("ignore")


def initialize(context):
    # To show if the handle_data has been run
    context.run = False
    context.episode = 0
    context.cnt = 0
    context.game_over = False

    # Define configuration parameters
    context.rl_config_live = {
                  'LEARNING_RATE': 0.00001,
                  'LOSS_FUNCTION': 'mse',
                  'ACTIVATION_FUN': 'relu',
                  'NUM_ACTIONS': 3,
                  'HIDDEN_MULT': 2,
                  'DISCOUNT_RATE': 0.9,
                  'LKBK': 30,
                  'BATCH_SIZE': 30,
                  'MAX_MEM': 600,
                  'EPSILON': 0.0001,
                  'EPS_MIN': 0.001,  # 0.1% chance to explore the environment
                  'START_IDX': -1,
                  'WEIGHTS_FILE': 'indicator_model.h5',
                  'REPLAY_FILE': 'replay_buffer.bz2',
                  'RF': reward_exponential_pnl,
                  'TEST_MODE': True,
                  'PRELOAD': True,  # Always set true for live
                  'UPDATE_QR': True,
                  'STOCK_SYMBOL': 'CASH,EUR,USD',
                  'TRADE_HOURS': 24,  # 6.25 for NSE , 24 for testing purposes
                  'TRADE_FREQUENCY': 5  # What frequency to run the program at
                  }

    # Define a security for which to run the program
    context.security = symbol(context.rl_config_live['STOCK_SYMBOL'])
    context.epsilon = context.rl_config_live['EPS_MIN']

    bars5m, bars1h, bars1d = get_latest_data(context)

    """---Initialise a NN and a set up initial game parameters---"""
    context.env = Game(bars5m,
                       bars1d,
                       bars1h,
                       context.rl_config_live['RF'],
                       lkbk=context.rl_config_live['LKBK'],
                       init_idx=-1)
    context.q_network, context.r_network = init_net(
                                                    context.env,
                                                    context.rl_config_live
                                                    )

    context.exp_replay = ExperienceReplay(
                                        max_memory=context.rl_config_live['MAX_MEM'],
                                        discount=context.rl_config_live['DISCOUNT_RATE']
                                        )

    context.game_over = True

    """---Preloading the model weights---"""
    print(context.rl_config_live['WEIGHTS_FILE'])
    if context.rl_config_live['PRELOAD']:
        try:
            context.q_network.load_weights(context.rl_config_live['WEIGHTS_FILE'])
            context.r_network.load_weights(context.rl_config_live['WEIGHTS_FILE'])
            context.exp_replay.memory = pickle.load(open(context.rl_config_live['REPLAY_FILE'], 'rb'))
        except:
            print("Unable to pre-load the network files. Please place them in the same folder as RUN_ME.py")
            end()

    context.r_network.set_weights(context.q_network.get_weights())

    total_minutes = context.rl_config_live['TRADE_HOURS']*60

    # Scheduling the strategy
    for i in range(0, total_minutes+1):
        if i % context.rl_config_live['TRADE_FREQUENCY'] == 0:
            schedule_function(
                                strategy,
                                date_rules.every_day(),
                                time_rules.market_open(minutes=i)
                            )

    print("#### INITIALIZATION COMPLETE. STRATEGY WILL RUN AT 5 MIN MARK ####")


def get_latest_data(context):
    ohlcv_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    bars5m = request_historical_data(
                                    context.security,
                                    '5 mins',
                                    '{} D'.format(context.rl_config_live['LKBK'])
                                    )
    bars1h = bars5m.resample('1H', label='right', closed='right').agg(ohlcv_dict).dropna()
    bars1d = bars1h.resample('1D', label='right', closed='right').agg(ohlcv_dict).dropna()
    return bars5m , bars1h , bars1d


def strategy(context, data):
    print("Running strategy at {}".format(datetime.now()))
    
    if context.game_over:
        context.episode += 1

        """---Initialise a new game---"""
        bars5m, bars1h, bars1d = get_latest_data(context)
        context.env = Game(bars5m,
                           bars1d,
                           bars1h,
                           context.rl_config_live['RF'],
                           lkbk=context.rl_config_live['LKBK'],
                           init_idx=-1)

        """---Calculate epsilon for exploration vs exploitation random action generator---"""
        context.game_over = False
        context.cnt = 0

        """---Walk through time steps starting from the end of the last game---"""
    if not context.game_over:
        context.cnt += 1
        state_t = context.env.get_state()

        """---Generate a random action or through q_network---"""
        if np.random.rand() <= context.epsilon:
            action = np.random.randint(0, 3, size=1)[0]
        else:
            q = context.q_network.predict(state_t)
            action = np.argmax(q[0])
            
        actions_dict = {0:"Hold action",
                        1:"Sell action",
                        2:"Buy action"}
        
        print("{} encountered".format(actions_dict[action]))

        """---Updating the Game---"""
        reward, context.game_over = context.env.act(action)

        # Placing Orders
        if context.game_over:
            print("Trade over. Exiting Position")
            order_target(context.security, 0, style = MarketOrder ( ) )

        elif context.env.position == 1:
            print("Maintaining a long position")
            order_target(context.security, 20000, style = MarketOrder ( ) )

        elif context.env.position == -1:
            print("Maintaining a short position")
            order_target(context.security, -20000, style = MarketOrder ( ) )

        # Sleeping before the next run of Strategy
        time.sleep(context.rl_config_live['TRADE_FREQUENCY']*60 - 20)  # -20 to allow the NN to run

        context.env.bars5m, context.env.bars1h, context.env.bars1d = get_latest_data(context)
        state_tp1 = context.env.get_state()

        """---Adding state to memory---"""
        context.exp_replay.remember(
            [state_t, action, reward, state_tp1], context.game_over)

        """---Creating a new Q-Table---"""

        inputs, targets = context.exp_replay.process(
            context.q_network,
            context.r_network,
            batch_size=context.rl_config_live['BATCH_SIZE'])


        """---Update the NN model with a new Q-Table"""
        context.q_network.train_on_batch(inputs, targets)

        if context.game_over and context.rl_config_live['UPDATE_QR']:
            context.r_network.set_weights(context.q_network.get_weights())

    print("Strategy run ended at {}".format(datetime.now()))

    if context.game_over:
        if not context.episode % 10:
            print('----saving weights, trade logs and replay buffer-----')
            context.r_network.save_weights(context.rl_config_live['WEIGHTS_FILE'], overwrite=True)
            pickle.dump(context.exp_replay.memory, open(context.rl_config_live['REPLAY_FILE'], 'wb'))


def handle_data(context, data):
    pass
