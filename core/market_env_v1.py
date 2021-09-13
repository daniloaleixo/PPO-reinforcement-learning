import numpy as np
from collections import deque
import pandas as pd
import math

from .reward_functions import PercentChange, SimpleProfit, RiskAdjustedReturns

class MarketEnvironmentV1():
    '''
    Simple market environment, in which the agent could only buy or sell
    '''
    def __init__(self, x, actions, state_size, 
                start_index=0, 
                end_index=10_000,
                undo_detrend=[], 
                close_col=0, 
                profit_window_size=10):
        self.start_index = start_index
        self.end_index = end_index
        self.x = x[self.start_index:self.end_index]
        self.undo_detrend = undo_detrend[self.start_index: self.end_index]
        self.detrended = len(undo_detrend) > 0
        self.num_steps = self.x.shape[0]

        self.close_col = close_col


        # Actions
        self.actions = actions
        
        # State size
        self.state_size = state_size

        self.profit_window_size = profit_window_size

        self.reset()


    def get_action_size(self):
        return len(self.actions)

    def get_state_size(self):
        return self.state_size

    def step(self, action, debug=False):
        '''
        Take one step at the environment
        '''
        if debug: print('-' * 50)
        state = self.get_state(self.step_value)
        if debug: print(">> State: ", state)
        reward = self.make_action(state, action, debug)
        done = self.check_done()
        
        self.step_value += 1
        next_state = self.get_state(self.step_value)

        return next_state, reward, done, None











    #
    # HELPERS
    #
    def get_info(self):
      '''
      Log all the info from the environment
      '''
      return { 
          "holdings": str(self.holdings), 
          "step_value": str(self.step_value),
      }

    def get_state(self, step=0):
        state = np.array(self.x[step])
        return state

    def next_state(self):
        return self.get_state(self.step_value + 1)


    def reset(self):
        """
        Resets the game, clears the state buffer.
        """


        # Params
        self.holdings = 0
        self.step_value = 0
        self.last_close_prices_vs_position = deque(maxlen=self.profit_window_size)

        return self.get_state(self.step_value)
        

    def make_action(self, state, action, debug=False):
        '''
        Perform an action and change the state
        '''
        close = self.get_last_close()
        if debug: print(">> Close: ", close)

        # Sit
        if(action == 0):
            if debug: print("Sitting...")

        # Buy
        elif (action == 1):
          self.holdings = 1

        # Short
        elif(action == 2):
            self.holdings = -1

        # Update portfolio info
        self._update_portfolio(close)

        reward = self._get_reward()

        if debug: print("Reward: ", reward)

        return reward


    def get_last_close(self):
        if self.detrended: return self.undo_detrend[self.step_value][self.close_col]
        return self.x[self.step_value][self.close_col]

    def _update_portfolio(self, close):
        '''
        Update our portfolio value (essential to calculate the reward)
        '''
        position = close * self.holdings

        self.last_close_prices_vs_position.append(position)

        if len(self.last_close_prices_vs_position) > self.profit_window_size:
            self.last_close_prices_vs_position.popleft()

    def _get_reward(self):
      if len(self.last_close_prices_vs_position) <= 1: return 0

      total = 0
      last_close = self.last_close_prices_vs_position[0]
      for i in range(1, len(self.last_close_prices_vs_position)):
        next_close = self.last_close_prices_vs_position[i]

        if last_close > 0 and next_close > 0:  
          total = total + (abs(next_close) - abs(last_close)) 
        elif last_close < 0 and next_close < 0:  
          total = total + (abs(next_close) - abs(last_close)) * -1 
        elif last_close < 0 and next_close > 0:  
          total = total + (abs(next_close) - abs(last_close)) * -1 
        elif last_close > 0 and next_close < 0:  
          total = total + (abs(next_close) - abs(last_close)) 

        last_close = next_close

      return total

    def close(self):
      return None

    def check_done(self, debug=False):
      ''' 
      Check if we get to the end of the run
      '''
      return self.step_value == self.num_steps - 2
