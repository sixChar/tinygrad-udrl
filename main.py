import numpy as np
from tinygrad.tensor import Tensor
import tinygrad.optim as optim
import gym


'''
    Feedforward neural net with softmax or no output activation function.
'''
class FFNet:
  def __init__(self, num_ins, hid_shapes, num_outs, out_func=True):
    self.Ws = [Tensor.uniform(num_ins, hid_shapes[0])]
    self.bs = [Tensor.uniform(1, hid_shapes[0])]

    for hid_in, hid_out in zip(hid_shapes[:-1], hid_shapes[1:]):
      self.Ws.append(Tensor.uniform(hid_in, hid_out))
      self.bs.append(Tensor.uniform(1, hid_out))

    self.Ws.append(Tensor.uniform(hid_shapes[-1], num_outs))
    self.bs.append(Tensor.uniform(1, num_outs))
    
    self.out_func = out_func

  def forward(self, x):
    for W,b in zip(self.Ws[:-1], self.bs[:-1]):
      x = x.dot(W).add(b).relu()

    y_no_act = x.dot(self.Ws[-1]).add(self.bs[-1])

    if self.out_func:
      return y_no_act.softmax()
    else:
      return y_no_act

  def get_vars(self):
    return self.Ws + self.bs


'''
  Memory of episodes from which to draw training data from.
'''
class Memory:
  def __init__(self, size):
    self.max_size = size
    self.data = []

  '''
    Add a tuple of a given state, the action taken, the reward, and the resulting state.
  '''
  def add(self, s, a, r, s_):
    self.data.append((s,a,r,s_))
    if len(self.data) > self.max_size:
      self.data = self.data[-self.max_size:]
  
  '''
    Get the last n state, action, reward, next state tuples. 
  '''
  def get_last_n(self, n):
    n_s = []
    n_a = []
    n_r = []
    n_s_ = []

    for (s,a,r,s_) in self.data[-n:]:
      n_s.append(s)
      n_a.append(a)
      n_r.append(r)
      n_s_.append(s_)

    return np.asarray(n_s), np.asarray(n_a), np.asarray(n_r), np.asarray(n_s_)  

  '''
    Randomly sample a sequential run of n tuples from the data.
  '''
  def sample_n(self, n):
    start = np.random.randint(0,len(self.data)-n)
    n_s, n_a, n_r, n_s_ = list(zip(*self.data[start:start+n]))
    return np.array(n_s), np.array(n_a), np.array(n_r), np.array(n_s_)

  def get_length(self):
    return len(self.data)

'''
  An upside down reinforcement learning agent.
'''
class UDRLAgent:
  def __init__(self, num_obsv, num_act, model_mem_size=3, memory_size=10000, init_best_score=-100, init_best_time=1):
    self.num_act = num_act
    self.num_obsv = num_obsv
    self.model_mem_size = model_mem_size    

    # input contains past *model_mem_size* number of observations + desired future reward + future 
    # time horizon 
    self.model = FFNet(
                    num_obsv * model_mem_size + 2,
                    [64,128,64],
                    num_act,
                    out_func=True
                 )
    self.optim = optim.Adam(self.model.get_vars(), lr=1e-3)

    self.memory = Memory(memory_size)

    # Best score so far as well as the length of time that score was accumulated over.
    self.best_score = init_best_score
    self.best_time = init_best_time


  '''
    Get the next action produced by the agent.
  '''
  def get_next_act(self):
    states, acts, rewards, states_ = self.memory.get_last_n(max(self.model_mem_size, self.best_time*2))

    # Short score is the score accumulated on the same amount of time that best score was
    short_score = np.sum(rewards[-self.best_time:])
    # Long score is the score accumulated over twice the time of the best score
    long_score = np.sum(rewards[-self.best_time * 2:])

    # If long score better than 2 * best then it becomes the new best along with the new best time
    if long_score > 2 * self.best_score:
      self.best_score = long_score
      self.best_time = self.best_time * 2
    # If short score-1 is better it becomes the new best (-1 to avoid a situation where the best score is such that it the agent can't score more than double over double the time)
    elif short_score-1 > self.best_score:
      self.best_score = short_score-1

    # If best score <= 0 or not enough states to use model just choose randomly. Since target will be 
    # 2*best score, 2*0 isn't going to give any incentive to explore and if there are too few states then
    # random exploration is the best policy anyways.
    if self.best_score <= 0 or len(states) < max(self.model_mem_size, self.best_time*2):
      act = np.random.choice(np.arange(self.num_act))
    else:      
      model_x = np.reshape(states_[-self.model_mem_size:], (1,-1))
      # Run model on last few states with a target of getting twice the best score in twice the timesteps
      model_x = np.concatenate([model_x, 
                              np.array([[self.best_score * 2]]), 
                              np.array([[self.best_time * 2]])], axis=1)
    
      model_x = Tensor(model_x.astype(np.float32), requires_grad=False)

      act_probs = self.model.forward(model_x)
      act_probs = act_probs.data[0]
      # Sample from actions using the network output as probabilities
      act = np.random.choice(np.arange(act_probs.shape[0]), p=act_probs)
    return act


  def add_episode(self, s, a, r, s_):
    # Add only non-negative rewards since we are telling the network to produce 
    # actions that result in at least 2 * the best score so far (if it's 
    # negative then the agent will try to do worse)
    self.memory.add(s,a,max(r,0),s_)

  '''
    Train the network on one batch of data from the memory.
  '''
  def train_step(self, min_length=8, max_length=128, batch_size=16):

    if self.memory.get_length() > min_length + self.model_mem_size: 
      batch_xs = []
      batch_ys = []

      # length of time horizon to for getting reward
      length = np.random.randint(min_length, min(max_length, self.memory.get_length() - self.model_mem_size))
    
      for i in range(batch_size):
        states, actions, rewards, states_ = self.memory.sample_n(length + self.model_mem_size - 1)
      
        reward_sum = np.sum(rewards[self.model_mem_size-1:])

        # make the target reward in training <= the true sum of rewards so network learns to produce
        # actions that can be greater than the target reward.
        expected_reward = np.random.random() * reward_sum

        train_x = np.reshape(states[:self.model_mem_size], (1,-1))
        train_x = np.concatenate([train_x, 
                                  np.array([[expected_reward]]), 
                                  np.array([[length]])], axis=1)
      
        train_y = np.zeros((1,self.num_act))
        train_y[:, actions[self.model_mem_size-1]] = 1

        batch_xs.append(train_x)
        batch_ys.append(train_y)

      batch_xs = np.concatenate(batch_xs, axis=0)
      batch_ys = np.concatenate(batch_ys, axis=0)
      batch_xs = Tensor(batch_xs.astype(np.float32), requires_grad=True)
      batch_ys = Tensor(batch_ys.astype(np.float32), requires_grad=True)
      
      model_y = self.model.forward(batch_xs)

      # mean square error loss
      diff = model_y.sub(batch_ys)
      loss = diff.mul(diff).mean()
    
      # backprop and weight update
      self.optim.zero_grad()
      loss.backward()
      self.optim.step()

    

        
    


if __name__=="__main__":

  env = gym.make('Acrobot-v1')
  env.reset()

  num_obsv = 6

  num_act = 3

  agent = UDRLAgent(num_obsv, num_act) 
  last_feedback = np.zeros((num_obsv,), np.float32)
  for i in range(2000000):
    env.render()
    
    act = agent.get_next_act()
    feedback, reward, done, _ = env.step(act)
    #reward +1 because this environment is set up in the worst possible way (-1 all the time except when 
    #it does good in which case it gets 0)
    agent.add_episode(last_feedback, act, reward+1, feedback)
    agent.train_step() 
    
    last_feedback = feedback
    if i % 400 == 0:
      print(f'Best Score: {agent.best_score}, Best Time: {agent.best_time}')

