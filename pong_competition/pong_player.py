import random
import itertools
import math
import gym
import numpy as np
import torch
from collections import namedtuple

import torch.nn.functional as F

from pong_env import PongEnv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# TODO replace this class with your model
class MyModelClass(torch.nn.Module):
    
    def __init__(self):
        super(MyModelClass, self).__init__()
        self.linear1 = torch.nn.Linear(7, 5)
        self.linear2 = torch.nn.Linear(5, 3 )
        self.steps = 0
        
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x
        


# TODO fill out the methods of this class
class PongPlayer(object):

    def __init__(self, save_path, load=False):
        self.build_model()
        self.build_optimizer()
        self.steps = 0
        self.save_path = save_path
        if load:
            self.load()

    def build_model(self):
        # TODO: define your model here
        # I would suggest creating another class that subclasses
        # torch.nn.Module. Then you can just instantiate it here.
        # your not required to do this but if you don't you should probably
        # adjust the load and save functions to work with the way you did it.
        self.model = MyModelClass()

    def build_optimizer(self):
        # TODO: define your optimizer here
        # self.optimizer = None
        self.dqn = MyModelClass()
        self.optimizer = torch.optim.RMSprop(self.dqn.parameters(), lr=0.0001)
        
    policy_net = MyModelClass()

    def get_action(self, state):
        # TODO: this method should return the output of your model
        print(state)
        self.steps += 1
        choice = random.random()
        eps_treshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * self.steps / EPS_DECAY)
        if choice > eps_treshold:
            with torch.no_grad():
                tensor = MyModelClass().to(device)(torch.tensor(state, dtype=torch.float32))
                print(tensor)
                print(tensor.max(0)[1])
                out = tensor.max(0)[1].numpy()
                return out
        else:
            out =  torch.tensor([[random.randrange(2)]],device = device , dtype=torch.long).numpy()[0, 0]
            return out
        

    def reset(self):
        # TODO: this method will be called whenever a game finishes
        # so if you create a model that has state you should reset it here
        # NOTE: this is optional and only if you need it for your model
        pass

    def load(self):
        state = torch.load(self.save_path)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])

    def save(self):
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(state, self.save_path)

    
def play_game(player, render=True):
    # call this function to run your model on the environment
    # and see how it does
    env = PongEnv()
    state = env.reset()
    action = player.get_action(state)
    done = False
    total_reward = 0
    while not done:
        next_state, reward, done, _ = env.step(action)
        if render:
            env.render()
        action = player.get_action(next_state)
        total_reward += reward
    
    env.close()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_DECAY = 200
EPS_END = 0.05
TARGET_UPDATE = 10
policy_net = MyModelClass().to(device)
target_net = MyModelClass().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
memory = ReplayMemory(10000)

    
    
p1 = PongPlayer('/Users/mtorjyan/Projects/Berkeley/Fall18/hackNew/hack/pong_competition/out.txt')
play_game(p1)
