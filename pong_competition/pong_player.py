import random
import itertools

import gym
import numpy as np
import torch

from pong_env import PongEnv

# TODO replace this class with your model
class MyModelClass(torch.nn.Module):
    
    def __init__(self):
        super(MyModelClass, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.head = torch.nn.Linear(448, 2)
        
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
        


# TODO fill out the methods of this class
class PongPlayer(object):

    def __init__(self, save_path, load=False):
        self.build_model()
        self.build_optimizer()
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
        
        print('hello')
        
        choice = np.random.choice([0, 1], p=(1, (1 - 1)))
        
        if choice == 0:
            return np.random.choice(range(1))
        else:
            state = np.expand_dims(state, 0)
            actions = self.predict_q_values(state)
            return np.argmax(actions.data.cpu().numpy())

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
    
    
p1 = PongPlayer('/Users/mtorjyan/Projects/Berkeley/Fall18/hackNew/hack/pong_competition/out.txt')
play_game(p1)