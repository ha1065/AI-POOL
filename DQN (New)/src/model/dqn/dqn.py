import math
import pickle
import numpy as np
import sys



import torch
import matplotlib.pyplot as plt

import numpy as np
import random
from collections import namedtuple, deque

# from . import nn_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#<---------Neural Network--------->#

class QNetwork(nn.Module): #Inspired from (https://github.com/udacity/deep-reinforcement-learning)
    """Neural network Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias. data.fill_(0.01)

        self.apply(init_weights)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    def reduce(self, state):#(To get angles of shot between 0 and 1)
        x = self.forward(state)
        return F.softmax(x)

#<---------Q-Learning set up--------------->

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 4         # minibatch size
GAMMA = 0.99            # discount factor
target_update_PARAM = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_STEPS = 4        # how often to update the network

count = 0

device = torch.device("cpu")#(where to store the models , alternatively could use 'cuda:0' if gpu available)

class Agent():
    """Interacts with and learns from the environment."""
    Loss = []#(Store the losses after each update when learning)
    def __init__(self, state_size, action_size, seed):
        """Initialize a RL Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        

        # Q-Network
        self.qnetwork_policy = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_policy.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_STEPS steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # record experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learning every UPDATE_STEPS time steps.
        self.t_step = (self.t_step + 1) % UPDATE_STEPS
        if self.t_step == 0:
            # If enough samples are available in memory, get a shuffled subset and learn
            if len(self.memory) > BATCH_SIZE:
            #     #print('Learning...')
                experiences = self.memory.sample()
                states, actions, rewards, next_states, dones = experiences

                # Get max predicted Q values (for next states) from target network
                Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
                # Compute Q targets for current states
                Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

                # Get expected Q values from policy(local) network
                Q_expected = self.qnetwork_policy(states).gather(1, actions)

                # Compute loss
                loss = F.mse_loss(Q_expected, Q_targets)
        
                # Minimize the loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.Loss.append(loss)
        
        
                # ------------------- update target network ------------------- #
                self.target_update(self.qnetwork_policy, self.qnetwork_target, target_update_PARAM)
                # return loss

                

                # self.learn(experiences , GAMMA)
                

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_policy.eval()
        with torch.no_grad():
            action_values = self.qnetwork_policy.reduce(state)
            # print(action_values)
            # print(list(action_values.cpu().data.numpy())[0])
        self.qnetwork_policy.train()

        action_values = list(action_values.cpu().data.numpy())[0]
        # action_values = [float(i)/sum(action_values) for i in action_values]
        # print(action_values)


        # Epsilon-greedy action selection
        if random.random() > eps:
            #print('Most optimal action.')
            # print (np.random.choice(np.arange(self.action_size), p=action_values))
            return np.random.choice(np.arange(self.action_size), p=action_values)
        else:
            return random.choice(np.arange(self.action_size))


    def target_update(self, local_model, target_model, target_update_param):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            target_update_param (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(target_update_param*local_param.data + (1.0-target_update_param)*target_param.data)
    #plot losses over each epoch
    def plot_losses(self, epochs, losses):
        plt.plot(losses, label='train')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()    
        plt.show()

    #plot losses over each batch update, i.e after every 4 episodes in an epoch in this case
    def plot_avg_losses(self, epochs, losses, batch_size):
        avg_losses = []
        for i in range(0, len(losses) // batch_size):
            avg_losses.append(sum(losses[i * batch_size: (i+1) * batch_size])/batch_size)
    
        plt.plot(avg_losses, label='train')
        plt.xlabel("Batch")
        plt.ylabel("Average Loss")
        plt.legend()  
        plt.rcParams.update({'font.size': 5})

        plt.show()



class ReplayBuffer:
    #Fixed-size memory buffer to record experience tuples

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

#<-----------Training and testing set up--------------->#

#saves model after training
def save_model(filepath, model):
    torch.save(model.qnetwork_policy.state_dict(), filepath)


#loads model for testing
def load_model(modelpath, model_params):
    state_size = model_params['s_dim']
    print('State size is ' + str(state_size))
    action_buckets = model_params['buckets']
    print('Action buckets are ' + str(action_buckets))
    action_size = action_buckets[0] * action_buckets[1]
    print('Action size is ' + str(action_size))

    agent = Agent(state_size, action_size, seed = 360)
    agent.qnetwork_policy.load_state_dict(torch.load(modelpath))
    

    return agent


def action_to_tuple(action, action_buckets):#reformatting outputted action from network in format acceptable for environment.
    
    return(float(int(action) % action_buckets[0]),\
        int(action/action_buckets[0]))

def choose_action(state, model, action_space, epsilon = 0.):#choosing action through epsilon-greedy startegy
    action = action_to_tuple(model.act(state, epsilon), action_space.buckets)
    print('action was ' + str(action))
    return action

def train(env, model_path, episodes=800, episode_length=25):
    print('DQN training')
    print(torch.cuda.is_available())

    # Initialize DQN Agent
    state_size = env.state_space.n
    action_buckets = [360, 1]
    env.set_buckets(action=action_buckets)
    action_size = action_buckets[0] * action_buckets[1]
    # print(action_size) ---debugging

    agent = Agent(state_size, action_size, seed = 360)
    
    #bellman equation variables
    # Learning related constants; factors should be determined by trial-and-error (inspired from https://github.com/udacity/deep-reinforcement-learning)
    get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/25))) # epsilon-greedy, factor to explore randomly; discounted over time
    get_lr = lambda i: max(0.01, min(0.5, 1.0 - math.log10((i+1)/25))) # learning rate; discounted over time
    gamma = 0.8 # reward discount factor

    # Q-learning
    for i_episode in range(episodes):
        epsilon = get_epsilon(i_episode)
        lr = get_lr(i_episode)

        state = env.reset() # reset environment to initial state for each episode
        rewards = 0 # accumulate rewards for each episode
        done = False
        for t in range(episode_length):
            # Agent takes action using epsilon-greedy algorithm, get reward
            
            action = agent.act(state, epsilon)
            
            next_state, reward, done = env.step(action_to_tuple(action, action_buckets))
            rewards += reward
            
            # Agent learns over New Step
            agent.step(state, action, reward, next_state, done)

            # Transition to next state
            state = next_state

            



            if done:
                print('Episode {} finished after {} timesteps, total rewards {}'.format(i_episode, t+1, rewards))
                with open("dqn-log.txt", "a") as myfile:
                    myfile.write('Episode {} finished after {} timesteps, total rewards {}\n'.format(i_episode, t+1, rewards))
                break
        if not done:
            print('Episode {} finished after {} timesteps, total rewards {}'.format(i_episode, episode_length, rewards))
            with open("dqn-log.txt", "a") as myfile:
                myfile.write('Episode {} finished after {} timesteps, total rewards {}\n'.format(i_episode, episode_length, rewards))

        save_model(model_path, agent)
    
    
       
    losses = agent.Loss
    print(losses)
    agent.plot_avg_losses(episodes, losses, 4)
    

    
