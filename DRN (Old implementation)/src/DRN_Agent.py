import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from DRN_interface import Interface
import pandas as pd



class DQN(nn.Module):

    def __init__(self, inputs, hidden1, hidden2 , outputs):
        super(DQN, self).__init__()

        self.n1 = nn.Linear(inputs, hidden1)
        self.n2 = nn.Linear(hidden1, hidden2)
        self.n3 = nn.Linear(hidden2,outputs)

        self.lrelu = nn.LeakyReLU(0.2)


    def forward(self, x):
        
        x = self.lrelu(self.n1(x))
        x = self.lrelu(self.n2(x))
        x = self.lrelu(self.n3(x))


        return x



def save_data(data, column_name, path):
    df = pd.DataFrame(data, columns = [column_name])
    df.to_csv(path, index=False)


def plot_losses(epochs, losses):
    plt.plot(losses, label='train')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")  
    plt.show()


def plot_avg_rewards_1(epochs, losses, batch_size):
    avg_losses = []
    for i in range(0, len(losses) // batch_size):
        avg_losses.append(sum(losses[i * batch_size: (i+1) * batch_size])/batch_size)
    
    plt.plot(avg_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Average Reward")
    plt.rcParams.update({'font.size': 5})

    plt.show()


def plot_avg_rewards_2(epochs, avg_rewards):
    plt.plot(avg_rewards, label='avg_rewards')
    plt.xlabel("Epoch")
    plt.ylabel("Avg. Reward")  
    plt.show()


#calculates the reward from the raw_output
def calculate_reward(output,): 
    #the output is in the form of [0xi, 0yi, 1xi, 1yi, a, p, 0x, 0y, 1x, 1y, 0h, 1h] for a 2 ball environment

    if output[2] == output[-4] and output[3] == output[-3]: #target ball wasn't touched
        reward = -3
        if output[-2] != 0: #sunk cue ball AND missed target ball // WORST CASE POSSIBLE
            reward = -1
    elif output[-2] != 0 and output[-1] != 0: #sunk target and cue ball
        reward = -1.5
    elif output[-1] != 0: #sunk the target ball, cue ball stull up
        reward = 0
    elif output[-2] != 0: #hit target ball and sunk cue ball
        reward = -2
    else: #hit target ball and both are still up
        reward = -1
    
    return reward
    

def fit_model(model, interface, loss_fn, optimizer, batch_size, save_interval, epochs = 1000):
    data = []
    losses = []
    all_rewards = []
    rewards = torch.tensor([], dtype= float, requires_grad=True)
    max_rewards = torch.zeros([batch_size], dtype=float, requires_grad= True)
    power = 100
    save_num = 1

    for epoch in range(epochs):
        print("\n\nEpoch: " + str(epoch))

        #Get the current state
        state = interface.get_state()
        print("State: " + str(state))

        #Get action from DRN
        action = model(state)
        print("Action: " + str(action))

        #Apply the shot and get the next state
        raw_output = interface.take_shot(action.data, power)
        
        #Calculate the reward
        reward = calculate_reward(raw_output)
        print("Reward: " + str(reward))

        #Save the rewards to batch reward tensor and output list
        all_rewards.append(reward)
        rewards = torch.cat((rewards,torch.tensor([reward], dtype=float)), 0)

        #Compute the loss at the end of the batch
        if epoch != 0 and (epoch+1) % batch_size == 0:
            
            #Calculate MSE loss of batch and push it to output list
            loss = loss_fn(rewards, max_rewards)
            losses.append(loss.data)

            #!!! The Magic Zone !!! :)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Reset batch rewards tensor for next batch
            rewards = torch.tensor([], dtype= float, requires_grad=True)

        #Save the data and model at the save interval
        if epoch != 0 and (epoch+1) % save_interval == 0:
            torch.save(model.state_dict(), "saves/state_dict_" + str(save_num*save_interval))
            torch.save(model, "saves/model_" + str(save_num*save_interval))
            
            print("Model Saved!!!")

            save_data(all_rewards, "Rewards", "big_run_data/Rewards_" + str(save_num*save_interval))
            save_data(losses, "Losses", "big_run_data/Losses_" + str(save_num*save_interval))

            print("Data Saved!!!")

            save_num += 1
    
    return losses, all_rewards










def test_model(model, interface, batch_size, epochs):
    
    all_rewards = []
    rewards = []
    avg_rewards = []
    power = 100

    for epoch in range(epochs):
        print("\n\nEpoch: " + str(epoch))

        #Get the current state
        state = interface.get_state()
        print("State: " + str(state))

        #Get the action from the DRN
        action = model(state)
        print("Action: " + str(action))

        #Apply the shot and get the next state
        raw_output = interface.take_shot(action.data, power)

        #Calculate the reward
        reward = calculate_reward(raw_output)

        #Save the rewards to batch reward tensor and output list
        all_rewards.append(reward)
        rewards.append(reward)
        print("Reward: " + str(rewards))

        #Compute the average reward at the end of the batch
        if epoch != 0 and (epoch+1) % batch_size == 0:
            total = 0
            for i in rewards:
                total += i
            avg = total/batch_size
            print(avg)
            avg_rewards.append(avg)

            rewards = []

    return avg_rewards, all_rewards


############################################################################


def train_drn():
    interface = Interface()
    model = DQN(4, 72, 32, 1)
    epochs = 500000
    batch_size = 100
    save_interval = 50000

    learning_rate = 0.01
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    losses, rewards = fit_model(model, interface, loss_fn, optimizer, batch_size, save_interval, epochs)
    plot_losses(epochs/batch_size, losses)
    plot_avg_rewards_1(epochs, rewards, batch_size)


def test_drn():
    interface = Interface()
    model = torch.load("saves/model_500000")
    model.eval()
    epochs = 2000
    batch_size = 50

    avg_rewards, all_rewards = test_model(model, interface, batch_size, epochs)
    plot_avg_rewards_2(epochs/batch_size, avg_rewards)

    totals = 0
    for i in all_rewards:
        totals += i

    avg = totals/epochs

    print(avg)
    print(avg_rewards)
    print(all_rewards)








train_drn()

#test_drn()




















































































































