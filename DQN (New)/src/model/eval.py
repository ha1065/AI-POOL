import argparse
import sys
import pickle

from .env import PoolEnv
from .dqn import dqn
import matplotlib.pyplot as plt



EPISODES = 100
EPISODE_LENGTH = 25

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL model evaluation.')
    parser.add_argument('--model', type=str, default='model.pkl',
            help='Input model path. Default: model.pkl')
    parser.add_argument('--algo', type=str, default='random',
            help='dqn (Deep Q-Network)')
    parser.add_argument('--balls', type=int, default=2,
            help='Number of balls on table (including white ball), should be >= 2. Default: 2')
    parser.add_argument('--visualize', dest='visualize', action='store_true',
            help='To see the visualization of the pool game.')
    args = parser.parse_args()

    if args.balls < 2:
        print('Number of balls should be >= 2.')
        sys.exit(1)

    env = PoolEnv(args.balls, visualize=args.visualize)
    model = None
    
    
    if args.algo == 'dqn':
        choose_action = dqn.choose_action
        env.set_buckets(action=[360, 1])
        model_params = { 's_dim': env.state_space.n,
                         'a_dim': env.action_space.n,
                         'buckets': env.action_space.buckets}
        model = dqn.load_model(args.model, model_params)
    
    else:
        print('Algorithm not supported! Should be one of q-table or dqn.')
        sys.exit(1)
    rewards = []

    total_rewards = 0
    for i_episode in range(EPISODES):
        state = env.reset()
        rewards_int = 0
        done = False
        for t in range(EPISODE_LENGTH):
            running_rewards = 0
            action = choose_action(state, model, env.action_space)
            next_state, reward, done = env.step(action)
            rewards_int += reward
            state = next_state

            if done:
                print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards_int))
                #<----Different implementation for reward visualization , didn't prove to be too useful-----># 
                # if running_rewards == 0:
                #     running_rewards = rewards_int
                # else:
                #     running_rewards = running_rewards * 0.99 + rewards_int * 0.01

                
                total_rewards += rewards_int
                rewards.append(rewards_int/t+1)
                break
        if not done:
            print('Episode finished after {} timesteps, total rewards {}'.format(EPISODE_LENGTH, rewards_int))
            #<----Different implementation for reward visualization , didn't prove to be too useful-----># 
            # if running_rewards == 0:
            # running_rewards = rewards_int
            # else:
            #         running_rewards = running_rewards * 0.99 + rewards_int * 0.01

                
            total_rewards += rewards_int
            rewards.append(rewards_int/EPISODE_LENGTH)

   #print average rewards over total test epochs         
    print('Average rewards: {}'.format(total_rewards / EPISODES))
   
    #plot reward per epoch/episode
    plt.plot(rewards, label='test')
    plt.xlabel("Episodes")
    plt.ylabel("rewards")
    plt.title("Rewards Trend")
    plt.legend()    
    plt.show()