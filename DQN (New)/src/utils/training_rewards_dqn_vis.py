import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean 

#<--------Visualizing average training rewards from stored rewards log "dqn-log.txt"
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python {} INPUT_FILE OUTPUT_FILE'.format(sys.argv[0]))
        sys.exit(1)

    input_file, output_file = sys.argv[1], sys.argv[2]

    y = []
    with open(input_file, 'r') as fin:
        running_rewards = 0
        for line in fin:
            split_line = line.strip().split(' ')
            timesteps, rewards = int(split_line[4]), int(split_line[8])
            if running_rewards == 0:
                running_rewards = rewards
            else:
                running_rewards = running_rewards * 0.99 + rewards * 0.01
            y += [running_rewards]
    x = list(range(len(y)))

    #uncomment this section if want the average reward over the last batch to be printed
    # mean_y = mean(y)
    # list_mean = []
    # print(mean_y)
    # for i in range(len(y)):
    #     if i > 795:
    #         list_mean.append(y[i])

        

    # print(list_mean)
    # print(mean(list_mean))

    
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='episode', ylabel='avg rewards', title='Avg. Rewards Trend (exponential moving average)')

    fig.savefig(output_file)# saving output png file#
