import math
import numpy as np
import copy

import gamestate_mlp as gamestate
import config
import collisions
import random

import torch
import torch.nn as nn
import torch.nn.functional as F



class Interface():
    #Sets up the interface for a 2 ball environment
    def __init__(self):
        self.edge_offset = config.table_margin + config.hole_radius + config.ball_radius
        self.white_ball_pos = np.array([self.edge_offset + config.table_size[0]/4,config.resolution[1]/2])
        self.other_ball_pos = self.get_random_pos(self.white_ball_pos)
        self.game = gamestate.GameState()
        self.set_game = True
        

    #Resets the ball postions in a random location
    def random_reset_balls(self):
        self.white_ball_pos = self.get_random_pos_2(True)
        self.other_ball_pos = self.get_random_pos_2(False)


    #Returns a random ball position that is not in conflict with the inputted ball position
    def get_random_pos(self, ball_pos):
        x_good = False
        y_good = False
        while not x_good:
            x = random.randint(self.edge_offset, config.resolution[0] - self.edge_offset)
            if x - config.ball_radius > ball_pos[0] + config.ball_radius or x + config.ball_radius < ball_pos[0] - config.ball_radius:
                x_good = True
                
        while not y_good:
            y = random.randint(self.edge_offset, config.resolution[1] - self.edge_offset)
            if y - config.ball_radius > ball_pos[1] + config.ball_radius or y + config.ball_radius < ball_pos[1] - config.ball_radius:
                y_good = True
                
        return np.array([x,y])


    #Returns a random ball position that is not in conflict with the inputted ball position if the ball is the target ball, otherwise returns a random ball position
    def get_random_pos_2(self, cue_ball):
        if cue_ball == True:
            x = random.randint(self.edge_offset, config.resolution[0] - self.edge_offset)
            y = random.randint(self.edge_offset, config.resolution[1] - self.edge_offset)
            return np.array([float(x),float(y)])

        else:
            x_good = False
            y_good = False
            while not x_good:
                x = random.randint(self.edge_offset, config.resolution[0] - self.edge_offset)
                if x - config.ball_radius > self.white_ball_pos[0] + config.ball_radius or x + config.ball_radius < self.white_ball_pos[0] - config.ball_radius:
                    x_good = True
                    
            while not y_good:
                y = random.randint(self.edge_offset, config.resolution[1] - self.edge_offset)
                if y - config.ball_radius > self.white_ball_pos[1] + config.ball_radius or y + config.ball_radius < self.white_ball_pos[1] - config.ball_radius:
                    y_good = True
                    
            return np.array([x,y])


    #Returns the state (positions of the balls) of the environment 
    def get_state(self):
        state = torch.tensor([self.white_ball_pos[0], self.white_ball_pos[1], self.other_ball_pos[0], self.other_ball_pos[1]], requires_grad=True)
        return state


    #Returns the hole number that the hole location is at
    def get_hole_num(self, hole_loc):
        holenum = 0
        if hole_loc == [1.0, 1.0]:
            holenum = 1
        elif hole_loc == [2.0, 1.0]:
            holenum = 2
        elif hole_loc == [3.0, 1.0]:
            holenum = 3
        elif hole_loc == [1.0, 2.0]:
            holenum = 4
        elif hole_loc == [2.0, 2.0]:
            holenum = 5
        elif hole_loc == [3.0, 2.0]:
            holenum = 6
        else:
            holenum = 999
        return holenum


    #Applies a shot with the inputted angle and power and returns the current state, action, power, and next state, then resets the interface for the next shot
    def take_shot(self, angle, power):
        move_done = False

        #Gets the ball's positions
        white_ball_pos = self.white_ball_pos
        other_ball_pos = self.other_ball_pos

        #Sets the balls and takes the shot
        self.game.start_pool_mlp(white_ball_pos, other_ball_pos, angle, power)

        while not move_done: #Lets the balls do their thing according to the game rules
            collisions.resolve_all_collisions(self.game.balls, self.game.holes, self.game.table_sides)
        
            for ball in self.game.balls:
                ball.update()

            if self.game.all_not_moving():
                move_done = True
        
        #Shot is over at this point

        #Gathers end ball positions
        for ball in self.game.balls:
            ball_pos = ball.ball.pos

            if int(ball.number) == 0:
                white_ball_end_pos = ball_pos
            else:
                other_ball_end_pos = ball_pos

        #Checks for pocketed balls (0 means not pocketed)
        white_ball_pocket = 0
        other_ball_pocket = 0

        for hole in config.potted.values():
            if int(hole[0]) == 0:
                white_ball_end_pos = hole[1]
                white_ball_pocket = self.get_hole_num(hole[2])
            elif int(hole[0]) == 1:
                other_ball_end_pos = hole[1]
                other_ball_pocket = self.get_hole_num(hole[2])

        #Resets the potted config file for the next shot
        config.potted = {}

        #Assembles the output
        data = np.array([
            white_ball_pos[0], white_ball_pos[1], other_ball_pos[0], other_ball_pos[1], 
            angle, power,
            white_ball_end_pos[0], white_ball_end_pos[1], other_ball_end_pos[0], other_ball_end_pos[1],
            white_ball_pocket, other_ball_pocket
            ])
        

        output_data = copy.deepcopy(data) 

        self.random_reset_balls()

        return output_data 





