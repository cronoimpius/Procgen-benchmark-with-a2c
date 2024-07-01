"""
Script for testing pretrained models
"""
from agent import *
import argparse 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-f','--file', type=str, default="", action='store', help='Path of saved model')
    parser.add_argument('-a','--agent', type=str, default='random', action='store', help='Select the agent type: random, a2c')
    parser.add_argument('-g','--game', type=str, default='heist', action='store',help='Select game to play')
    parser.add_argument('-sl', '--start', type=int, default=0, action='store', help='Insert the number of the starting level')
    parser.add_argument('-nl', '--num_levels', type=int, default=200, action='store', help='Insert the number of levels used in training')
    parser.add_argument('-s', '--total_steps', type=int, default=100000, action='store', help='Select the number of steps for training')
    args = parser.parse_args()

    ag =args.agent
    game = args.game
    sl = args.start
    nl = args.num_levels
    total_steps = args.total_steps
    file = args.file
    
    replay = Agent(ag)
    replay.replay(file, game, nl, sl)
    print("done, ending...")
