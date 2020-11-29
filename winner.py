import retro
import numpy as np
import cv2 
import neat
import pickle

#Create new gym for showcasing the fittest genome in action
env = retro.make('SuperMarioWorld-Snes', 'Start.state')

imgarray = []

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward-SMW.txt')

with open('winner.pkl', 'rb') as input_file:
    genome = pickle.load(input_file)

ob = env.reset()

inx, iny, inc = env.observation_space.shape

inx = int(inx/8)
iny = int(iny/8)

network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

current_max_fitness = 0
fitness_current = 0
counter = 0
xpos = 0
xpos_max = 0

done = False

while not done:
    
    env.render()

    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx,iny))

    for x in ob:
        for y in x:
            imgarray.append(y)
    
    nnOutput = network.activate(imgarray)
    ob, rew, done, info = env.step(nnOutput)

    fitness_current += rew
    imgarray.clear()
    
    if fitness_current > current_max_fitness:
        current_max_fitness = fitness_current
        counter = 0
    
    else:
        counter += 1
    
    if done or counter == 350:
        done = True
        