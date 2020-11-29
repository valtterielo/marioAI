import retro
import numpy as np
import cv2 
import neat
import pickle

#Creating enviroment for training
enviroment = retro.make('SuperMarioWorld-Snes', 'DonutPlains1.state')


imgarray = []

def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        ob = enviroment.reset()
        action = enviroment.action_space.sample()

        #Get the screen size
        x, y, c = enviroment.observation_space.shape
        x = int(x/8)
        y = int(y/8)

        network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        
        current_max_fitness = 0
        fitness_current = 0
        counter = 0
        
        done = False
        while not done:
            
            enviroment.render()

            #Turn the screen into 8x8 chunks 
            ob = cv2.resize(ob, (x, y))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (x,y))

            #Put the screen chunks into array that 
            imgarray = np.ndarray.flatten(ob)
            nnOutput = network.activate(imgarray)  

            ob, rew, done, info = enviroment.step(nnOutput)                       

            dead = info['dead']
            fitness_current += rew

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
                
            if done or counter == 250 or dead == 0:
                done = True
                print(genome_id, fitness_current)
                
            genome.fitness = fitness_current

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward-SMW.txt')

population = neat.Population(config)
population = neat.Checkpointer.restore_checkpoint('neat-checkpoint-55')

population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.add_reporter(neat.Checkpointer(10))

winner = population.run(eval_genomes)

with open('winner1.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
