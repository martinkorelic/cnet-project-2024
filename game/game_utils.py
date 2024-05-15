import random
import os
import csv
from button import Button

class MiniCluster():
    def __init__(self, root, words):
        self.root = root
        self.words = words

    def __repr__(self):
        return f'Root Word: {self.root}\n4s :  {self.words}\n'


class Connections():
    def __init__(self, clusters):
        self.clusters = clusters
        self.table = self.init_table()
        self.queue = []
        self.GAME_BUTTONS = []
        
    def init_table(self):
        words, table = [], []
        for cluster in self.clusters:
            words.extend(cluster.words)
        random.shuffle(words)
        L = len(self.clusters)
        for i in range(L):
            table.append(words[i*4:(i+1)*4])
        return table  
    
    def empty(self):
        self.GAME_BUTTONS = []



def sampler(path='cluster_data/random_walk_sim'):
    dirs = os.listdir(path)
    groups = random.sample(dirs, k=4)
    clusters = []
    for g in groups:
        root, _ = g.split('.')
        group = []
        with open(f'{path}/{g}', "r", newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                group.append(row[0])
            samples = random.sample(group, k=4)
            clusters.append(MiniCluster(root, samples))

    return clusters
