#!/usr/bin/python3
# -*- coding: utf-8 -*- 
# Script permettant de créer deux csv pool.csv et train.csv
# Pool est l'ensemble des données non labélisé on a donc le chemine et l'entropie
# Train est l'ensemble des données labélisé on a donc le chemin et le label
# Entrée attendue : chemin du dossier des images

import sys
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

try :
    path = sys.argv[1]
except :
    print("No argument where one is required")

files = [path + "/" + f for f in listdir(path) if isfile(join(path, f))]
entropy = np.zeros(len(files))

data = {'path': files,
        'entropy': entropy}

pool = pd.DataFrame(data)

pool.to_csv('../shared/pool.csv', index=False)

data = {'path': [],
        'label': []}

train = pd.DataFrame(data)

train.to_csv('../shared/train.csv', index=False)
