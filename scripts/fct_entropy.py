#Active learning : ajout des données en fonction de l'entropie
import pandas as pd
import numpy as np
import torch

def entropy(model, loaders, pool_act, pool_rand):
    entropy_list = torch.empty(0)

    for x, y in loaders:
        output = torch.nn.functional.softmax(model(x), dim=1)

        entrop = (-(output+10**-7)*torch.log(output+10**-7)).sum(1)
        entropy_list = torch.cat((entropy_list, entrop), 0)

    pool_act['entropy'] = entropy_list.tolist()
    top5 = pool_act.sort_values(by = 'entropy', ascending=False).head()
    top5['path'].to_csv('../shared/topEntropy.txt', sep=' ', index=False)
    
    rand =  pool_rand.sample(n=5)
    rand['path'].to_csv('../shared/random.txt', sep=' ', index=False)