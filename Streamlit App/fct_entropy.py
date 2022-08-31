#Active learning : ajout des données en fonction de l'entropie
import pandas as pd
import numpy as np
import torch

def entropy(model, loaders, batch_size):
    pool = pd.read_csv("./pool.csv")
    entropy_list = torch.empty(0)

    for x, y in loaders:
        output = torch.nn.functional.softmax(model(x), dim=1)

        entrop = (-(output+10**-7)*torch.log(output+10**-7)).sum(1)
        entropy_list = torch.cat((entropy_list, entrop), 0)

    pool['entropy'] = entropy_list.tolist()
    top = pool.sort_values(by = 'entropy', ascending=False).head(batch_size)

    return top 