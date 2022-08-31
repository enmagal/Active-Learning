import numpy as np
import pandas as pd

def labelisation(df):
    path = open("../shared/topEntropy.txt", "r")
    data = {'label' : ['label']*5}
    df_label = pd.DataFrame()
    paths = path.readlines()[1:]
    for i in range(len(paths)):
        paths[i] = paths[i][:-1]
        label = df['label'][np.where(df['path'] == paths[i])[0][0]]
        df_label.loc[i, 'label'] = label
    df_label['label'].to_csv('../shared/labels.txt', sep=' ', index=False)

    path = open("../shared/random.txt", "r")
    data = {'label' : ['label']*5}
    df_label2 = pd.DataFrame()
    paths = path.readlines()[1:]
    for i in range(len(paths)):
        paths[i] = paths[i][:-1]
        label = df['label'][np.where(df['path'] == paths[i])[0][0]]
        df_label2.loc[i, 'label'] = label
    df_label2['label'].to_csv('../shared/labels_rand.txt', sep=' ', index=False)