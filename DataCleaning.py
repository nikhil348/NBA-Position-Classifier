# -*- coding: utf-8 -*-

import pandas as pd
import random
random.seed(123)

df = pd.read_csv("regular_season_players.csv") # raw data

target = df['pos'] 

# Delete columns in the raw data that are redundant and/or do not
# contribute to defining player position
del_columns = ['Unnamed: 0','rk','age', 'g', 'gs', 'ws', 'vorp', 'ws_48', 'fg', 'fga', 'efgpercent', 'per', 'ps_g', 'mp_total', 'per']
df.drop(del_columns, inplace=True, axis=1)


# Use only players who played more than 15 minutes per game that season
# Brings the number of players in the data from 19647 to 12354
df = df[df.mp_per_game >= 15]

# Give each row and index for easier access
df = df.reset_index()

# Prints the number of NaNs for each column
# Only columns with NaNs were x3ppercent with 1109 and ftpercent with 39
df.isna().sum()

# Gets the mean free throw percentage of all players in the dataset by year
year_ft = pd.Series(df.groupby('i')['ftpercent'].mean()).to_dict()

# Looks for all NaNs in ftpercent and replaces them with 
# the league avg free throw percentage for that year
x = pd.Series(df.isna()['ftpercent']).to_dict()
for key in x:
    if x[key] == True:
        year = df.iloc[key, 36]
        replace = year_ft[year]
        df.iloc[key, 14] = replace
        
#df['ftpercent'].isna().sum()  
#df['x3ppercent'].isna().sum()  

# Looks for all NaNs in x3ppercent and replaces them with 
# a zero percent three point percentage
y = pd.Series(df.isna()['x3ppercent']).to_dict()
for key in y:
    if y[key] == True:
        df.iloc[key, 8] = 0
        
#df['x3ppercent'].isna().sum()  
#df.isna().sum() (shows no more NaNs)
        

import numpy as np

size = len(df)

# Array to hold the target values of the various models to be used:
# 1 means point guard, 2 means shooting guard, 3 means small forward,
# 4 means power forward, 5 means center
# Those classified as multiple positon players were grouped into their
# first position listed
targets = np.array([]) 

for i in range(size):
    if df.pos[i] == "PG":
        targets = np.append(targets, 1)
    elif df.pos[i] == "SG":
        targets = np.append(targets, 2)
    elif df.pos[i] == "SF":
        targets = np.append(targets, 3)
    elif df.pos[i] == "PF":
        targets = np.append(targets, 4)
    elif df.pos[i] == "C":
        targets = np.append(targets, 5)
    else:
        if df.pos[i] == "PG-SG":
            df.pos[i] = "PG"
            targets = np.append(targets, 1)
        elif df.pos[i] == "SG-PG":
            df.pos[i] = "SG"
            targets = np.append(targets, 2)
        elif df.pos[i] == "SG-SF":
            df.pos[i] = "SG"
            targets = np.append(targets, 2)
        elif df.pos[i] == "SF-SG":
            df.pos[i] = "SF"
            targets = np.append(targets, 3)
        elif df.pos[i] == "SF-PF":
            df.pos[i] = "SF"
            targets = np.append(targets, 3)
        elif df.pos[i] == "PF-SF":
            df.pos[i] = "PF"
            targets = np.append(targets, 4)
        elif df.pos[i] == "C-PF":
            df.pos[i] = "C"
            targets = np.append(targets, 5)
        elif df.pos[i] == "PF-C":
            df.pos[i] = "PF"
            targets = np.append(targets, 4)
        elif df.pos[i] == "PG-SF":
            df.pos[i] = "PG"
            targets = np.append(targets, 1)
        elif df.pos[i] == "SG-PF":
            df.pos[i] = "SG"
            targets = np.append(targets, 2)
        else:
            print "error ", i, df.pos[i]
    
#df.pos.value_counts()
#len(df)

df.to_csv('cleandata.csv', index=False)