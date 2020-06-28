# %%
# Import base packages
import os
import numpy as np
import pandas as pd
from IPython.display import display

# Pandas Formatting and Styling:
pd.options.display.max_rows = 200
pd.options.display.max_columns = 500
pd.set_option('display.float_format',lambda x: '%.3f' % x)

# Data Visualization Packages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'image.cmap': 'cubehelix'})
sns.set_context('poster')

# Import scaling from sklearn
from sklearn.preprocessing import normalize, MinMaxScaler

# %%
# Read in files
data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data', 'interim'))
X_path = os.path.join(data_path, 'EV.csv')
y_path = os.path.join(data_path, 'EV_labels.csv')

# %%
# Read in files
ev = pd.read_csv(X_path)
ev_labels = pd.read_csv(y_path)

display(ev.shape, ev.head(25))
display(ev_labels.shape, ev_labels.head(10))

# %%
# Drop House ID
try:
    ev.drop('House ID', axis = 1, inplace = True)
    ev_labels.drop('House ID', axis = 1, inplace = True)
except:
    pass

# %%
# Find number of houses that have an EV
y_hot = np.array([1 if t > 0 else 0 for t in ev_labels.sum(axis = 1).values.tolist()]).reshape(-1,1)

print("# of Houses with an EV: ", y_hot.sum()) # 485 houses with an EV

# %%
# Plot a non EV home
fig = plt.figure(1)
plt.figure(figsize=(60,20))
plt.plot(ev.iloc[0,:].values)
plt.title('A non EV house', fontsize=48)
plt.xlabel('Timestamps', fontsize=36)
plt.ylabel('Smart Meter Reading', fontsize=36)

# %%
# Plot an EV home
fig = plt.figure(2)
plt.figure(figsize=(60,20))
plt.plot(ev.iloc[4,:].values)
plt.title('An EV house', fontsize=48)
plt.xlabel('Timestamps', fontsize=36)
plt.ylabel('Smart Meter Reading', fontsize=36)

# %%
# Take a min/max scaler of each house with and without and EV
with_ev = ev[y_hot == 1]
without_ev = ev[y_hot == 0]

# %%
scaler_ev = MinMaxScaler()
scaled_ev = scaler_ev.fit_transform(with_ev)

scaler_wo_ev = MinMaxScaler()
scaled_wo_ev = scaler_wo_ev.fit_transform(without_ev)

# %%
# Plot scaled average of houses without EVs
fig = plt.figure(4)
plt.figure(figsize=(60,20))
plt.plot(scaled_wo_ev.mean(axis = 0))
plt.title('Average of non-EV houses', fontsize=48)
plt.xlabel('Timestamps', fontsize=36)
plt.ylabel('Smart Meter Reading', fontsize=36)

# %%
# Plot scaled average of houses with EVs
fig = plt.figure(3)
plt.figure(figsize=(60,20))
plt.plot(scaled_ev.mean(axis = 0))
plt.title('Average of EV houses', fontsize=48)
plt.xlabel('Timestamps', fontsize=36)
plt.ylabel('Smart Meter Reading', fontsize=36)



# %%
# * Notes: Houses with EVS clearly have stronger and more visible peaks whereas houses without EVs don't show spikes, only the general daily profile of energy use. It looks like houses with EVs show multiple profiles combined, 1 for EV charging and another 1 just for daily energy use. 

