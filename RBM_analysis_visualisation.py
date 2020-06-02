import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# get data
df = pickle.load(open("df_important","rb"))
df.rename(columns={0:"momentum",1:"regularisation",2:"learning rate",3:"batches",4:"hidden variables",5:"validation RMSE"},inplace=True)

# find the top 3 results
df.sort_values(by=['validation RMSE']).head(3)

# Create new dataframe for mean analysis
de = pd.DataFrame()

de['momentum'] = df.groupby(by=['momentum']).mean()['validation RMSE']
de.reset_index(drop=True,inplace=True)
de['regularisation'] = df.groupby(by=['regularisation']).mean()['validation RMSE'].reset_index(drop=True)
de['learning rate'] = df.groupby(by=['learning rate']).mean()['validation RMSE'].reset_index(drop=True)
de['batches'] = df.groupby(by=['batches']).mean()['validation RMSE'].reset_index(drop=True)
de['hidden variables'] = df.groupby(by=['hidden variables']).mean()['validation RMSE'].reset_index(drop=True)
de.fillna(0,inplace=True)
print(de)


# set width of bar
barWidth = 0.25
 
# set height of bar
bar1 = de.iloc[0]
bar2 = de.iloc[1]
bar3 = de.iloc[2]
 
# Set position of bar on X axis
r1 = np.arange(5)
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.figure(figsize=(8,6))
plt.bar(r1, bar1, color='#7f6d5f', width=barWidth, edgecolor='white',label='option 1')
plt.bar(r2, bar2, color='#557f2d', width=barWidth, edgecolor='white',label='option 2')
plt.bar(r3, bar3, color='#2d7f5e', width=barWidth, edgecolor='white',label='option 3')
 
# Add xticks on the middle of the group bars
plt.xticks([r + barWidth for r in range(len(bar1))], ["momentum","regularisation","learning rate","batches","hidden variables"])
 

plt.ylabel("Validation RMSE")
plt.ylim(1.12,1.2)
# Create legend & Show graphic
plt.legend()
plt.show()