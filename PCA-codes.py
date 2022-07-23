import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
#df=pd.read_casv('D:/LAX-air-pollution.csv')
df=pd.read_csv('D:/tehranairaghdasie.csv')
df.dropna(inplace=True)


X=df.iloc[:,1:]

#X=df
scaledX=scale(X)
p=PCA()
p.fit(scaledX)
W=p.components_.T
#Get the PC scores based on the centered X 
y=p.fit_transform(scaledX)

#Compute the PC scores based on the original values of X (just for easier interpretation)

plt.figure(1)


#Get the scatter plot of the first two PC scores
#plt.scatter(yhat.iloc[:,0],yhat.iloc[:,1],c="b",marker='o',alpha=0.5)
plt.scatter(y[:,0],y[:,1],c="b",marker='o',alpha=0.5)

plt.xlabel('PC Scores 1')
plt.ylabel('PC Scores 2')

#Put the name of the contries on the plotted datapoints (this is called annotation)
names=df.iloc[:,0].agg(lambda x:x[-4:])
for i, txt in enumerate(names):
    plt.annotate(txt, (y[i,0], y[i,1]))



#Biplots
xs=y[:,0]#xs represents PC score 1
ys=y[:,1]#ys represents PC score 2
#plot the arrows associated with variables
for i in range(len(W[:,0])):
# arrows project features (ie columns from csv) as vectors onto PC axes
#here we multiply W by abs(max(xs)) and abs(max(ys)) to scale the biplots
    plt.arrow(np.mean(xs), np.mean(ys), W[i,0]*abs(max(xs)), W[i,1]*abs(max(ys)),
              color='r', width=0.0005, head_width=0.0025)
    plt.text(W[i,0]*abs(max(xs))+np.mean(xs), +np.mean(ys)+W[i,1]*abs(max(ys)),
             list(df.columns.values[1:])[i], color='r')


#Get the first three columns of the matrix of loadings 
pd.DataFrame(W[:,:2],index=df.columns[1:],columns=['PC1','PC2'])
#Compute the explained variability by the PC scores
pd.DataFrame(p.explained_variance_ratio_,index=np.arange(1,len(p.explained_variance_ratio_)+1),columns=['Explained Variability'])
#Get the scree plot
plt.figure(2)
plt.bar(np.arange(1,len(p.explained_variance_ratio_)+1),p.explained_variance_,color="blue",edgecolor="Red")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
#df=pd.read_casv('D:/LAX-air-pollution.csv')
df=pd.read_csv('D:/tehranairaghdasie.csv')
df.dropna(inplace=True)
X=df.iloc[:,1:]
scaledX=scale(X)
p=PCA()
p.fit(scaledX)
#compute the explained variability by the first two PC scores
orgexplained=p.explained_variance_ratio_.cumsum()[1]


explained=np.zeros(1000)
for i in np.arange(1000):
    #perform random sampling with replacement and generate new matrices
    s=np.random.choice(X.shape[0],X.shape[0],replace=True)
    Xnew=scaledX[s,:]
    #apply PCA on the new matrices and compute the explained variability
    p=PCA()
    p.fit(Xnew)
    explained[i]=p.explained_variance_ratio_.cumsum()[1]

#plot the histogram for the explained variability and draw the 2.5% and 97.5% quantiles
plt.hist(explained,bins=60,color=(1,0.94,0.86),edgecolor=(0.54,0.51,0.47))
plt.axvline(np.quantile(explained,0.975),color=(0.46,0.93,0))
plt.axvline(np.quantile(explained,0.025),color=(0.46,0.93,0))
plt.axvline(orgexplained,color='red')
plt.xlabel('Explained Variability')
plt.ylabel('Frequency')

