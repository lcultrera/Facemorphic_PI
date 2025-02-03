import matplotlib
import matplotlib.markers
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
mStyles = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_",0,1,2,3,4,5,6,7,8,9,10,11
]
X_raw = np.load("features.npy",allow_pickle=True)
X_hall = np.array(X_raw[:,0].tolist())
X_priv =  np.array(X_raw[:,1].tolist())
X_original =  np.array(X_raw[:,2].tolist())

X_labels= np.array(X_raw[:,3].tolist())
colors_0 = [ [0,121/255,180/255] for i in range(X_hall.shape[0])]
colors_1 = [[106/255,163/255,78/255] for i in range(X_priv.shape[0])]
colors_2 = [[209/255,73/255,91/255] for i in range(X_original.shape[0])]
colors = np.concatenate([colors_0,colors_1,colors_2])
X = np.concatenate([X_hall,X_priv,X_original])
mks = [X_labels[i]%len(mStyles) for i in range(X_labels.shape[0])]
plt.axis('off')
plt.tight_layout()
X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=100).fit_transform(X)

plt.scatter(X_embedded[:,0],X_embedded[:,1],c=colors,s=2)
plt.savefig("tsne_RGB_AS_PI.pdf")