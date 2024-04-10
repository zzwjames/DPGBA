import torch
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import faiss
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
torch.set_printoptions(threshold=100000000)
x = torch.load('tensor.pt', map_location='cuda')
x=x.repeat(2,1)
print('x',x)
print('x',x.size())
kmeans = faiss.Kmeans(x.shape[1], 2, niter=20) 
kmeans.train(x.cpu().detach().numpy())
centroids = torch.FloatTensor(kmeans.centroids).to(x.device)

# Assign clusters
D, I = kmeans.index.search(x.cpu().detach().numpy(), 1)

# Convert centroids to PyTorch tensor
centroids = torch.FloatTensor(kmeans.centroids).to(x.device)

# Dimensionality reduction for visualization (if needed)
pca = PCA(n_components=2)
x_reduced = pca.fit_transform(x.cpu().detach().numpy())

# Visualization
plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=I.squeeze(), cmap='viridis')
plt.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='red', marker='x')  # centroids
plt.title("Cluster Visualization")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig('./clu.png')
plt.show()

indomain = []
# Output cluster labels
for idx, label in enumerate(I):
    if label[0] == 0:
        indomain.append(idx)
    print(f"Index: {idx}, Cluster: {label[0]}")
print(indomain)