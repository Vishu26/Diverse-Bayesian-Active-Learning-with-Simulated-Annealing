import numpy as np
from sklearn.cluster import KMeans
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import pairwise_distances
from tensorflow.keras.models import Model
import pdb
from collections import Counter

class ALSA:

    def __init__(self, X, alpha, beta, e, m, k, p, epochs) -> None:
        self.alpha = alpha
        self.beta = beta
        self.m = m
        self.k = k
        self.p = p
        self.epochs = epochs
        self.T = np.exp(-np.array(range(epochs))/(epochs)*beta)
        self.cluster_labels, self.cluster_centers = self.make_clusters(X, m)
        self.E_m = [e]*k
    
    def make_clusters(self, X, m):
        print(F"KMeans Clustering with {m} clusters")
        cluster = KMeans(n_clusters=m)
        cluster.fit(X.reshape(X.shape[0], X.shape[1]*X.shape[2]))
        X_cluster = cluster.labels_
        centers = cluster.cluster_centers_ 

        return X_cluster, centers

    def query(self, active_learner, X, labeled_idx, epoch):
        unlabeled_idx = np.arange(X.shape[0])[np.logical_not(np.in1d(np.arange(X.shape[0]), labeled_idx))]
        indexes = []
        b=0

        while b<self.k:
            T_s = self.T[epoch]
            for c in range(self.m):
                index = np.random.choice(np.where(self.cluster_labels[unlabeled_idx]==c)[0], size=self.p)

                for idx in index:

                    dist = np.linalg.norm(X[unlabeled_idx[idx]].reshape(X.shape[1]*X.shape[2],) - self.cluster_centers[c])
                    div = 1 / (1 + dist)
                    mi = 1 / (1 + np.sqrt(X.shape[1]*X.shape[2]))
                    div = (div - mi)/(1 - mi)
            
                    prob = np.mean([active_learner(X[unlabeled_idx[idx]].reshape(1, X.shape[1], X.shape[2], 1), training=True) for _ in range(10)], axis=0).flatten()
                    marg = np.argpartition(prob, -2)[-2:]
                    U = 1 - abs(prob[marg[0]]-prob[marg[1]])

                    E = T_s * div + (1-T_s) * U

                    if E  > self.E_m[c]:
                        self.E_m[c] = E
                        indexes.append(unlabeled_idx[idx])
                        b+=1

                    else:
                        if np.random.uniform() < np.exp(-self.alpha*(self.E_m[c] - E)*T_s):
                            self.E_m[c] = E
                            indexes.append(unlabeled_idx[idx])
                            b+=1
            epoch+=1
            if epoch==0 or epoch==self.epochs-1:
                return indexes, epoch
        return indexes, epoch

    
class ClusterMargin:
    
    def __init__(self, model, X, k, p, eps):
        self.k=k
        self.p=p
        self.eps=eps
        self.labels, self.asc = self.make_clusters(model, X, eps)
    
    def make_clusters(self, model, X, eps):
        layer_name = 'embedding'
        intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
        out = intermediate_layer_model(X)
        dist = pdist(np.array(out))
        cluster = linkage(dist, 'average')
        labels = fcluster(cluster, t=eps, criterion='distance')
        c = Counter(labels)
        asc = c.most_common()
        return labels, asc

    def query(self, active_learner, X, labeled_idx, epoch):
        unlabeled_idx = np.arange(X.shape[0])[np.logical_not(np.in1d(np.arange(X.shape[0]), labeled_idx))]
        cout = np.array(active_learner(X[unlabeled_idx]))
        L = np.argsort(-cout, axis=1)
        cprob = np.take_along_axis(cout, L, axis=1)
        marg = cprob[:, 0] - cprob[:, 1]
        ind = np.argpartition(marg, self.p)[:self.p]

        b = 0
        clus_i = len(self.asc)-1
        indexes = []
        while b < self.k:

            if np.where(self.labels[unlabeled_idx[ind]] == self.asc[clus_i][0])[0].size != 0:
                index = np.random.choice(np.where(self.labels[unlabeled_idx[ind]] == self.asc[clus_i][0])[0], size=1).tolist()
                indexes.extend(unlabeled_idx[ind[index]])
                b+=1

            if clus_i==0:
                clus_i = len(self.asc)-1
            else:
                clus_i-=1
        return indexes, epoch+1


class BADGE:

    def __init__(self, k, p):
        self.k=k
        self.p=p
        
    def get_grad_embedding(self, model, intermediate_layer_model, X, n):
        embDim = 120
        nLab = 10
        embedding = np.zeros([n, embDim * nLab])
        cout = model(X)
        out = intermediate_layer_model(X)
        maxInds = np.argmax(cout, axis=1).flatten()
        for j in range(n):
            for c in range(nLab):
                if c==maxInds[j]:
                    embedding[j, embDim * c : embDim * (c+1)] = out[j] * (1 - cout[j][c])
                else:
                    embedding[j, embDim * c : embDim * (c+1)] = out[j] * (-1 * cout[j][c])
        return embedding
    
    def init_centers(self, X, K):
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] >  newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2)/ sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

    def query(self, active_learner, X, labeled_idx, epoch):
        unlabeled_idx = np.arange(X.shape[0])[np.logical_not(np.in1d(np.arange(X.shape[0]), labeled_idx))]
        
        layer_name = 'embedding'
        intermediate_layer_model = Model(inputs=active_learner.input,
                                    outputs=active_learner.get_layer(layer_name).output)

        idx = np.random.choice(unlabeled_idx, size=self.p)
        gradEmbed = self.get_grad_embedding(active_learner, intermediate_layer_model, X[idx], self.p)
        ind = self.init_centers(gradEmbed, self.k)
        return idx[ind], epoch+1


class BALD:
    
    def __init__(self, k):
        self.k=k
    
    def Compute_Disagreement(self, model, X, n, mc=10):
        prob = np.stack([model(X, training=True) for _ in range(mc)])
        pb = np.mean(prob, axis=0)
        entropy1 = (-pb*np.log(pb)).sum(axis=1)
        entropy2 = (-prob*np.log(prob)).sum(axis=2).mean(axis=0)
        un = entropy2 - entropy1
        return np.argpartition(un, n)[:n]
    
    def query(self, active_learner, X, labeled_idx, epoch):
        unlabeled_idx = np.arange(X.shape[0])[np.logical_not(np.in1d(np.arange(X.shape[0]), labeled_idx))]
        indexes = self.Compute_Disagreement(active_learner, X[unlabeled_idx], self.k).tolist()
        return unlabeled_idx[indexes], epoch+1
    
class CoreSet:
    
    def __init__(self, k, p):
        self.k=k
        self.p=p
    
    def greedy_k_center(self, labeled, unlabeled, amount):

        greedy_indices = []

        min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j+100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for i in range(amount-1):
            dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        return np.array(greedy_indices)
    
    def query(self, active_learner, X, labeled_idx, epoch):
        unlabeled_idx = np.arange(X.shape[0])[np.logical_not(np.in1d(np.arange(X.shape[0]), labeled_idx))]
        active_idx = np.random.choice(unlabeled_idx, size=self.p)
        representation = active_learner.predict(X, verbose=0)
        new_indices = self.greedy_k_center(representation[labeled_idx, :], representation[active_idx, :], self.k)
        return active_idx[new_indices], epoch+1
    
class DBAL:
    
    def __init__(self, k):
        self.k=k
    
    def MaxEntropy(self, model, X, n, mc=10):
        prob = np.stack([model(X, training=True) for _ in range(mc)])
        pb = np.mean(prob, axis=0)
        entropy = (-pb*np.log(pb)).sum(axis=1)
        return np.argpartition(entropy, -n)[-n:]

    def query(self, active_learner, X, labeled_idx, epoch):
        unlabeled_idx = np.arange(X.shape[0])[np.logical_not(np.in1d(np.arange(X.shape[0]), labeled_idx))]
        indexes = self.MaxEntropy(active_learner, X[unlabeled_idx], self.k).tolist()
        return unlabeled_idx[indexes], epoch+1

class Random:
    
    def __init__(self, k):
        self.k=k
    
    def query(self, active_learner, X, labeled_idx, epoch):
        unlabeled_idx = np.arange(X.shape[0])[np.logical_not(np.in1d(np.arange(X.shape[0]), labeled_idx))]
        indexes = np.random.choice(unlabeled_idx, size=self.k).tolist()
        return indexes, epoch+1

class ClusterSampling:
    
    def __init__(self, X, m, k, p):
        self.m = m
        self.k = k
        self.p = p
        self.cluster_labels, self.cluster_centers = self.make_clusters(X, m)
                 
    def make_clusters(self, X, m):
        print(F"KMeans Clustering with {m} clusters")
        cluster = KMeans(n_clusters=m)
        cluster.fit(X.reshape(X.shape[0], X.shape[1]*X.shape[2]))
        X_cluster = cluster.labels_
        centers = cluster.cluster_centers_ 

        return X_cluster, centers
    
    def query(self, active_learner, X, labeled_idx, epoch):
        unlabeled_idx = np.arange(X.shape[0])[np.logical_not(np.in1d(np.arange(X.shape[0]), labeled_idx))]
        indexes = []
        for clus in range(self.m):
            index = np.random.choice(np.where(self.cluster_labels[unlabeled_idx]==clus)[0], size=self.p)
            prob = np.mean([active_learner(X[unlabeled_idx[index]].reshape(-1, X.shape[1], X.shape[2], 1), training=True) for _ in range(10)], axis=0)
            L = np.argsort(-prob, axis=1)
            cprob = np.take_along_axis(prob, L, axis=1)
            marg = cprob[:, 0] - cprob[:, 1]
            ind = np.argpartition(marg, self.k)[:self.k]
            indexes.extend((unlabeled_idx[index[ind.tolist()]]).tolist())
        
        return indexes, epoch+1

class UALSA:

    def __init__(self, X, alpha, beta, e, m, k, p, epochs) -> None:
        self.alpha = alpha
        self.beta = beta
        self.m = m
        self.k = k
        self.p = p
        self.epochs = epochs
        self.T = np.exp(-np.array(range(epochs))/(epochs)*beta)
        self.cluster_labels, self.cluster_centers = self.make_clusters(X, m)
        self.E_m = [e]*k
    
    def make_clusters(self, X, m):
        print(F"KMeans Clustering with {m} clusters")
        cluster = KMeans(n_clusters=m)
        cluster.fit(X.reshape(X.shape[0], X.shape[1]*X.shape[2]))
        X_cluster = cluster.labels_
        centers = cluster.cluster_centers_ 

        return X_cluster, centers

    def query(self, active_learner, X, labeled_idx, epoch):
        unlabeled_idx = np.arange(X.shape[0])[np.logical_not(np.in1d(np.arange(X.shape[0]), labeled_idx))]
        indexes = []
        b=0

        while b<self.k:
            T_s = self.T[epoch]
            for c in range(self.m):
                index = np.random.choice(np.where(self.cluster_labels[unlabeled_idx]==c)[0], size=self.p)

                for idx in index:

                    prob = np.mean([active_learner(X[unlabeled_idx[idx]].reshape(1, X.shape[1], X.shape[2], 1), training=True) for _ in range(10)], axis=0).flatten()
                    marg = np.argpartition(prob, -2)[-2:]
                    U = 1 - abs(prob[marg[0]]-prob[marg[1]])

                    E = U

                    if E  > self.E_m[c]:
                        self.E_m[c] = E
                        indexes.append(unlabeled_idx[idx])
                        b+=1

                    else:
                        if np.random.uniform() < np.exp(-self.alpha*(self.E_m[c] - E)*T_s):
                            self.E_m[c] = E
                            indexes.append(unlabeled_idx[idx])
                            b+=1
            epoch+=1
            if epoch==0 or epoch==self.epochs-1:
                return indexes, epoch
        return indexes, epoch

class RandomPoolALSA:

    def __init__(self, X, alpha, beta, e, m, k, p, epochs) -> None:
        self.alpha = alpha
        self.beta = beta
        self.m = m
        self.k = k
        self.p = p
        self.epochs = epochs
        self.T = np.exp(-np.array(range(epochs))/(epochs)*beta)
        self.cluster_labels, self.cluster_centers = self.make_clusters(X, m)
        self.E_m = [e]*k
    
    def make_clusters(self, X, m):
        print(F"KMeans Clustering with {m} clusters")
        cluster = KMeans(n_clusters=m)
        cluster.fit(X.reshape(X.shape[0], X.shape[1]*X.shape[2]))
        X_cluster = cluster.labels_
        centers = cluster.cluster_centers_ 

        return X_cluster, centers

    def query(self, active_learner, X, labeled_idx, epoch):
        unlabeled_idx = np.arange(X.shape[0])[np.logical_not(np.in1d(np.arange(X.shape[0]), labeled_idx))]
        indexes = []
        b=0

        while b<self.k:
            T_s = self.T[epoch]
            for c in range(self.m):
                index = np.random.choice(unlabeled_idx, size=self.p)

                for idx in index:

                    dist = np.linalg.norm(X[idx].reshape(X.shape[1]*X.shape[2],) - self.cluster_centers[self.cluster_labels[idx]])
                    div = 1 / (1 + dist)
                    mi = 1 / (1 + np.sqrt(X.shape[1]*X.shape[2]))
                    div = (div - mi)/(1 - mi)
            
                    prob = np.mean([active_learner(X[idx].reshape(1, X.shape[1], X.shape[2], 1), training=True) for _ in range(10)], axis=0).flatten()
                    marg = np.argpartition(prob, -2)[-2:]
                    U = 1 - abs(prob[marg[0]]-prob[marg[1]])

                    E = T_s * div + (1-T_s) * U

                    if E  > self.E_m[c]:
                        self.E_m[c] = E
                        indexes.append(idx)
                        b+=1

                    else:
                        if np.random.uniform() < np.exp(-self.alpha*(self.E_m[c] - E)*T_s):
                            self.E_m[c] = E
                            indexes.append(idx)
                            b+=1
            epoch+=1
            if epoch==0 or epoch==self.epochs-1:
                return indexes, epoch
        return indexes, epoch