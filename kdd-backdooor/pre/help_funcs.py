import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj,dense_to_sparse
import torch
from models.reconstruct import MLPAE
from models.str_reconstruct import str_MLPAE
import scipy.sparse as sp
import faiss
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
from torch_geometric.utils import subgraph
def edge_sim_analysis(edge_index, features):
    sims = []
    for (u,v) in edge_index:
        sims.append(float(F.cosine_similarity(features[u].unsqueeze(0),features[v].unsqueeze(0))))
    sims = np.array(sims)
    # print(f"mean: {sims.mean()}, <0.1: {sum(sims<0.1)}/{sims.shape[0]}")
    return sims

def prune_unrelated_edge(args,edge_index,edge_weights,x,device,large_graph=True):
    edge_index = edge_index[:,edge_weights>0.0].to(device)
    edge_weights = edge_weights[edge_weights>0.0].to(device)
    x = x.to(device)
    # calculate edge simlarity
    if(large_graph):
        edge_sims = torch.tensor([],dtype=float).cpu()
        N = edge_index.shape[1]
        num_split = 100
        N_split = int(N/num_split)
        for i in range(num_split):
            if(i == num_split-1):
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:]],x[edge_index[1][N_split * i:]]).cpu()
            else:
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:N_split*(i+1)]],x[edge_index[1][N_split * i:N_split*(i+1)]]).cpu()
            # print(edge_sim1)
            edge_sim1 = edge_sim1.cpu()
            edge_sims = torch.cat([edge_sims,edge_sim1])
        # edge_sims = edge_sims.to(device)
    else:
        edge_sims = F.cosine_similarity(x[edge_index[0]],x[edge_index[1]])
    # find dissimilar edges and remote them
    # update structure
    updated_edge_index = edge_index[:,edge_sims>args.prune_thr]
    updated_edge_weights = edge_weights[edge_sims>args.prune_thr]
    return updated_edge_index,updated_edge_weights

def clu_prune_unrelated_edge(args,edge_index,edge_weights,x,device,large_graph=True):
    # edge_index = edge_index[:,edge_weights>0.0].to(device)
    # edge_weights = edge_weights[edge_weights>0.0].to(device)
    edge_index = edge_index.to(device)
    edge_weights = edge_weights.to(device)
    x = x.to(device)

    kmeans = faiss.Kmeans(x.shape[1], 2, niter=20) 
    kmeans.train(x.cpu().detach().numpy())
    centroids = torch.FloatTensor(kmeans.centroids).to(x.device)

    # Assign clusters
    D, I = kmeans.index.search(x.cpu().detach().numpy(), 1)

    # Count the number of samples in each cluster
    cluster_counts = np.bincount(I.squeeze())

    # Determine which cluster has more samples
    dominant_cluster = np.argmax(cluster_counts)

    last = len(edge_weights)-1

    indomain = []
    # Output cluster labels and store indices for the dominant cluster
    for idx, label in enumerate(I):
        if label[0] == dominant_cluster:
            indomain.append(idx)
    # print(indomain)

    # Convert indomain to a tensor
    indomain_tensor = torch.tensor(indomain, device=edge_index.device)

    # Create the mask
    mask = torch.isin(edge_index, indomain_tensor).all(dim=0)

    # Print the mask to see what it looks like
    # print("Mask:", mask)

    # Apply the mask to edge_index and edge_weights
    updated_edge_index = edge_index[:, mask]
    updated_edge_weights = edge_weights[mask]
    # updated_edge_index = edge_index[:, :-2]
    # updated_edge_weights = edge_weights[:-2]

    print("Updated edge_index:", updated_edge_index)
    print("Updated edge_weights:", updated_edge_weights)
    # updated_edge_index = edge_index[:,:-1]
    # updated_edge_weights = edge_weights[:-1]
    return updated_edge_index,updated_edge_weights

def reconstruct_prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,ori_x,ori_edge_index,device, idx, large_graph=True):
    # edge_index = edge_index[:,edge_weights>0.0].to(device)
    # edge_weights = edge_weights[edge_weights>0.0].to(device)
    edge_index = poison_edge_index.to(device)
    edge_weights = poison_edge_weights.to(device)
    poison_x = poison_x.to(device)
    # ori_x = ori_x[:20000]
    # poison_x = poison_x[100000:]
    num_nodes = ori_x.size(0)  # Total number of nodes
    subset = torch.randperm(num_nodes)[:60000]  # Randomly select 10 nodes
    ori_edge_index, _ = subgraph(subset, ori_edge_index, relabel_nodes=True)
    ori_x_ = ori_x[subset]

    # AE = MLPAE(poison_x, poison_x[len(ori_x):], device, args.rec_epochs)
    AE = MLPAE(poison_x, poison_x[len(ori_x):], device, args.rec_epochs)
    AE.fit()
    # AE.eval()
    # rec_score_ori = AE.inference(ori_x)
    rec_score_ori = AE.inference(poison_x)


    # plt.hist(rec_score_ori, bins=30, alpha=0.7, color='blue')

    # plt.xlabel('Element Value')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Elements in the List')
    # plt.grid(True)
    # # plt.show()
    # plt.savefig('score.png')


    print(torch.mean(rec_score_ori))
    # print(rec_score_ori)
    rec_score_triggers = AE.inference(poison_x[len(ori_x):])
    # print(rec_score)
    print(torch.mean(rec_score_triggers))

    # Calculate mean and standard deviation of in-domain scores
    mean_ori = torch.mean(rec_score_ori)
    std_ori = torch.std(rec_score_ori)

    threshold = 2 * std_ori

    # Function to assess in-domain likelihood
    def is_in_domain(score, mean, threshold):
        return score < mean or abs(score - mean) <= threshold
        # return torch.abs(score - mean) <= threshold
    # Assess each score in rec_score_triggers
    for score in rec_score_triggers:
        if is_in_domain(score, mean_ori, threshold):
            print(f"Score: {score} is likely in-domain.")
        else:
            print(f"Score: {score} is likely out-of-domain.")
    print('mean_ori',mean_ori)
    print('std_ori',std_ori)

    # def is_in_domain(score, mean, threshold):
    #     return abs(score - mean) <= threshold

    # Count in-domain scores
    in_domain_count = sum(is_in_domain(score, mean_ori, threshold) for score in rec_score_triggers)

    # Calculate the percentage of in-domain scores
    percentage_in_domain = (in_domain_count / len(rec_score_triggers)) * 100

    print(f"Percentage of scores likely in-domain: {percentage_in_domain}%")

    np.save('ori.npy', rec_score_ori[:len(ori_x)].detach().cpu().numpy())
    np.save('poison.npy', rec_score_ori[len(ori_x):].detach().cpu().numpy())

    ori = rec_score_ori[:len(ori_x)].detach().cpu().numpy()
    poison = rec_score_ori[len(ori_x):].detach().cpu().numpy()

    losses = np.concatenate((ori, poison))
    labels = np.concatenate((np.zeros(len(ori)), np.ones(len(poison))))
    from sklearn.metrics import roc_auc_score

# Calculate AUC-ROC
    auc_roc = roc_auc_score(labels, losses)
    print('auc_roc',auc_roc)

    # Calculate the threshold for the top 3% largest values in rec_score_ori
    threshold = np.percentile(rec_score_ori.detach().cpu().numpy(), 97)

    mask = rec_score_ori>threshold

    keep_edges_mask = ~(mask[poison_edge_index[0]] | mask[poison_edge_index[1]])
    
    # Filter the edge_index by the edges we want to keep
    filtered_poison_edge_index = poison_edge_index[:, keep_edges_mask]
    
    # Filter the edge weights similarly
    filtered_poison_edge_weights = poison_edge_weights[keep_edges_mask]

    

    # Check each element in poison against this threshold
    top_3_percent_flag = poison >= threshold

    # Calculate the percentage of poison elements that are in the top 3%
    percentage_in_top_3 = np.mean(top_3_percent_flag) * 100  # Convert to percentage

    print('percentage_in_top_3',percentage_in_top_3)

    threshold = np.percentile(rec_score_ori.detach().cpu().numpy(), 98)

    # Check each element in poison against this threshold
    top_2_percent_flag = poison >= threshold

    # Calculate the percentage of poison elements that are in the top 3%
    percentage_in_top_2 = np.mean(top_2_percent_flag) * 100  # Convert to percentage

    print('percentage_in_top_2',percentage_in_top_2)

    threshold = np.percentile(rec_score_ori.detach().cpu().numpy(), 99)

    # Check each element in poison against this threshold
    top_1_percent_flag = poison >= threshold

    # Calculate the percentage of poison elements that are in the top 3%
    percentage_in_top_1 = np.mean(top_1_percent_flag) * 100  # Convert to percentage

    print('percentage_in_top_1',percentage_in_top_1)
    
    rec_score_ori = rec_score_ori[:len(ori_x)].detach().cpu().numpy()
    
    plt.hist(rec_score_ori, bins=30, alpha=0.7, color='blue')

    plt.xlabel('Element Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Elements in the List')
    plt.grid(True)
    # plt.show()
    plt.savefig('score.png')


    # kmeans = faiss.Kmeans(x.shape[1], 2, niter=20) 
    # kmeans.train(x.cpu().detach().numpy())
    # centroids = torch.FloatTensor(kmeans.centroids).to(x.device)

    # # Assign clusters
    # D, I = kmeans.index.search(x.cpu().detach().numpy(), 1)

    # # Count the number of samples in each cluster
    # cluster_counts = np.bincount(I.squeeze())

    # # Determine which cluster has more samples
    # dominant_cluster = np.argmax(cluster_counts)

    # last = len(edge_weights)-1

    # indomain = []
    # # Output cluster labels and store indices for the dominant cluster
    # for idx, label in enumerate(I):
    #     if label[0] == dominant_cluster:
    #         indomain.append(idx)
    # # print(indomain)

    # # Convert indomain to a tensor
    # indomain_tensor = torch.tensor(indomain, device=edge_index.device)

    # # Create the mask
    # mask = torch.isin(edge_index, indomain_tensor).all(dim=0)

    # # Print the mask to see what it looks like
    # # print("Mask:", mask)

    # # Apply the mask to edge_index and edge_weights
    # updated_edge_index = edge_index[:, mask]
    # updated_edge_weights = edge_weights[mask]
    # # updated_edge_index = edge_index[:, :-2]
    # # updated_edge_weights = edge_weights[:-2]

    # print("Updated edge_index:", updated_edge_index)
    # print("Updated edge_weights:", updated_edge_weights)
    # updated_edge_index = edge_index[:,:-1]
    # updated_edge_weights = edge_weights[:-1]
    return filtered_poison_edge_index,filtered_poison_edge_weights

def str_reconstruct_prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,ori_x, ori_edge_index,device, idx, large_graph=True):
    # edge_index = poison_edge_index.to(device)
    edge_weights = poison_edge_weights.to(device)
    poison_x = poison_x.to(device)

    # ori_x
    # ori_edge_index
    # poison_x
    # poison_edge_index
    num_nodes = ori_x.size(0)  # Total number of nodes
    subset = torch.randperm(num_nodes)[:10000]  # Randomly select 10 nodes
    ori_edge_index, _ = subgraph(subset, ori_edge_index, relabel_nodes=True)
    ori_x = ori_x[subset]

    num_nodes = poison_x.size(0)  # Total number of nodes in poison_x
    start_index = max(0, num_nodes - 10000)  # Start index for the last 10,000 nodes
    end_index = num_nodes  # End index
    subset_indices = torch.arange(start_index, end_index)  # Node indices for the subset
    poison_edge_index, _ = subgraph(subset_indices, poison_edge_index, relabel_nodes=True)
    poison_x = poison_x[subset_indices]

    # AE = MLPAE(poison_x, poison_x[len(ori_x):], device, epochs=15)
    # AE = str_MLPAE(poison_x, poison_x[len(ori_x):], edge_index, poison_edge_index,  device, args.rec_epochs)
    AE = str_MLPAE(poison_x, ori_x, poison_edge_index, ori_edge_index, device, args.rec_epochs)
    AE.fit()
    # AE.eval()
    # rec_score_ori = AE.inference(ori_x)
    rec_score_ori = AE.inference(poison_x)


    # plt.hist(rec_score_ori, bins=30, alpha=0.7, color='blue')

    # plt.xlabel('Element Value')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Elements in the List')
    # plt.grid(True)
    # # plt.show()
    # plt.savefig('score.png')


    # print(torch.mean(rec_score_ori))
    # print(rec_score_ori)
    rec_score_triggers = AE.inference(poison_x[len(ori_x):])
    # print(rec_score)
    # print(torch.mean(rec_score_triggers))

    # Calculate mean and standard deviation of in-domain scores
    mean_ori = torch.mean(rec_score_ori)
    std_ori = torch.std(rec_score_ori)

    threshold = 2 * std_ori

    # Function to assess in-domain likelihood
    def is_in_domain(score, mean, threshold):
        # return score < mean or abs(score - mean) <= threshold
        return torch.abs(score - mean) <= threshold
    # Assess each score in rec_score_triggers
    for score in rec_score_triggers:
        if is_in_domain(score, mean_ori, threshold):
            print(f"Score: {score} is likely in-domain.")
        else:
            print(f"Score: {score} is likely out-of-domain.")
    print('mean_ori',mean_ori)
    print('std_ori',std_ori)

    # def is_in_domain(score, mean, threshold):
    #     return abs(score - mean) <= threshold

    # Count in-domain scores
    in_domain_count = sum(is_in_domain(score, mean_ori, threshold) for score in rec_score_triggers)

    # Calculate the percentage of in-domain scores
    percentage_in_domain = (in_domain_count / len(rec_score_triggers)) * 100

    print(f"Percentage of scores likely in-domain: {percentage_in_domain}%")

    rec_score_ori = rec_score_ori[:len(ori_x)].detach().cpu().numpy()
    plt.hist(rec_score_ori, bins=30, alpha=0.7, color='blue')

    plt.xlabel('Element Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Elements in the List')
    plt.grid(True)
    # plt.show()
    plt.savefig('score.png')


    # kmeans = faiss.Kmeans(x.shape[1], 2, niter=20) 
    # kmeans.train(x.cpu().detach().numpy())
    # centroids = torch.FloatTensor(kmeans.centroids).to(x.device)

    # # Assign clusters
    # D, I = kmeans.index.search(x.cpu().detach().numpy(), 1)

    # # Count the number of samples in each cluster
    # cluster_counts = np.bincount(I.squeeze())

    # # Determine which cluster has more samples
    # dominant_cluster = np.argmax(cluster_counts)

    # last = len(edge_weights)-1

    # indomain = []
    # # Output cluster labels and store indices for the dominant cluster
    # for idx, label in enumerate(I):
    #     if label[0] == dominant_cluster:
    #         indomain.append(idx)
    # # print(indomain)

    # # Convert indomain to a tensor
    # indomain_tensor = torch.tensor(indomain, device=edge_index.device)

    # # Create the mask
    # mask = torch.isin(edge_index, indomain_tensor).all(dim=0)

    # # Print the mask to see what it looks like
    # # print("Mask:", mask)

    # # Apply the mask to edge_index and edge_weights
    # updated_edge_index = edge_index[:, mask]
    # updated_edge_weights = edge_weights[mask]
    # # updated_edge_index = edge_index[:, :-2]
    # # updated_edge_weights = edge_weights[:-2]

    # print("Updated edge_index:", updated_edge_index)
    # print("Updated edge_weights:", updated_edge_weights)
    # updated_edge_index = edge_index[:,:-1]
    # updated_edge_weights = edge_weights[:-1]
    return updated_edge_index,updated_edge_weights

def prune_unrelated_edge_isolated(args,edge_index,edge_weights,x,device,large_graph=True):
    edge_index = edge_index[:,edge_weights>0.0].to(device)
    edge_weights = edge_weights[edge_weights>0.0].to(device)
    x = x.to(device)
    # calculate edge simlarity
    if(large_graph):
        edge_sims = torch.tensor([],dtype=float).cpu()
        N = edge_index.shape[1]
        num_split = 100
        N_split = int(N/num_split)
        for i in range(num_split):
            if(i == num_split-1):
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:]],x[edge_index[1][N_split * i:]]).cpu()
            else:
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:N_split*(i+1)]],x[edge_index[1][N_split * i:N_split*(i+1)]]).cpu()
            # print(edge_sim1)
            edge_sim1 = edge_sim1.cpu()
            edge_sims = torch.cat([edge_sims,edge_sim1])
        # edge_sims = edge_sims.to(device)
    else:
        # calculate edge simlarity
        edge_sims = F.cosine_similarity(x[edge_index[0]],x[edge_index[1]])
    # find dissimilar edges and remote them
    dissim_edges_index = np.where(edge_sims.cpu()<=args.prune_thr)[0]
    edge_weights[dissim_edges_index] = 0
    # select the nodes between dissimilar edgesy
    dissim_edges = edge_index[:,dissim_edges_index]    # output: [[v_1,v_2],[u_1,u_2]]
    dissim_nodes = torch.cat([dissim_edges[0],dissim_edges[1]]).tolist()
    dissim_nodes = list(set(dissim_nodes))
    # update structure
    updated_edge_index = edge_index[:,edge_weights>0.0]
    updated_edge_weights = edge_weights[edge_weights>0.0]
    return updated_edge_index,updated_edge_weights,dissim_nodes 

def select_target_nodes(args,seed,model,features,edge_index,edge_weights,labels,idx_val,idx_test):
    test_ca,test_correct_index = model.test_with_correct_nodes(features,edge_index,edge_weights,labels,idx_test)
    test_correct_index = test_correct_index.tolist()
    '''select target test nodes'''
    test_correct_nodes = idx_test[test_correct_index].tolist()
    # filter out the test nodes that are not in target class
    target_class_nodes_test = [int(nid) for nid in idx_test
            if labels[nid]==args.target_class] 
    # get the target test nodes
    idx_val,idx_test = idx_val.tolist(),idx_test.tolist()
    rs = np.random.RandomState(seed)
    cand_atk_test_nodes = list(set(test_correct_nodes) - set(target_class_nodes_test))  # the test nodes not in target class is candidate atk_test_nodes
    atk_test_nodes = rs.choice(cand_atk_test_nodes, args.target_test_nodes_num)
    '''select clean test nodes'''
    cand_clean_test_nodes = list(set(idx_test) - set(atk_test_nodes))
    clean_test_nodes = rs.choice(cand_clean_test_nodes, args.clean_test_nodes_num)
    '''select poisoning nodes from unlabeled nodes (assign labels is easier than change, also we can try to select from labeled nodes)'''
    N = features.shape[0]
    cand_poi_train_nodes = list(set(idx_val)-set(atk_test_nodes)-set(clean_test_nodes))
    poison_nodes_num = int(N * args.vs_ratio)
    poi_train_nodes = rs.choice(cand_poi_train_nodes, poison_nodes_num)
    
    return atk_test_nodes, clean_test_nodes,poi_train_nodes

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
    
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()