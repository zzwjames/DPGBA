#%%
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from models.MLP import MLP
from models.GCN import GCN
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from models.reconstruct import MLPAE
#%%
from sklearn.manifold import TSNE

class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input>thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None

class GraphTrojanNet(nn.Module):
    # In the furture, we may use a GNN model to generate backdoor
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers.append(nn.Linear(nfeat, nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        
        self.layers = nn.Sequential(*layers).to(device)

        self.feat = nn.Linear(nfeat,nout*(nfeat))
        # self.pattern = nn.Linear(nfeat,40)
        self.edge = nn.Linear(nfeat, int(nout*(nout-1)/2))
        self.device = device

    def forward(self, input, thrd):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """

        GW = GradWhere.apply
        self.layers = self.layers
        h = self.layers(input)

        feat = self.feat(h)
        

        edge_weight = self.edge(h)

        edge_weight = GW(edge_weight, thrd, self.device)

        return feat, edge_weight


class HomoLoss(nn.Module):
    def __init__(self,args,device):
        super(HomoLoss, self).__init__()
        self.args = args
        self.device = device
        
    def forward(self,trigger_edge_index,trigger_edge_weights,x,thrd):

        trigger_edge_index = trigger_edge_index[:,trigger_edge_weights>0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]],x[trigger_edge_index[1]])
        
        loss = torch.relu(thrd - edge_sims).mean()
        # print(edge_sims.min())
        return loss

#%%
import numpy as np
class Backdoor:

    def __init__(self,args, device):
        self.args = args
        self.device = device
        self.weights = None
        self.trigger_index = self.get_trigger_index(args.trigger_size)

        
    def get_trigger_index(self,trigger_size):
        edge_list = []
        edge_list.append([0,0])
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j,k])
        edge_index = torch.tensor(edge_list,device=self.device).long().T
        # print('edge_index',edge_index)
        # edge_index tensor([[0, 1, 2, 2],
        # [0, 0, 0, 1]], device='cuda:0')
        return edge_index
    
    def calculate_mean_cosine_similarity(self,trojan_feat):
        n = trojan_feat.size(0)  # Number of samples

        # Initialize a tensor to store cosine similarities
        similarities = torch.zeros((n, n))

        # Calculate cosine similarity for each pair of rows
        for i in range(n):
            for j in range(i + 1, n):
                sim = cosine_similarity(trojan_feat[i].unsqueeze(0), trojan_feat[j].unsqueeze(0))
                similarities[i, j] = sim
                similarities[j, i] = sim  # The similarity matrix is symmetric

        # Exclude self-similarities and calculate mean
        similarities.fill_diagonal_(0)
        mean_similarity = similarities.sum() / (n * (n - 1))

        return mean_similarity.item()

    def get_trojan_edge(self,start, idx_attach, trigger_size):
        edge_list = []
        for idx in idx_attach:
            edges = self.trigger_index.clone()
            edges[0,0] = idx
            edges[1,0] = start
            edges[:,1:] = edges[:,1:] + start

            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list,dim=1)
        # to undirected
        # row, col = edge_index
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1],edge_index[0]])
        edge_index = torch.stack([row,col])

        return edge_index

    # def get_trojan_edge(self, start, idx_attach, trigger_size):
    #     edge_list = []
    #     for idx in idx_attach:
    #         # Copy the subgraph structure and update indices
    #         edges = self.trigger_index.clone()
    #         edges = edges[:,1:]
    #         edges+=start
    #         # edges[0,0] = idx

    #         # Add updated subgraph to the edge list
    #         edge_list.append(edges)
    #         # edge_list.append(torch.stack([edges[1],edges[0]]))
    #         # Connect each node in the subgraph to the target node 'idx'
    #         for i in range(trigger_size):
    #             # edge_list.append(torch.tensor([[idx, start + i], [start + i, idx]], device=self.device))
    #             edge_list.append(torch.tensor([[idx], [start + i]], device=self.device))

    #         # Increment 'start' for the next set of subgraph nodes
    #         # print(edge_list)
    #         start += trigger_size

    #     # Convert the list of tensors to a single tensor
    #     edge_index = torch.cat(edge_list, dim=1)
    #     row = torch.cat([edge_index[0], edge_index[1]])
    #     col = torch.cat([edge_index[1],edge_index[0]])
    #     edge_index = torch.stack([row,col])

    #     return edge_index

        
    def inject_trigger(self, idx_attach, features,edge_index,edge_weight,device):
        self.trojan = self.trojan.to(device)
        idx_attach = idx_attach.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        self.trojan.eval()

        trojan_feat, trojan_weights = self.trojan(features[idx_attach],self.args.thrd) # may revise the process of generate
        
        trojan_weights = torch.cat([torch.ones([len(idx_attach),1],dtype=torch.float,device=device),trojan_weights],dim=1)
        trojan_weights = trojan_weights.flatten()

        trojan_feat = trojan_feat.view([-1,features.shape[1]])

        trojan_edge = self.get_trojan_edge(len(features),idx_attach,self.args.trigger_size).to(device)

        update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
        update_feat = torch.cat([features,trojan_feat])
        update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

        self.trojan = self.trojan.cpu()
        idx_attach = idx_attach.cpu()
        features = features.cpu()
        edge_index = edge_index.cpu()
        edge_weight = edge_weight.cpu()
        return update_feat, update_edge_index, update_edge_weights
    # def inject_trigger(self, idx_attach, features,edge_index,edge_weight, device):
    #     self.trojan = self.trojan.to(device)
    #     idx_attach = idx_attach.to(device)
    #     features = features.to(device)
    #     edge_index = edge_index.to(device)
    #     edge_weight = edge_weight.to(device)
    #     self.trojan.eval()

    #     trojan_feat, trojan_weights = self.trojan(features[idx_attach],self.args.thrd) # may revise the process of generate
        
    #     trojan_weights = torch.cat([trojan_weights,torch.ones([len(trojan_feat),1*self.args.trigger_size],dtype=torch.float,device=self.device)],dim=1)
    #     trojan_weights = trojan_weights.flatten()

    #     trojan_feat = trojan_feat.view([-1,features.shape[1]])

    #     trojan_edge = self.get_trojan_edge(len(features),idx_attach,self.args.trigger_size).to(device)

    #     update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
    #     update_feat = torch.cat([features,trojan_feat])
    #     update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

    #     self.trojan = self.trojan.cpu()
    #     idx_attach = idx_attach.cpu()
    #     features = features.cpu()
    #     edge_index = edge_index.cpu()
    #     edge_weight = edge_weight.cpu()
    #     return update_feat, update_edge_index, update_edge_weights

    def visualize_embeddings_sne(self, features, labels, idx_attach):
        if features.is_cuda:
            features = features.cpu()

        # Convert features and labels to numpy
        features = features.detach().numpy()
        labels = labels.cpu().detach().numpy()

        # print(len(features))
        # print(len(labels))

        # Reduce the dimensionality of features to 2D using t-SNE
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)  # Adjust perplexity and learning_rate as needed
        features_2d = tsne.fit_transform(features)

        # Plotting
        plt.figure(figsize=(8, 6))

        unique_labels = list(set(labels))  # Get unique class labels

        last_30 = features_2d[-len(idx_attach)*self.args.trigger_size:]  # Last 30 rows
        rest = features_2d[:-len(idx_attach)*self.args.trigger_size]    # All other rows

        target_nodes = features_2d[-len(idx_attach)*self.args.trigger_size-len(idx_attach):-len(idx_attach)*self.args.trigger_size]

        # indices = []
        # for i in range(3, 33, 3):  # Start from 0, up to 29, step by 3
        #     indices.append(-i)

        # Plot the last 30 rows with one color
        plt.scatter(last_30[:, 0], last_30[:, 1], color='red', label='Trigger', alpha=0.5)
        plt.scatter(target_nodes[:, 0], target_nodes[:, 1], color='blue', label='Target Nodes', alpha=0.5)

        # Assign unique colors to each class
        color_map = plt.cm.get_cmap('viridis', len(unique_labels))

        for label in unique_labels:
            # Get the indices of samples with the current label
            label_indices = [i for i, l in enumerate(labels) if l == label]

            # Extract the 2D features for samples with the current label
            label_features = rest[label_indices]

            # Plot the samples with the same label using a unique color
            plt.scatter(label_features[:, 0], label_features[:, 1], label=f'Class {label}', cmap=color_map, alpha=0.5)

        # Split the data into two groups
        

        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title('2D Visualization of Embeddings with Each Class in Labels Colored Separately')
        plt.legend()
        
        if len(features[0]) == self.args.hidden:
            plt.savefig('./highlighted_embeddings.png')
        else:
            plt.savefig('./highlighted_features.png')

    def visualize_embeddings(self, features, labels, idx_attach):
        if features.is_cuda:
            features = features.cpu()

        # Convert features to numpy
        features = features.detach().numpy()

        # Reduce the dimensionality of features to 2D using PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        # Split the data into two groups
        last_30 = features_2d[-len(idx_attach)*self.args.trigger_size:]  # Last 30 rows
        rest = features_2d[:-len(idx_attach)*self.args.trigger_size]    # All other rows
        # last_30 = features_2d[-len(idx_attach)*2:]  # Last 30 rows
        # rest = features_2d[:-len(idx_attach)*2]    # All other rows

        indices = []
        for i in range(3, 33, 3):  # Start from 0, up to 29, step by 3
            indices.append(-i)


        # Plotting
        plt.figure(figsize=(8, 6))

        # Plot for clean samples
        plt.scatter(rest[:, 0], rest[:, 1], color='cornflowerblue', label='Clean', s=20)
        plt.scatter(last_30[:, 0], last_30[:, 1], color='red', label='Triggers', s=20)

        legend = plt.legend(fontsize=22, loc='upper left')

        legend.legendHandles[0]._sizes = [50]  # Adjust clean sample size
        legend.legendHandles[1]._sizes = [50]

        # Remove x and y ticks
        plt.xticks([])
        plt.yticks([])

        # Save the plot as a PDF
        plt.savefig('./highlighted_embeddings.png')
        plt.savefig('./highlighted_embeddings.pdf', format='pdf', bbox_inches='tight')


    def simi(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        s = torch.mm(z1, z2.t())
        return s

    def con_loss(self, z1, z2, labels):
        # calculate SimCLR loss
        f = lambda x: torch.exp(x / 1.0)

        z2_target = z2[labels == self.args.target_class]
        z2_non_target = z2[labels != self.args.target_class]
        # print('z2_filtered',z2_filtered)

        loss = 0

        intra = f(self.simi(z1, z2_target))        
        inter = f(self.simi(z1, z2_non_target))
        one_to_all_inter=inter.sum(1,keepdim=True)
        one_to_all_inter=one_to_all_inter.repeat(1,len(intra[0]))
        denomitor = one_to_all_inter + intra
        loss = -torch.log(intra/denomitor).mean()

        return loss
    # sim = self.con_loss_1(all_features[idx_outter],all_features[idx_train],self.labels[idx_train],output[idx_train])
    def con_loss_1(self, z1, z2, labels, output):
        f = lambda x: torch.exp(x / 1.0)
        loss = 0

        z2_target = z2[labels == self.args.target_class]
        output = output[:,self.args.target_class]
        output_target = torch.exp(output[labels == self.args.target_class])
        # print('output_target',output_target)
        z2_non_target = z2[labels != self.args.target_class]
        # print('z2_filtered',z2_filtered)

        _, sorted_indices = torch.sort(output_target, descending=True)

        sorted_z2_target = z2_target[sorted_indices]

        # Select top N embeddings
        N = 10  # You can change this as needed
        top_embeddings = sorted_z2_target[:N]

        # Synthesize new embedding
        synthesized_embedding = torch.mean(top_embeddings, dim=0, keepdim=True)
        
        intra = f(self.simi(z1, top_embeddings))              
        inter = f(self.simi(z1, z2_non_target))
        one_to_all_inter=inter.sum(1,keepdim=True)
        one_to_all_inter=one_to_all_inter.repeat(1,len(intra[0]))
        denomitor = one_to_all_inter + intra
        loss += -torch.log(intra/denomitor).mean()

        return loss

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach,idx_unlabeled,test=False):
        
        args = self.args
        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]],device=self.device,dtype=torch.float)
        self.idx_attach = idx_attach
        self.features = features
        self.edge_index = edge_index
        self.edge_weights = edge_weight
        
        # initial a shadow model
        self.shadow_model = GCN(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=labels.max().item() + 1,
                         dropout=0.0, device=self.device).to(self.device)
        
        self.ood_detector = MLP(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=2,
                         dropout=0.0, device=self.device).to(self.device)
        # initalize a trojanNet to generate trigger
        self.trojan = GraphTrojanNet(self.device, features.shape[1], args.trigger_size, layernum=2).to(self.device)
        self.homo_loss = HomoLoss(self.args,self.device)

        

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_detector = optim.Adam(self.ood_detector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # find representative nodes in idx_unlabeled

        features_select = features[torch.cat([idx_train, idx_attach, idx_unlabeled])]
        AE = MLPAE(features_select, features_select, self.device, args.rec_epochs)
        AE.fit()
        rec_score_ori = AE.inference(features_select)
        mean_ori = torch.mean(rec_score_ori)
        std_ori = torch.std(rec_score_ori)

        condition = torch.abs(rec_score_ori - mean_ori) < args.range*std_ori

        # Apply the condition to filter the features
        selected_features = features_select[condition]

        

    
        # change the labels of the poisoned node to the target class
        self.labels = labels.clone()
        self.labels[idx_attach] = args.target_class

        # get the trojan edges, which include the target-trigger edge and the edges among trigger
        trojan_edge = self.get_trojan_edge(len(features),idx_attach,args.trigger_size).to(self.device)

        # update the poisoned graph's edge index
        poison_edge_index = torch.cat([edge_index,trojan_edge],dim=1)


        # furture change it to bilevel optimization
        
        if test==True:
            state_dict = torch.load('model_weights.pth')
            # state_dict = torch.load('../pre_arxiv_3_mul_3/stored_weights/model_weights.pth')
            # state_dict = torch.load('../../UGBA/model_weights.pth')
            self.trojan.load_state_dict(state_dict)
            return 0

        loss_best = 1e8
        for i in range(args.trojan_epochs):
            # self.trojan.train()
            for j in range(self.args.inner):
                # print('inner')
                optimizer_shadow.zero_grad()
                trojan_feat, trojan_weights = self.trojan(features[idx_attach],args.thrd) # may revise the process of generate
                # print('trojan_feat',trojan_feat.size())
                trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                # trojan_weights = torch.cat([trojan_weights,torch.ones([len(trojan_feat),1*self.args.trigger_size],dtype=torch.float,device=self.device)],dim=1)
                # print('trojan_weights',trojan_weights)
                trojan_weights = trojan_weights.flatten()
                # trojan_weights = torch.ones_like((trojan_weights),device=self.device)
                trojan_feat = trojan_feat.view([-1,features.shape[1]])

                poison_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
                poison_x = torch.cat([features,trojan_feat]).detach()

                output, all_features = self.shadow_model(poison_x, poison_edge_index, poison_edge_weights)

                
                
                # original sample with original label in train set, attached sample with target label
                loss_inner = F.nll_loss(output[torch.cat([idx_train,idx_attach])], self.labels[torch.cat([idx_train,idx_attach])]) # add our adaptive loss
                # if j == self.args.inner-1:
                # if i==args.trojan_epochs-1 and j==self.args.inner-1:

                if i>0 and i%100==0 and j==self.args.inner-1:
                    self.visualize_embeddings(poison_x,labels,idx_attach )

                    
                loss_inner.backward()
                optimizer_shadow.step()

                ood_x = torch.cat([selected_features,trojan_feat]).detach()
                for k in range(self.args.k):
                    optimizer_detector.zero_grad()

                    # output_detector, all_features_detector = self.ood_detector(poison_x, poison_edge_index, poison_edge_weights)
                    output_detector = self.ood_detector(ood_x)

                    ood_labels = torch.cat([torch.ones(len(trojan_feat), device=self.device),torch.zeros(len(trojan_feat), device=self.device)])

                    num_to_select = self.args.vs_number * self.args.trigger_size

                    # # Generate random indices for selection
                    random_indices = torch.randperm(len(output_detector) - len(trojan_feat))[:num_to_select]

                    # # Select elements from output_detector excluding the last len(trojan_feat) elements
                    # selected_elements = output_detector[:-len(trojan_feat)][random_indices]

                    concatenated_tensors = torch.cat((output_detector[:-len(trojan_feat)][random_indices], 
                                                      output_detector[-len(trojan_feat):]),
                                                      dim=0)
  
                    loss_detector = F.nll_loss(concatenated_tensors, ood_labels.long())

                    # loss_detector.backward(retain_graph=True)
                    loss_detector.backward()
                    optimizer_detector.step()


            # print('outter')
            acc_train_clean = utils.accuracy(output[idx_train], self.labels[idx_train])
            acc_train_attach = utils.accuracy(output[idx_attach], self.labels[idx_attach])
            optimizer_trigger.zero_grad()

            # rs = np.random.RandomState(self.args.seed)
            rs = np.random.RandomState()
            # select target nodes for triggers
            if self.args.outter_size<=len(idx_unlabeled):
                idx_outter = torch.cat([idx_attach,idx_unlabeled[rs.choice(len(idx_unlabeled),size=4096,replace=False)]])
            else:
                idx_outter = torch.cat([idx_attach,idx_unlabeled])
            

            trojan_feat, trojan_weights = self.trojan(features[idx_outter],self.args.thrd) # may revise the process of generate

            trojan_weights = torch.cat([torch.ones([len(idx_outter),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
            # trojan_weights = torch.cat([trojan_weights,torch.ones([len(trojan_feat),1*self.args.trigger_size],dtype=torch.float,device=self.device)],dim=1)
            trojan_weights = trojan_weights.flatten()
            trojan_feat = trojan_feat.view([-1,features.shape[1]])
            trojan_edge = self.get_trojan_edge(len(features),idx_outter,self.args.trigger_size).to(self.device)
            # edge [idx_train, idx_outter]
            update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
            update_feat = torch.cat([features,trojan_feat])
            update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

            output,all_features = self.shadow_model(update_feat, update_edge_index, update_edge_weights)            
            output_detector= self.ood_detector(update_feat)

            labels_outter = labels.clone()
            labels_outter[idx_outter] = args.target_class

            probabilities = torch.exp(output[idx_outter])
            probabilities_target = probabilities[:,self.args.target_class]
            weights = torch.exp(-probabilities_target)
            weights = weights.detach()
            
            losses = F.nll_loss(output[idx_outter], labels_outter[idx_outter],reduction='none')  # This will be a tensor of the same shape as your batch
            loss_target = losses * (weights+1)
            loss_target = loss_target.mean()
            sim = self.con_loss_1(all_features[idx_outter],all_features[idx_train],self.labels[idx_train],output[idx_train])
            loss_targetclass = sim - self.simi(all_features[idx_outter],all_features[idx_outter]).fill_diagonal_(0).mean()
            loss_dis = F.nll_loss(output_detector[-len(trojan_feat):], torch.ones(len(trojan_feat),device=self.device).long())

            # if(self.args.homo_loss_weight > 0):
            #     loss_homo = self.homo_loss(trojan_edge[:,:int(trojan_edge.shape[1]/2)],\
            #                                 trojan_weights,\
            #                                 update_feat,\
            #                                 self.args.homo_boost_thrd)
                
            loss_outter = 0
            loss_outter += loss_target * self.args.weight_target
            loss_outter += loss_dis * self.args.weight_ood
            loss_outter += loss_targetclass * self.args.weight_targetclass
            # pubmed
            # loss_outter += loss_homo * 10

            loss_outter.backward()
            optimizer_trigger.step()
            acc_train_outter =(output[idx_outter].argmax(dim=1)==args.target_class).float().mean()

            if loss_outter<loss_best:
                self.weights = deepcopy(self.trojan.state_dict())
                loss_best = float(loss_outter)

            if args.debug and i % 50 == 0:
                print('Epoch {}, loss_inner: {:.5f}, loss_target: {:.5f},  loss_dis: {:.5f}, loss_diff: {:.5f}, loss_adv: {:.5f}, loss_targetclass: {:.5f}, ood_score: {:.5f} '\
                        .format(i, loss_inner, loss_target,  loss_dis, loss_dis, loss_target, loss_targetclass,  torch.exp(output_detector[-len(trojan_feat):][:,-1:]).mean()))
                print("acc_train_clean: {:.4f}, ASR_train_attach: {:.4f}, ASR_train_outter: {:.4f}"\
                        .format(acc_train_clean,acc_train_attach,acc_train_outter))
        # if args.debug:
        #     print("load best weight based on the loss outter")
        # self.trojan.load_state_dict(self.weights)
        state_dict = self.trojan.state_dict()
        torch.save(state_dict, 'model_weights.pth')

        self.trojan.eval()

        # torch.cuda.empty_cache()

    def get_poisoned(self):

        with torch.no_grad():
            poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger(self.idx_attach,self.features,self.edge_index,self.edge_weights,self.device)
        poison_labels = self.labels
        # poison_edge_index = poison_edge_index[:,poison_edge_weights>0.0]
        # poison_edge_weights = poison_edge_weights[poison_edge_weights>0.0]
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels

# %%
