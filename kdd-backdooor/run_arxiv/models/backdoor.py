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
        feat = F.sigmoid(feat)
        edge_weight = self.edge(h)
        # feat = GW(feat, thrd, self.device)
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
        return edge_index
    

    def get_trojan_edge(self, start, idx_attach, trigger_size):
        edge_list = []
        for idx in idx_attach:
            # Copy the subgraph structure and update indices
            edges = self.trigger_index.clone()
            edges = edges[:,1:]
            edges+=start
            edge_list.append(edges)
            for i in range(trigger_size):
                edge_list.append(torch.tensor([[idx], [start + i]], device=self.device))
            start += trigger_size
        edge_index = torch.cat(edge_list, dim=1)
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1],edge_index[0]])
        edge_index = torch.stack([row,col])

        return edge_index

    def inject_trigger(self, idx_attach, features,edge_index,edge_weight, device):
        self.trojan = self.trojan.to(device)
        idx_attach = idx_attach.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        self.trojan.eval()

        trojan_feat, trojan_weights = self.trojan(features[idx_attach],self.args.thrd) # may revise the process of generate
        
        trojan_weights = torch.cat([trojan_weights,torch.ones([len(trojan_feat),1*self.args.trigger_size],dtype=torch.float,device=self.device)],dim=1)
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

    def simi(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        s = torch.mm(z1, z2.t())
        return s

    def con_loss(self, z1, z2, labels, output):
        f = lambda x: torch.exp(x / 1.0)
        loss = 0
        z2_target = z2[labels == self.args.target_class]
        output = output[:,self.args.target_class]
        output_target = torch.exp(output[labels == self.args.target_class])
        z2_non_target = z2[labels != self.args.target_class]
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
        features_select = features[torch.cat([idx_train, idx_attach, idx_unlabeled])]
        AE = MLPAE(features_select, features_select, self.device, args.rec_epochs)
        AE.fit()
        rec_score_ori = AE.inference(features_select)
        mean_ori = torch.mean(rec_score_ori)
        std_ori = torch.std(rec_score_ori)
        condition = torch.abs(rec_score_ori - mean_ori) < args.range*std_ori
        selected_features = features_select[condition]
        self.labels = labels.clone()
        self.labels[idx_attach] = args.target_class
        trojan_edge = self.get_trojan_edge(len(features),idx_attach,args.trigger_size).to(self.device)
        poison_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

        import time

# Record the starting time
        start_time = time.time()
        for i in range(args.trojan_epochs):
            for j in range(self.args.inner):
                optimizer_shadow.zero_grad()
                trojan_feat, trojan_weights = self.trojan(features[idx_attach],args.thrd) # may revise the process of generate
                trojan_weights = torch.cat([trojan_weights,torch.ones([len(trojan_feat),1*self.args.trigger_size],dtype=torch.float,device=self.device)],dim=1)
                trojan_weights = trojan_weights.flatten()
                trojan_feat = trojan_feat.view([-1,features.shape[1]])
                poison_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
                poison_x = torch.cat([features,trojan_feat]).detach()
                output, all_features = self.shadow_model(poison_x, poison_edge_index, poison_edge_weights)
                # original sample with original label in train set, attached sample with target label
                loss_inner = F.nll_loss(output[torch.cat([idx_train,idx_attach])], self.labels[torch.cat([idx_train,idx_attach])]) # add our adaptive loss
                loss_inner.backward()
                optimizer_shadow.step()
                ood_x = torch.cat([selected_features,trojan_feat]).detach()
                for k in range(self.args.k):
                    optimizer_detector.zero_grad()
                    output_detector = self.ood_detector(ood_x)
                    ood_labels = torch.cat([torch.ones(len(trojan_feat), device=self.device),torch.zeros(len(trojan_feat), device=self.device)])
                    num_to_select = self.args.vs_number * self.args.trigger_size
                    random_indices = torch.randperm(len(output_detector) - len(trojan_feat))[:num_to_select]
                    concatenated_tensors = torch.cat((output_detector[:-len(trojan_feat)][random_indices], 
                                                      output_detector[-len(trojan_feat):]),
                                                      dim=0)
                    loss_detector = F.nll_loss(concatenated_tensors, ood_labels.long())
                    loss_detector.backward()
                    optimizer_detector.step()

            optimizer_trigger.zero_grad()
            rs = np.random.RandomState()
            if self.args.outter_size<=len(idx_unlabeled):
                idx_outter = torch.cat([idx_attach,idx_unlabeled[rs.choice(len(idx_unlabeled),size=4096,replace=False)]])
            else:
                idx_outter = torch.cat([idx_attach,idx_unlabeled])

            trojan_feat, trojan_weights = self.trojan(features[idx_outter],self.args.thrd) # may revise the process of generate
            trojan_weights = torch.cat([trojan_weights,torch.ones([len(trojan_feat),1*self.args.trigger_size],dtype=torch.float,device=self.device)],dim=1)
            trojan_weights = trojan_weights.flatten()
            trojan_feat = trojan_feat.view([-1,features.shape[1]])
            trojan_edge = self.get_trojan_edge(len(features),idx_outter,self.args.trigger_size).to(self.device)
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
            sim = self.con_loss(all_features[idx_outter],all_features[idx_train],self.labels[idx_train],output[idx_train])
            loss_targetclass = sim - self.simi(all_features[idx_outter],all_features[idx_train]).fill_diagonal_(0).mean()
            loss_dis = F.nll_loss(output_detector[-len(trojan_feat):], torch.ones(len(trojan_feat),device=self.device).long())
            loss_homo = 0.0
            loss_outter = 0
            loss_outter += loss_target * self.args.weight_target
            loss_outter += loss_dis * self.args.weight_ood
            loss_outter += loss_targetclass * self.args.weight_targetclass
            loss_outter.backward()
            optimizer_trigger.step()
            if args.debug and i % 20 == 0:
                print('Epoch {}, loss_inner: {:.5f}, loss_target: {:.5f}, homo loss: {:.5f}, loss_dis: {:.5f}, loss_diff: {:.5f}, loss_adv: {:.5f}, loss_targetclass: {:.5f}, ood_score: {:.5f} '\
                        .format(i, loss_inner, loss_target, loss_homo, loss_dis, loss_dis, loss_target, loss_target,  torch.exp(output_detector[-len(trojan_feat):][:,-1:]).mean()))
        end_time = time.time()
        state_dict = self.trojan.state_dict()
        torch.save(state_dict, 'model_weights.pth')
        elapsed_time = end_time - start_time
        print("Time taken:", elapsed_time, "seconds")
        self.trojan.eval()

        # torch.cuda.empty_cache()

    def get_poisoned(self):

        with torch.no_grad():
            poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger(self.idx_attach,self.features,self.edge_index,self.edge_weights,self.device)
        poison_labels = self.labels
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels

# %%
