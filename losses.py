import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.manifold import Isomap

from utils import power_iteration, binarize, l2_norm

from pytorch_metric_learning import miners, losses
from fabulous.color import fg256


def Linear_discriminant_analysis(batch, labels, dimension, regularizer=None, method="naive"):
    unique_labels = np.unique(labels)
    mean_vectors = []
    for cl in range(np.shape(unique_labels)[0]):
        ids = torch.tensor(np.array([labels==unique_labels[cl]], dtype=np.uint8), dtype=torch.bool).squeeze(0).cuda()
        aaa = batch[ids,:]
        mean_vectors.append(torch.mean(aaa, dim=0))
    mean_vectors = torch.stack(mean_vectors)

    S_W = torch.zeros((dimension,dimension))
    for cl in range(np.shape(unique_labels)[0]):
        ids = torch.tensor(np.array([labels==unique_labels[cl]], dtype=np.uint8), dtype=torch.bool).squeeze(0).cuda()
        class_sc_mat = torch.zeros((dimension,dimension))
        for row in batch[ids,:]:
            row, mv = row.reshape(dimension,1).detach().cpu(), mean_vectors[cl,:].reshape(dimension,1).detach().cpu()
            class_sc_mat += (row-mv) @ torch.t(row-mv)
        S_W += class_sc_mat

    overall_mean = torch.mean(batch, dim=0).reshape(dimension,1).detach().cpu()
    S_B = torch.zeros((dimension,dimension))
    for i, mean_vec in enumerate(mean_vectors):
        mean_vec = mean_vec.reshape(dimension,1).detach().cpu()
        S_B += 4.0 * (mean_vec - overall_mean) @ torch.t(mean_vec - overall_mean)

    if regularizer:
        fisher_criterion = torch.inverse(S_W + torch.tensor(1e-6*torch.eye(S_W.size()[0]) * regularizer)) @ S_B
    else:
        fisher_criterion = torch.inverse(S_W + 1e-3*torch.eye(S_W.size()[0])) @ S_B

    if method == "naive":
        eig = torch.eig(fisher_criterion, eigenvectors=True)
        eigval = eig[0]; eigvec = eig[1]
        eigval, ind = torch.sort(eigval[:,0], 0, descending=True)
        eigvec = eigvec[:,ind[0]].unsqueeze(1)

    elif method == "PI":
        eigval, eigvec = power_iteration(fisher_criterion)
        eigval = torch.tensor([eigval]).type(torch.FloatTensor)
        eigvec = torch.tensor([eigvec]).type(torch.FloatTensor).t()

    elif method == "LOBPCG":
        eigval, eigvec = torch.lobpcg(fisher_criterion)
        eigval, ind = torch.sort(eigval, 0, descending=True)
        eigvec = eigvec[:,ind[0]].unsqueeze(1)
    return eigval, eigvec


def General_Logmap(X):
    dim = X.size()[1]
    log_map = torch.zeros(X.size()[0], dim-1).cuda()
    for i in range(X.size()[0]):
        theta = torch.acos(X[i,dim-1])
        mean = torch.mean(X[i,:])
        for j in range(dim-1):
            log_map[i,j] = torch.tensor([(X[i,j]-mean)*(theta/torch.sin(theta))]).cuda()
    return log_map


class PPGML_Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        super(PPGML_Proxy_Anchor, self).__init__()
        
        # torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.orthogonal_(self.proxies)

        self.keys = range(3)
        values = [i+3 for i in range(1,len(self.keys)+1)]
        self.LDS_dim = dict(zip(self.keys, values))
        print(fg256("cyan", "self.LDS_dim is {}".format(self.LDS_dim)))
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

        self.p = {}
        for i in range(len(self.keys)):
            self.p[i] = torch.cat([torch.zeros(self.LDS_dim[i]-1), torch.ones(1)]).cuda()

        self.offset = 0.50
        
    def forward(self, X, T):
        P = self.proxies
        
        cos = F.linear(l2_norm(X), l2_norm(P))
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot

        X_proj = {}
        for i in range(len(self.keys)):
            X_proj[i] = torch.from_numpy(Isomap(n_components=self.LDS_dim[i]).fit_transform(X.detach().cpu().numpy())).type(torch.cuda.FloatTensor)

        pip_loss = {}
        for i in range(len(self.keys)):
            pip_loss[i] = 0.001 * (X.size(1) - X_proj[i].size(1)) + 2.0 * torch.norm(X_proj[i].t() @ X[:,X.size(1)-(self.LDS_dim[i])], p=2).pow(2)
        min_ind = min(pip_loss, key=pip_loss.get)

        X_proj = F.normalize(X_proj[min_ind], dim=1)
        X_log_map = General_Logmap(X_proj)
        k = self.LDS_dim[min_ind] - 1
        
        sss_eigval, sss_eigvec = Linear_discriminant_analysis(X_log_map, T.detach().cpu().numpy(),
                                                              X_log_map.size(1), regularizer=None,
                                                              method="naive")
        
        mean_log = torch.abs(sss_eigval.mean()) * 1e-1
        center = torch.abs(cos.mean()) + self.offset

        unif = torch.distributions.uniform.Uniform(center-mean_log, center+mean_log)
        unif_ref = torch.zeros(X.size(0), k).cuda()
        for i in range(X.size(0)):
            for j in range(k):
                unif_ref[i,j] = unif.rsample()
        ref = unif_ref @ sss_eigvec.cuda()
#        ref = torch.sum(torch.abs(ref))/cos.size(1)
        ref = torch.mean(torch.abs(ref))

        ref_pos = torch.clamp(torch.sqrt(ref), 0.7, 1.5)
        ref_neg = torch.clamp(torch.sqrt(ref), 0.7, 1.5)

        pos_exp = torch.exp(-48.0 * (cos - 0.10))
        neg_exp = torch.exp(48.0 * (cos + 0.10))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)
        num_valid_proxies = len(with_pos_proxies)

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = (torch.log(1.0 + P_sim_sum*ref_pos).sum()) / num_valid_proxies
        neg_term = (torch.log(1.0 + N_sim_sum*ref_neg).sum()) / self.nb_classes

        loss = pos_term + neg_term - 1e-6 * sss_eigval.mean().cuda()
        return loss


class PPGML_TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(PPGML_TripletLoss, self).__init__()
        self.margin = margin  # 0.1
        self.miner = miners.TripletMarginMiner(0.5, type_of_triplets = 'semihard')
        self.loss_func = losses.TripletMarginLoss(margin = self.margin)

        self.keys = range(3)
        values = [i+3 for i in range(1,len(self.keys)+1)]
        self.LDS_dim = dict(zip(self.keys, values))
        print(fg256("green", "self.LDS_dim is {}".format(self.LDS_dim)))

        self.p = {}
        for i in range(len(self.keys)):
            self.p[i] = torch.cat([torch.zeros(self.LDS_dim[i]-1), torch.ones(1)]).cuda()
        self.offset = 0.50

    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)

        X = embeddings; T = labels

        X_proj = {}; pip_loss = {}
        for i in range(len(self.keys)):
            X_proj[i] = torch.from_numpy(Isomap(n_components=self.LDS_dim[i]).fit_transform(X.detach().cpu().numpy())).type(torch.cuda.FloatTensor)

        for i in range(len(self.keys)):
            pip_loss[i] = 0.001 * (X.size(1) - X_proj[i].size(1)) + 2.0 * torch.norm(X_proj[i].t() @ X[:,X.size(1)-(self.LDS_dim[i])], p=2).pow(2)
        min_ind = min(pip_loss, key=pip_loss.get)

        X_proj = F.normalize(X_proj[min_ind], dim=1)
        X_log_map = General_Logmap(X_proj)
        k = self.LDS_dim[min_ind] - 1
        
        sss_eigval, sss_eigvec = Linear_discriminant_analysis(X_log_map, T.detach().cpu().numpy(),
                                                              X_log_map.size(1), regularizer=None,
                                                              method="naive")
        
        mean_log = torch.abs(sss_eigval.mean()) * 1e-1
        center = torch.abs(X.mean()) + self.offset

        unif = torch.distributions.uniform.Uniform(center-mean_log, center+mean_log)
        unif_ref = torch.zeros(X.size(0), k).cuda()
        for i in range(X.size(0)):
            for j in range(k):
                unif_ref[i,j] = unif.rsample()
        ref = unif_ref @ sss_eigvec.cuda()
#        ref = torch.sum(torch.abs(ref))/cos.size(1)
        ref = torch.mean(torch.abs(ref))

        ref_pos = torch.clamp(torch.sqrt(ref), 0.7, 1.5)
        ref_neg = torch.clamp(torch.sqrt(ref), 0.7, 1.5)

        loss = self.loss_func(embeddings, labels, hard_pairs, ref_pos, ref_neg) - 1e-6 * sss_eigval.mean().cuda()
        return loss


class PPGML_ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(PPGML_ContrastiveLoss, self).__init__()
        self.margin = margin
        self.miner = miners.PairMarginMiner(0.75, 0.5, use_similarity=0)
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin) 

        self.keys = range(3)
        values = [i+3 for i in range(1,len(self.keys)+1)]
        self.LDS_dim = dict(zip(self.keys, values))
        print(fg256("green", "[INFO] self.LDS_dim is {}".format(self.LDS_dim)))

        self.p = {}
        for i in range(len(self.keys)):
            self.p[i] = torch.cat([torch.zeros(self.LDS_dim[i]-1), torch.ones(1)]).cuda()
        self.offset = 0.50
        
    def forward(self, embeddings, labels):
        cont_pairs = self.miner(embeddings, labels)

        X = embeddings; T = labels

        X_proj = {}; pip_loss = {}
        for i in range(len(self.keys)):
            X_proj[i] = torch.from_numpy(Isomap(n_components=self.LDS_dim[i]).fit_transform(X.detach().cpu().numpy())).type(torch.cuda.FloatTensor)

        for i in range(len(self.keys)):
            pip_loss[i] = 0.001 * (X.size(1) - X_proj[i].size(1)) + 2.0 * torch.norm(X_proj[i].t() @ X[:,X.size(1)-(self.LDS_dim[i])], p=2).pow(2)
        min_ind = min(pip_loss, key=pip_loss.get)

        X_proj = F.normalize(X_proj[min_ind], dim=1)
        X_log_map = General_Logmap(X_proj)
        k = self.LDS_dim[min_ind] - 1
        
        sss_eigval, sss_eigvec = Linear_discriminant_analysis(X_log_map, T.detach().cpu().numpy(),
                                                              X_log_map.size(1), regularizer=None,
                                                              method="naive")
        
        mean_log = torch.abs(sss_eigval.mean()) * 1e-1
        center = torch.abs(X.mean()) + self.offset

        unif = torch.distributions.uniform.Uniform(center-mean_log, center+mean_log)
        unif_ref = torch.zeros(X.size(0), k).cuda()
        for i in range(X.size(0)):
            for j in range(k):
                unif_ref[i,j] = unif.rsample()
        ref = unif_ref @ sss_eigvec.cuda()
#        ref = torch.sum(torch.abs(ref))/cos.size(1)
        ref = torch.mean(torch.abs(ref))

        ref_pos = torch.clamp(torch.sqrt(ref), 0.7, 1.5)
        ref_neg = torch.clamp(torch.sqrt(ref), 0.7, 1.5)

        loss = self.loss_func(embeddings, labels, cont_pairs, ref_pos, ref_neg) - 1e-6 * sss_eigval.mean().cuda()
        return loss


class PPGML_MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(PPGML_MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50

        self.LDS_dim_1 = 3
        self.LDS_dim_2 = 4
        self.LDS_dim_3 = 5

        self.p0 = torch.cat([torch.zeros(self.LDS_dim_1-1), torch.ones(1)]).cuda()
        self.p1 = torch.cat([torch.zeros(self.LDS_dim_2-1), torch.ones(1)]).cuda()
        self.p2 = torch.cat([torch.zeros(self.LDS_dim_3-1), torch.ones(1)]).cuda()
        self.offset = 0.50

        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        
    def forward(self, embeddings, labels):

        X = embeddings
        T = labels

        X_proj_1 = torch.from_numpy(Isomap(n_components=self.LDS_dim_1).fit_transform(X.detach().cpu().numpy())).type(torch.cuda.FloatTensor)
        X_proj_2 = torch.from_numpy(Isomap(n_components=self.LDS_dim_2).fit_transform(X.detach().cpu().numpy())).type(torch.cuda.FloatTensor)
        X_proj_3 = torch.from_numpy(Isomap(n_components=self.LDS_dim_3).fit_transform(X.detach().cpu().numpy())).type(torch.cuda.FloatTensor)

        pip_loss_1 = 0.001 * (X.size(1) - X_proj_1.size(1)) + 2.0 * torch.norm(X_proj_1.t() @ X[:,X.size(1)-(self.LDS_dim_1-1)], p=2).pow(2)
        pip_loss_2 = 0.001 * (X.size(1) - X_proj_2.size(1)) + 2.0 * torch.norm(X_proj_2.t() @ X[:,X.size(1)-(self.LDS_dim_2-1)], p=2).pow(2)
        pip_loss_3 = 0.001 * (X.size(1) - X_proj_3.size(1)) + 2.0 * torch.norm(X_proj_3.t() @ X[:,X.size(1)-(self.LDS_dim_3-1)], p=2).pow(2)
        min_ind = torch.argmin(torch.FloatTensor([pip_loss_1, pip_loss_2, pip_loss_3]))
        
        if min_ind == 0:
            X_proj = F.normalize(X_proj_1, dim=1)  # L2-norm (unit length)
            X_log_map = General_Logmap(X_proj)
            k = self.LDS_dim_1 - 1
        elif min_ind == 1:
            X_proj = F.normalize(X_proj_2, dim=1)  # L2-norm (unit length)
            X_log_map = General_Logmap(X_proj)
            k = self.LDS_dim_2 - 1
        else:
            X_proj = F.normalize(X_proj_3, dim=1)  # L2-norm (unit length)
            X_log_map = General_Logmap(X_proj)
            k = self.LDS_dim_3 - 1

        sss_eigval, sss_eigvec = Linear_discriminant_analysis(X_log_map, T.detach().cpu().numpy(),
                                                              X_log_map.size(1), regularizer=None,
                                                              method="naive")
    
        mean_log = torch.abs(sss_eigval.mean()) * 1e-1
        center = torch.abs(embeddings.mean()) + self.offset

        # Uniform distribution
        unif = torch.distributions.uniform.Uniform(center-mean_log, center+mean_log)
        unif_ref = torch.zeros(X.size(0), k).cuda()
        for i in range(X.size(0)):
            for j in range(k):
                unif_ref[i,j] = unif.rsample()
        ref = unif_ref @ sss_eigvec.cuda()
        ref = torch.mean(torch.abs(ref))

        ref_pos = torch.clamp(torch.sqrt(ref), 0.7, 1.5)
        ref_neg = torch.clamp(torch.sqrt(ref), 0.7, 1.5)

        #########################

        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs, ref_pos, ref_neg) - 1e-6 * sss_eigval.mean().cuda()
        return loss


### BASIC LOSSES
class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        super(Proxy_Anchor, self).__init__()
        # torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies
        
        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes = self.nb_classes, embedding_size = self.sz_embed, softmax_scale = self.scale).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50
        
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin) 
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    

class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'semihard')
        self.loss_func = losses.TripletMarginLoss(margin = self.margin)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    

class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(l2_reg_weight=self.l2_reg, normalize_embeddings = False)
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss