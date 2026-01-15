import datetime
import gc
import os
import pickle
import random
import time

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dataloader
from numpy import random

import DataHandler
import scipy.sparse as sp
from BGNN import myModel as BGNNModel
from MV_Net import MetaWeightNet
import graph_utils
from CHARM import CHIME, CHIMELoss
from Params import args
from Utils.TimeLogger import log

t.backends.cudnn.benchmark=True

if t.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

MAX_FLAG = 0x7FFFFFFF

now_time = datetime.datetime.now()
modelTime = datetime.datetime.strftime(now_time,'%Y_%m_%d__%H_%M_%S')

t.autograd.set_detect_anomaly(True)

class Model():
    def __init__(self):
   

        self.trn_file = args.path + args.dataset + '/trn_'
        self.tst_file = args.path + args.dataset + '/tst_int'     
        # self.tst_file = args.path + args.dataset + '/BST_tst_int_59' 
        # Tmall: 3,4,5,6,8,59
        # IJCAI_15: 5,6,8,10,13,53

        self.t_max = -1 
        self.t_min = 0x7FFFFFFF
        self.time_number = -1
 
        self.user_num = -1
        self.item_num = -1
        self.behaviors = []
        self.behaviors_data = {}
     
        #history
        self.train_loss = []
        self.his_hr = []
        self.his_ndcg = []
        gc.collect()  #

        self.relu = t.nn.ReLU()
        self.sigmoid = t.nn.Sigmoid()
        self.curEpoch = 0
        self.device = t.device('cuda' if use_cuda else 'cpu')
        self.enable_meta_learning = getattr(args, 'enable_meta_learning', False)
        self.bgnn_behavior_mats = None
        self.bgnn = None
        self.meta_weight_net = None
        self.criterion = None

            
        if args.dataset == 'Tmall':
            self.behaviors_SSL = ['pv','fav', 'cart', 'buy']
            self.behaviors = ['pv','fav', 'cart', 'buy']
            # self.behaviors = ['buy']

        elif args.dataset == 'IJCAI_15':
            self.behaviors = ['click','fav', 'cart', 'buy']
            # self.behaviors = ['buy']
            self.behaviors_SSL = ['click','fav', 'cart', 'buy']

        elif args.dataset == 'taobao':
            self.behaviors = ['view','cart', 'buy']
            # self.behaviors = ['buy']
            self.behaviors_SSL = ['view','cart', 'buy']

        elif args.dataset == 'beibei':
            self.behaviors = ['view','cart', 'buy']
            # self.behaviors = ['buy']
            self.behaviors_SSL = ['view','cart', 'buy']


        for i in range(0, len(self.behaviors)):
            with open(self.trn_file + self.behaviors[i], 'rb') as fs:  
                data = pickle.load(fs)
                self.behaviors_data[i] = data 

                if data.get_shape()[0] > self.user_num:  
                    self.user_num = data.get_shape()[0]  
                if data.get_shape()[1] > self.item_num:  
                    self.item_num = data.get_shape()[1]

             
                if data.data.max() > self.t_max:
                    self.t_max = data.data.max()
                if data.data.min() < self.t_min:
                    self.t_min = data.data.min()

        
                if self.behaviors[i]==args.target:
                    self.trainMat = data
                    self.trainLabel = 1*(self.trainMat != 0)  
                    self.labelP = np.squeeze(np.array(np.sum(self.trainLabel, axis=0)))  

        if self.enable_meta_learning:
            self.bgnn_behavior_mats = self._build_bgnn_behavior_mats()

        time = datetime.datetime.now()
        print("Start building LightGCN graphs:  ", time)
        self.behavior_graphs = self._build_behavior_graphs()
        time = datetime.datetime.now()
        print("End building:", time)

        self.target_index = self.behaviors.index(args.target)
        self.target_behavior = self.behaviors[self.target_index]
        raw_hesitation = getattr(args, 'hesitation_behavior', 'all')
        selected_behaviors = []
        if isinstance(raw_hesitation, str) and raw_hesitation.lower() != 'all':
            candidate_names = [name.strip() for name in raw_hesitation.split(',')]
            for beh in candidate_names:
                if beh and beh in self.behaviors and beh != self.target_behavior:
                    selected_behaviors.append(beh)
        if not selected_behaviors:
            selected_behaviors = [b for b in self.behaviors if b != self.target_behavior]
        self.hesitation_behavior_indices = [self.behaviors.index(b) for b in selected_behaviors]
        self.hesitation_behavior_names = selected_behaviors

        print("user_num: ", self.user_num)
        print("item_num: ", self.item_num)
        print("\n")
        print("hesitation behaviors: ", self.hesitation_behavior_names if self.hesitation_behavior_names else "None")


        #---------------------------------------------------------------------------------------------->>>>>
        #train_data
        train_u, train_v = self.trainMat.nonzero()
        train_data = np.hstack((train_u.reshape(-1,1), train_v.reshape(-1,1))).tolist()
        train_dataset = DataHandler.RecDataset_beh(self.behaviors, train_data, self.item_num, self.behaviors_data, True)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

        #valid_data


        # test_data  
        with open(self.tst_file, 'rb') as fs:
            data = pickle.load(fs)

        test_user = np.array([idx for idx, i in enumerate(data) if i is not None])
        test_item = np.array([i for idx, i in enumerate(data) if i is not None])
        # tstUsrs = np.reshape(np.argwhere(data!=None), [-1])
        test_data = np.hstack((test_user.reshape(-1,1), test_item.reshape(-1,1))).tolist()
        test_data = [i for i in test_data if i[0] in train_u and i[1] in train_v]
        # testbatch = np.maximum(1, args.batch * args.sampNum 
        test_dataset = DataHandler.RecDataset(test_data, self.item_num, self.trainMat, 0, False)
        self.test_loader = dataloader.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
        # -------------------------------------------------------------------------------------------------->>>>>

    def _build_bgnn_behavior_mats(self):
        mats = []
        for idx in range(len(self.behaviors)):
            mats.append(graph_utils.get_use(self.behaviors_data[idx], device=self.device))
        return mats

    def _build_behavior_graphs(self):
        graphs = {}
        for idx, behavior in enumerate(self.behaviors):
            graphs[behavior] = self._convert_to_lightgcn_graph(self.behaviors_data[idx])
        return graphs

    def _convert_to_lightgcn_graph(self, behavior_matrix):
        if not sp.isspmatrix_coo(behavior_matrix):
            behavior_matrix = behavior_matrix.tocoo()
        num_nodes = self.user_num + self.item_num
        rows = np.concatenate([behavior_matrix.row, behavior_matrix.col + self.user_num])
        cols = np.concatenate([behavior_matrix.col + self.user_num, behavior_matrix.row])
        data = np.ones(rows.shape[0], dtype=np.float32)
        adj = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum + 1e-8, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
        indices = np.vstack((norm_adj.row, norm_adj.col))
        indices_tensor = t.from_numpy(indices).long()
        values_tensor = t.from_numpy(norm_adj.data.astype(np.float32))
        return t.sparse.FloatTensor(indices_tensor, values_tensor, t.Size(norm_adj.shape))

    def prepareModel(self):
        self.modelName = self.getModelName()
        self.model = CHIME(
            num_users=self.user_num,
            num_items=self.item_num,
            embedding_dim=args.hidden_dim,
            behavior_graphs=self.behavior_graphs,
            target_behavior=self.target_behavior,
            n_layers=args.hesitation_layers,
            hesitation_temperature=args.hesitation_temperature,
            injection_alpha=args.injection_alpha,
            substitution_threshold=args.substitute_cosine_threshold,
            substitution_penalty=args.substitute_penalty,
        ).to(self.device)
        self.criterion = CHIMELoss(
            info_nce_temperature=args.mine_tau,
            general_bpr_weight=args.general_bpr_weight,
            hesitation_bpr_weight=args.hesitation_bpr_weight,
            hesitation_weight=args.hesitation_weight,
            substitution_margin=args.substitute_neg_margin,
            substitution_weight=args.substitute_loss_weight,
        )
        optim_params = [{
            'params': self.model.parameters(),
            'lr': args.lr,
            'weight_decay': args.opt_weight_decay,
        }]
        if self.enable_meta_learning:
            if self.bgnn_behavior_mats is None:
                self.bgnn_behavior_mats = self._build_bgnn_behavior_mats()
            self.bgnn = BGNNModel(self.user_num, self.item_num, self.behaviors, self.bgnn_behavior_mats).to(self.device)
            self.meta_weight_net = MetaWeightNet(len(self.behaviors)).to(self.device)
            optim_params.append({
                'params': self.bgnn.parameters(),
                'lr': args.lr,
                'weight_decay': args.opt_weight_decay,
            })
            optim_params.append({
                'params': self.meta_weight_net.parameters(),
                'lr': args.meta_lr,
                'weight_decay': args.meta_opt_weight_decay,
            })
        self.opt = t.optim.AdamW(optim_params)
        if args.isload:
            self.loadModel(args.loadModelPath)

    def innerProduct(self, u, i, j):  
        pred_i = t.sum(t.mul(u,i), dim=1)*args.inner_product_mult  
        pred_j = t.sum(t.mul(u,j), dim=1)*args.inner_product_mult
        return pred_i, pred_j

    def SSL(self, user_embeddings, item_embeddings, target_user_embeddings, target_item_embeddings, user_step_index):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[t.randperm(embedding.size()[0])]  
            return corrupted_embedding
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[t.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:,t.randperm(corrupted_embedding.size()[1])]  
            return corrupted_embedding
        def score(x1, x2):
            return t.sum(t.mul(x1, x2), 1)

        def neg_sample_pair(x1, x2, τ = 0.05):  
            for i in range(x1.shape[0]):
                index_set = set(np.arange(x1.shape[0]))
                index_set.remove(i)
                index_set_neg = t.as_tensor(np.array(list(index_set))).long().to(self.device)  

                x_pos = x1[i].repeat(x1.shape[0]-1, 1)
                x_neg = x2[index_set]  
                
                if i==0:
                    x_pos_all = x_pos
                    x_neg_all = x_neg
                else:
                    x_pos_all = t.cat((x_pos_all, x_pos), 0)
                    x_neg_all = t.cat((x_neg_all, x_neg), 0)
            x_pos_all = t.as_tensor(x_pos_all)  
            x_neg_all = t.as_tensor(x_neg_all)   

            return x_pos_all, x_neg_all

        def one_neg_sample_pair_index(i, step_index, embedding1, embedding2):

            index_set = set(np.array(step_index))
            index_set.remove(i.item())
            neg2_index = t.as_tensor(np.array(list(index_set))).long().to(self.device)

            neg1_index = t.ones((2,), dtype=t.long)
            neg1_index = neg1_index.new_full((len(index_set),), i)

            neg_score_pre = t.sum(compute(embedding1, embedding2, neg1_index, neg2_index).squeeze())
            return neg_score_pre

        def multi_neg_sample_pair_index(batch_index, step_index, embedding1, embedding2):  

            index_set = set(np.array(step_index.cpu()))
            batch_index_set = set(np.array(batch_index.cpu()))
            neg2_index_set = index_set - batch_index_set                         
            neg2_index = t.as_tensor(np.array(list(neg2_index_set))).long().to(self.device) 
            neg2_index = t.unsqueeze(neg2_index, 0)                             
            neg2_index = neg2_index.repeat(len(batch_index), 1)                  
            neg2_index = t.reshape(neg2_index, (1, -1))                          
            neg2_index = t.squeeze(neg2_index)                                   
                                                                                 
            neg1_index = batch_index.long().to(self.device)     
            neg1_index = t.unsqueeze(neg1_index, 1)                              
            neg1_index = neg1_index.repeat(1, len(neg2_index_set))               
            neg1_index = t.reshape(neg1_index, (1, -1))                                   
            neg1_index = t.squeeze(neg1_index)                                   

            neg_score_pre = t.sum(compute(embedding1, embedding2, neg1_index, neg2_index).squeeze().view(len(batch_index), -1), -1)  
            return neg_score_pre  

        def compute(x1, x2, neg1_index=None, neg2_index=None, τ = 0.05):  

            if neg1_index!=None:
                x1 = x1[neg1_index]
                x2 = x2[neg2_index]

            N = x1.shape[0]  
            D = x1.shape[1]

            x1 = x1
            x2 = x2

            scores = t.exp(t.div(t.bmm(x1.view(N, 1, D), x2.view(N, D, 1)).view(N, 1), np.power(D, 1)+1e-8))  
            
            return scores
        def single_infoNCE_loss_simple(embedding1, embedding2):
            pos = score(embedding1, embedding2)  
            neg1 = score(embedding2, row_column_shuffle(embedding1))  
            one = t.cuda.FloatTensor(neg1.shape[0]).fill_(1)  
            # one = zeros = t.ones(neg1.shape[0])
            con_loss = t.sum(-t.log(1e-8 + t.sigmoid(pos))-t.log(1e-8 + (one - t.sigmoid(neg1))))  
            return con_loss
   
        def single_infoNCE_loss(embedding1, embedding2):
            N = embedding1.shape[0]
            D = embedding1.shape[1]

            pos_score = compute(embedding1, embedding2).squeeze()  

            neg_x1, neg_x2 = neg_sample_pair(embedding1, embedding2)  
            neg_score = t.sum(compute(neg_x1, neg_x2).view(N, (N-1)), dim=1)    
            con_loss = -t.log(1e-8 +t.div(pos_score, neg_score))   
            con_loss = t.mean(con_loss)  
            return max(0, con_loss)

        def single_infoNCE_loss_one_by_one(embedding1, embedding2, step_index):  
            N = step_index.shape[0]
            D = embedding1.shape[1]

            pos_score = compute(embedding1[step_index], embedding2[step_index]).squeeze()  
            neg_score = t.zeros((N,), dtype = t.float64, device=self.device)  

            #-------------------------------------------------multi version-----------------------------------------------------
            steps = int(np.ceil(N / args.SSL_batch))  
            for i in range(steps):
                st = i * args.SSL_batch
                ed = min((i+1) * args.SSL_batch, N)
                batch_index = step_index[st: ed]

                neg_score_pre = multi_neg_sample_pair_index(batch_index, step_index, embedding1, embedding2)
                if i ==0:
                    neg_score = neg_score_pre
                else:
                    neg_score = t.cat((neg_score, neg_score_pre), 0)
            #-------------------------------------------------multi version-----------------------------------------------------

            con_loss = -t.log(1e-8 +t.div(pos_score, neg_score+1e-8))  


            assert not t.any(t.isnan(con_loss))
            assert not t.any(t.isinf(con_loss))

            return t.where(t.isnan(con_loss), t.full_like(con_loss, 0+1e-8), con_loss)

        if user_step_index is None:
            user_step_index = t.arange(self.user_num, device=self.device)
        elif not t.is_tensor(user_step_index):
            user_step_index = t.as_tensor(user_step_index, dtype=t.long, device=self.device)
        else:
            user_step_index = user_step_index.long().to(self.device)
        if user_step_index.numel() == 0:
            empty_losses = [t.zeros(0, device=self.device) for _ in range(len(self.behaviors_SSL))]
            return empty_losses, user_step_index

        user_con_loss_list = []
        item_con_loss_list = []

        SSL_len = max(1, int(user_step_index.shape[0]/10))
        SSL_len = min(SSL_len, user_step_index.shape[0])
        sampled_users = np.random.choice(user_step_index.cpu().numpy(), size=SSL_len, replace=False)
        user_step_index = t.as_tensor(sampled_users, dtype=t.long, device=self.device)

        for i in range(len(self.behaviors_SSL)):

            user_con_loss_list.append(single_infoNCE_loss_one_by_one(user_embeddings[-1], user_embeddings[i], user_step_index))

        user_con_losss = t.stack(user_con_loss_list, dim=0)  

        return user_con_loss_list, user_step_index  

    def run(self):

        self.prepareModel()
        if args.isload == True:
            print("----------------------pre test:")
            HR, NDCG = self.testEpoch(self.test_loader)
            print(f"HR: {HR} , NDCG: {NDCG}")
        log('Model Prepared')


        cvWait = 0
        self.best_HR = 0
        self.best_NDCG = 0
        flag = 0

        print("Test before train:")
        HR, NDCG = self.testEpoch(self.test_loader)

        for e in range(self.curEpoch, args.epoch+1):
            self.curEpoch = e

            log("*****************Start epoch: %d ************************"%e)

            if args.isJustTest == False:
                epoch_loss = self.trainEpoch()
                self.train_loss.append(epoch_loss)
                print(f"epoch {e/args.epoch},  epoch loss {epoch_loss}")
                self.train_loss.append(epoch_loss)
            else:
                break

            HR, NDCG = self.testEpoch(self.test_loader)
            self.his_hr.append(HR)
            self.his_ndcg.append(NDCG)

            if HR > self.best_HR:
                self.best_HR = HR
                self.best_epoch = self.curEpoch
                cvWait = 0
                print("--------------------------------------------------------------------------------------------------------------------------best_HR", self.best_HR)
                self.saveHistory()
                self.saveModel()



            if NDCG > self.best_NDCG:
                self.best_NDCG = NDCG
                self.best_epoch = self.curEpoch
                cvWait = 0
                print("--------------------------------------------------------------------------------------------------------------------------best_NDCG", self.best_NDCG)
                self.saveHistory()
                self.saveModel()



            if (HR<self.best_HR) and (NDCG<self.best_NDCG):
                cvWait += 1


            if cvWait == args.patience:
                print(f"Early stop at {self.best_epoch} :  best HR: {self.best_HR}, best_NDCG: {self.best_NDCG} \n")
                self.saveHistory()
                self.saveModel()
                break

        HR, NDCG = self.testEpoch(self.test_loader)
        self.his_hr.append(HR)
        self.his_ndcg.append(NDCG)

    def negSamp(self, temLabel, sampSize, nodeNum):
        negset = [None] * sampSize
        cur = 0
        while cur < sampSize:
            rdmItm = np.random.choice(nodeNum)
            if temLabel[rdmItm] == 0:
                negset[cur] = rdmItm
                cur += 1
        return negset

    def sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds.cpu()].toarray()
        batch = len(batIds)
        user_id = [] 
        item_id_pos = [] 
        item_id_neg = [] 
 
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
            sampNum = min(args.sampNum, len(posset))   
            if sampNum == 0:
                poslocs = [np.random.choice(labelMat.shape[1])]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = self.negSamp(temLabel[i], sampNum, labelMat.shape[1])

            for j in range(sampNum):
                user_id.append(batIds[i].item())
                item_id_pos.append(poslocs[j].item()) 
                item_id_neg.append(neglocs[j])
                cur += 1

        return t.as_tensor(np.array(user_id), dtype=t.long).to(self.device), t.as_tensor(np.array(item_id_pos), dtype=t.long).to(self.device), t.as_tensor(np.array(item_id_neg),dtype=t.long).to(self.device)

    def this_batch_pairwise(self, training_user, training_item):
        from random import choice  
        n_negs = 1  

        users = training_user
        items = training_item

        u_idx, i_idx, j_idx = [], [], [] 
        item_list = list(self.training_set_i.keys())  

        for i, user_id in enumerate(users):
            i_idx.append(self.node_table_dict_by_node_id[items[i]]['node_feature'])  
            u_idx.append(self.node_table_dict_by_node_id[user_id]['node_feature'])  

            
            for _ in range(n_negs):
                neg_item_id = choice(item_list)  
                while neg_item_id in self.training_set_u[user_id]:
                    neg_item_id = choice(item_list)  
                j_idx.append(self.node_table_dict_by_node_id[neg_item_id]['node_feature'])  

        
        return u_idx, i_idx, j_idx  

    def trainEpoch(self):
        self.model.train()
        if self.bgnn is not None:
            self.bgnn.train()
        if self.meta_weight_net is not None:
            self.meta_weight_net.train()
        train_loader = self.train_loader
        time = datetime.datetime.now()
        print("start_ng_samp:  ", time)
        train_loader.dataset.ng_sample()
        time = datetime.datetime.now()
        print("end_ng_samp:  ", time)

        epoch_loss = 0.0
        step = 0

        num_behaviors = len(self.behaviors)
        for user, target_item, item_i, item_j in train_loader:
            user = user.long().to(self.device)
            target_item = target_item.long().to(self.device)
            pos_array = np.array(item_i)
            neg_array = np.array(item_j)
            if pos_array.ndim == 2 and pos_array.shape[0] == num_behaviors:
                pos_array = pos_array.T
            if neg_array.ndim == 2 and neg_array.shape[0] == num_behaviors:
                neg_array = neg_array.T
            pos_tensor = t.as_tensor(pos_array, dtype=t.long, device=self.device)
            neg_tensor = t.as_tensor(neg_array, dtype=t.long, device=self.device)

            self.opt.zero_grad(set_to_none=True)
            target_negatives = neg_tensor[:, self.target_index]
            hesitation_tensor = self._collect_hesitation_candidates(pos_tensor)
            purchased_tensor = pos_tensor[:, self.target_index].unsqueeze(1)
            outputs = self.model(
                user,
                target_item,
                target_negatives,
                hesitation_item_indices=hesitation_tensor,
                purchased_item_indices=purchased_tensor,
            )
            loss_components = self.criterion(outputs)
            loss = loss_components["total"]

            if self.enable_meta_learning:
                bgnn_outputs = self.bgnn()
                ssl_loss, beh_loss = self._compute_multi_behavior_objective(user, pos_tensor, neg_tensor, bgnn_outputs)
                loss = loss + args.meta_ssl_weight * ssl_loss + args.meta_behavior_weight * beh_loss

            loss.backward()
            params_to_clip = list(self.model.parameters())
            if self.enable_meta_learning:
                params_to_clip += list(self.bgnn.parameters()) + list(self.meta_weight_net.parameters())
            nn.utils.clip_grad_norm_(params_to_clip, max_norm=20, norm_type=2)
            self.opt.step()

            epoch_loss += loss.item()
            step += 1

        return epoch_loss / max(step, 1)

    def _collect_hesitation_candidates(self, pos_tensor):
        allowed_indices = getattr(self, 'hesitation_behavior_indices', None)
        if not allowed_indices:
            return None
        valid_indices = [idx for idx in allowed_indices if 0 <= idx < pos_tensor.shape[1] and idx != self.target_index]
        if not valid_indices:
            return None
        return pos_tensor[:, valid_indices]

    def _compute_multi_behavior_objective(self, user, pos_tensor, neg_tensor, bgnn_outputs):
        if (not self.enable_meta_learning) or bgnn_outputs is None:
            zero = t.tensor(0.0, device=self.device)
            return zero, zero
        user_embed_all, item_embed_all, user_embeds_beh, item_embeds_beh = bgnn_outputs
        user_embed_all = user_embed_all.to(self.device)
        item_embed_all = item_embed_all.to(self.device)
        user_embeds_beh = user_embeds_beh.to(self.device)
        item_embeds_beh = item_embeds_beh.to(self.device)

        ssl_embeddings = [user_embeds_beh[i] for i in range(user_embeds_beh.shape[0])]
        ssl_embeddings.append(user_embed_all)
        unique_users = t.unique(user).detach()
        info_losses, user_step_index = self.SSL(ssl_embeddings, item_embeds_beh, user_embed_all, item_embed_all, unique_users)
        behavior_losses, user_index_list = self._calculate_behavior_bpr_losses(user, pos_tensor, neg_tensor, user_embeds_beh, item_embeds_beh)

        if user_step_index is None or (isinstance(user_step_index, t.Tensor) and user_step_index.numel() == 0):
            ssl_weights = [None for _ in range(len(info_losses))]
            behavior_weights = [None for _ in range(len(behavior_losses))]
        else:
            ssl_weights, behavior_weights = self.meta_weight_net(info_losses, behavior_losses, user_step_index, user_index_list, user_embeds_beh, user_embed_all)

        ssl_loss = self._aggregate_weighted_loss(info_losses, ssl_weights)
        beh_loss = self._aggregate_weighted_loss(behavior_losses, behavior_weights)
        return ssl_loss, beh_loss

    def _calculate_behavior_bpr_losses(self, user, pos_tensor, neg_tensor, user_embeds, item_embeds):
        losses = []
        user_index_list = []
        for idx in range(len(self.behaviors)):
            pos_items = pos_tensor[:, idx]
            neg_items = neg_tensor[:, idx]
            valid_mask = (pos_items >= 0) & (neg_items >= 0)
            if not t.any(valid_mask):
                losses.append(t.zeros(0, device=self.device))
                user_index_list.append(t.zeros(0, dtype=t.long, device=self.device))
                continue
            valid_users = user[valid_mask]
            user_index_list.append(valid_users)
            user_repr = user_embeds[idx][valid_users]
            pos_repr = item_embeds[idx][pos_items[valid_mask]]
            neg_repr = item_embeds[idx][neg_items[valid_mask]]
            diff = (user_repr * (pos_repr - neg_repr)).sum(dim=1)
            losses.append(-t.nn.functional.logsigmoid(diff))
        return losses, user_index_list

    def _aggregate_weighted_loss(self, losses, weights):
        device = self.device
        zero = t.tensor(0.0, device=device)
        if losses is None or weights is None:
            return zero
        total = zero
        counted = 0
        for loss_vec, weight_vec in zip(losses, weights):
            if loss_vec is None or (isinstance(loss_vec, t.Tensor) and loss_vec.numel() == 0):
                continue
            if not isinstance(loss_vec, t.Tensor):
                continue
            cur_loss = loss_vec.to(device).float()
            if weight_vec is None:
                cur_weight = t.ones_like(cur_loss, device=device)
            else:
                cur_weight = weight_vec.to(device).float()
                if cur_weight.shape != cur_loss.shape:
                    cur_weight = t.ones_like(cur_loss, device=device)
            total = total + (cur_loss * cur_weight).mean()
            counted += 1
        if counted == 0:
            return zero
        return total / counted

    def testEpoch(self, data_loader, save=False):
        self.model.eval()
        epochHR, epochNDCG = 0, 0
        cnt = 0
        tot = 0

        with t.no_grad():
            for user, item_i in data_loader:
                batch_user = user.cpu().numpy()
                batch_item = item_i.cpu().numpy()
                user_compute, item_compute, user_item1, user_item100 = self.sampleTestBatch(batch_user, batch_item)
                user_tensor = t.as_tensor(user_compute, dtype=t.long, device=self.device)
                item_tensor = t.as_tensor(item_compute, dtype=t.long, device=self.device)
                pred_scores = self.model.score_pairs(user_tensor, item_tensor)
                pred_matrix = pred_scores.view(len(batch_user), 100)

                hit, ndcg = self.calcRes(pred_matrix, user_item1, user_item100)
                epochHR += hit
                epochNDCG += ndcg
                cnt += 1
                tot += len(batch_user)

        result_HR = epochHR / tot
        result_NDCG = epochNDCG / tot
        print(f"Step {cnt}:  hit:{result_HR}, ndcg:{result_NDCG}")

        return result_HR, result_NDCG

    def calcRes(self, pred_i, user_item1, user_item100):  
     
        hit = 0
        ndcg = 0

    
        for j in range(pred_i.shape[0]):

            _, shoot_index = t.topk(pred_i[j], args.shoot) 
            shoot_index = shoot_index.cpu()
            shoot = user_item100[j][shoot_index]
            shoot = shoot.tolist()

            if type(shoot)!=int and (user_item1[j] in shoot):  
                hit += 1  
                ndcg += np.reciprocal( np.log2( shoot.index( user_item1[j])+2))  
            elif type(shoot)==int and (user_item1[j] == shoot):
                hit += 1  
                ndcg += np.reciprocal( np.log2( 0+2))
    
        return hit, ndcg  #int, float

    def sampleTestBatch(self, batch_user_id, batch_item_id):
       
        batch = len(batch_user_id)
        tmplen = (batch*100)

        sub_trainMat = self.trainMat[batch_user_id].toarray()  
        user_item1 = batch_item_id 
        user_compute = [None] * tmplen
        item_compute = [None] * tmplen
        user_item100 = [None] * (batch)

        cur = 0
        for i in range(batch):
            pos_item = user_item1[i] 
            negset = np.reshape(np.argwhere(sub_trainMat[i]==0), [-1])  
            pvec = self.labelP[negset] 
            pvec = pvec / np.sum(pvec)  
            
            random_neg_sam = np.random.permutation(negset)[:99]  
            user_item100_one_user = np.concatenate(( random_neg_sam, np.array([pos_item]))) 
            user_item100[i] = user_item100_one_user

            for j in range(100):
                user_compute[cur] = batch_user_id[i]
                item_compute[cur] = user_item100_one_user[j]
                cur += 1

        return user_compute, item_compute, user_item1, user_item100

    def setRandomSeed(self):
        np.random.seed(args.seed)
        t.manual_seed(args.seed)
        t.cuda.manual_seed(args.seed)
        random.seed(args.seed)

    def getModelName(self):  
        title = args.title
        ModelName = \
        args.point + \
        "_" + title + \
        "_" +  args.dataset +\
        "_" + modelTime + \
        "_lr_" + str(args.lr) + \
        "_reg_" + str(args.reg) + \
        "_batch_size_" + str(args.batch) + \
        "_gnn_layer_" + str(args.gnn_layer)

        return ModelName

    def saveHistory(self):  
        history = dict()
        history['loss'] = self.train_loss  
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        ModelName = self.modelName

        
        history_dir = os.path.join('./History', args.dataset)
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)

        with open(os.path.join(history_dir, ModelName + '.his'), 'wb') as fs:
            pickle.dump(history, fs)

    def saveModel(self):  
        ModelName = self.modelName

        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        savePath = r'./Model/' + args.dataset + r'/' + ModelName + r'.pth'
        params = {
            'epoch': self.curEpoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'history': history,
        }
        if self.enable_meta_learning and self.bgnn is not None and self.meta_weight_net is not None:
            params['bgnn_state_dict'] = self.bgnn.state_dict()
            params['meta_weight_state_dict'] = self.meta_weight_net.state_dict()

        
        model_dir = os.path.join('./Model', args.dataset)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        t.save(params, savePath)

    def loadModel(self, loadPath):
        ModelName = self.modelName
        # loadPath = r'./Model/' + args.dataset + r'/' + ModelName + r'.pth'
        loadPath = loadPath
        checkpoint = t.load(loadPath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.enable_meta_learning and self.bgnn is not None and 'bgnn_state_dict' in checkpoint:
            self.bgnn.load_state_dict(checkpoint['bgnn_state_dict'])
        if self.enable_meta_learning and self.meta_weight_net is not None and 'meta_weight_state_dict' in checkpoint:
            self.meta_weight_net.load_state_dict(checkpoint['meta_weight_state_dict'])

        self.curEpoch = checkpoint.get('epoch', 0) + 1
        history = checkpoint.get('history', None)
        if history is not None:
            self.train_loss = history.get('loss', [])
            self.his_hr = history.get('HR', [])
            self.his_ndcg = history.get('NDCG', [])

if __name__ == '__main__':
    print(args)
    my_model = Model()
    my_model.run()
    # my_model.test()
