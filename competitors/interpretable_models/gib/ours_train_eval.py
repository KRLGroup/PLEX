import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch_geometric.utils import subgraph
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def cross_validation_with_val_set(dataset, model, discriminator, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, inner_loop, mi_weight, pp_weight, seed, logger=None):
    
    def k_fold(dataset, folds, seed):
        skf = StratifiedKFold(folds, shuffle=True, random_state=seed)

        test_indices, train_indices = [], []
        for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
            test_indices.append(torch.from_numpy(idx).to(torch.long))

        val_indices = [test_indices[i - 1] for i in range(folds)]

        for i in range(folds):
            train_mask = torch.ones(len(dataset), dtype=torch.bool)
            train_mask[test_indices[i]] = 0
            train_mask[val_indices[i]] = 0
            train_indices.append(train_mask.nonzero().view(-1))
        return train_indices, test_indices, val_indices
    val_losses, accs, durations = [], [], []
    if folds == 1: k_fold = train_val_test_split

    results = []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds, seed))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        discriminator.to(device).reset_parameters()
        optimizer_local = Adam(discriminator.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        
        best_loss = float('inf')
        best_val_acc = 0
        best_acc = 0
        for epoch in range(1, epochs + 1):
            train_loss = train(model, discriminator, optimizer, optimizer_local, train_loader, mi_weight, pp_weight, inner_loop)

            if train_loss != train_loss:
                print('NaN')
                continue

            val_loss = eval_loss(model, val_loader)
            val_acc = eval_acc(model, val_loader)
            acc = eval_acc(model, test_loader)
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc':val_acc,
                'test_acc': acc,
                'best_test_acc': best_acc,
            }
            print(eval_info)
            

            # print(eval_info)
            #if val_loss< best_loss:
            #    best_loss = val_loss
            #    best_acc = acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_acc = acc



            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
        
#         expl_true = []
#         expl_pred = []
#         from torch_geometric.data import Data
#         test_loader = DenseLoader(test_dataset, 1, shuffle=False)
#         fidelity = []
#         expl_stab_pred = []
#         expl_stab_true = []
# #         for i, data in enumerate(test_dataset):
# #             if data.y.item() == 1: 
# #                 continue
# #             try:
# #                 data.batch=torch.zeros(data.x.shape[0])
# #                 out, _, _, _, assignment = model(data.to(device), with_assignment=True)
# #                 _,ind = torch.max(assignment,1)
# #                 expl = (1-ind).int()
# #                 batch = torch.zeros(data.x.shape[0]).to(data.x.device).long()
# #                 data_orig = Data(x=data.x, y=data.y,   edge_index=data.edge_index, edge_weight=None, batch=batch)
# #                 masked_edge_index, _ = subgraph(expl==0, data_orig.edge_index, relabel_nodes=True, num_nodes=data_orig.x.shape[0])
# #                 data_mask = Data(x=data.x[expl==0], y=data.y, edge_index=masked_edge_index, edge_weight=None, batch=batch[expl==0]) 
# #                 fidelity.append(model(data_orig)[0].softmax(-1)[0,0].item() - model(data_mask)[0].softmax(-1)[0,0].item())
# #                 # print(self.model(data_orig)[0].softmax(-1)[0,0].item(),  self.model(data_mask)[0].softmax(-1)[0,0].item())
# #                 for a, b in zip(expl, data.node_label):
# #                     expl_true.append(b.item())
# #                     expl_pred.append(a.item())
# #             except: pass
#         for i, data in enumerate(test_dataset):
#             # if data.y.item() == 1: 
#                 # continue
#             try:
#                 data.batch=torch.zeros(data.x.shape[0])
#                 out, _, _, _, assignment = model(data.to(device), with_assignment=True)
#                 c = out.argmax(-1).item()
#                 _,ind = torch.max(assignment,1)
#                 print(assignment)
#                 exit(1)
#                 expl = (ind==c).int()
                
#                 batch = torch.zeros(data.x.shape[0]).to(data.x.device).long()

#                 data_orig = Data(x=data.x, y=data.y,   edge_index=data.edge_index, edge_weight=None, batch=batch)
#                 masked_edge_index, _ = subgraph(expl==0, data_orig.edge_index, relabel_nodes=True, num_nodes=data_orig.x.shape[0])
#                 data_mask = Data(x=data.x[expl==0], y=data.y, edge_index=masked_edge_index, edge_weight=None, batch=batch[expl==0]) 
#                 fidelity.append(model(data_orig)[0].softmax(-1)[0,c].item() - model(data_mask)[0].softmax(-1)[0,c].item())
                
#             #     def duplicate_and_attach(graph, mask):
#             #         import copy
#             #         graph = copy.deepcopy(graph)
#             #         # Step 1: Identify the indices of the true values in the mask
#             #         true_indices = torch.nonzero(mask).view(-1)
                
#             #         if len(true_indices) == 0:
#             #             # No true values in the mask, nothing to duplicate and attach
#             #             return graph
                
#             #         # Step 2: Randomly pick an index from the true values of the mask
#             #         selected_index = torch.randint(len(true_indices), (1,)).item()
#             #         selected_node_index = true_indices[selected_index]
                
#             #         # Step 3: Duplicate the node corresponding to the selected index
#             #         duplicated_node = graph.x[selected_node_index].clone()
#             #         graph.x = torch.cat([graph.x, duplicated_node.view(1, -1)], dim=0)
                
#             #         # Step 4: Attach the duplicated node to another random node in the graph
#             #         attached_node_index = torch.randint(len(true_indices), (1,)).item()
#             #         graph.edge_index = torch.cat([graph.edge_index, torch.tensor([[selected_node_index, len(graph.x) - 1]], dtype=torch.long).t()], dim=1)
                
#             #         # Update the 'batch' parameter
#             #         graph.batch = torch.cat([graph.batch, torch.zeros(1, dtype=torch.long)])
                
#             #         return graph
#             #     data_stab = duplicate_and_attach(data_orig.cpu(), expl==0)
#             #     # stab_mask = (expl == 1) | ((expl == 0) & (torch.rand(data.node_label.shape)>0.2).to(expl.device))
#             #     # print(stab_mask)
#             #     # print(data[i].node_label)
#             #     # stab_edge_index, _ = subgraph(stab_mask, data_orig.edge_index, relabel_nodes=True, num_nodes=data_orig.x.shape[0])
#             #     # data_stab = Data(x=data.x[stab_mask], y=data.y, edge_index=stab_edge_index, edge_weight=None, batch=batch[stab_mask])
                
                
#             #     out_stab, _, _, _, assignment_stab = model(data_stab.to(device), with_assignment=True)
#             #     print(assignment_stab)
#             #     _,ind_stab = torch.max(assignment_stab,1)
#             #     expl_stab = (ind_stab==c).int()
#             #     print(expl_stab[:-1].detach().cpu().numpy().tolist())
#             #     print(expl.detach().cpu().numpy().tolist())
#             #     expl_stab_true.extend(expl_stab[:-1].detach().cpu().numpy().tolist())
#             #     expl_stab_pred.extend(expl.detach().cpu().numpy().tolist())
#             # # print(self.model(data_orig)[0].softmax(-1)[0,0].item(),  self.model(data_mask)[0].softmax(-1)[0,0].item())
               
#             except Exception as e: print(e)

#         from sklearn.metrics import accuracy_score
#         print('expl acc:',accuracy_score(expl_true, expl_pred))
#         print('fidelity:',np.mean(fidelity))
#         print('stability:',accuracy_score(expl_stab_true, expl_stab_pred))

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)
        accs.append(best_acc)
        
        print('(interm) Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(np.mean(accs), np.std(accs), np.mean(durations)))
        results.append((train_idx, val_idx, test_idx, model))

    print('Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(np.mean(accs), np.std(accs), np.mean(durations)))

    return 0, np.mean(accs), np.std(accs), results



def train_val_test_split(data, folds, seed):
    indices = list(range(len(data)))
    train_indices, test_val_indices = train_test_split(indices, test_size=0.2, shuffle=True, random_state=seed)
    val_indices = test_val_indices[:len(test_val_indices)//2]
    test_indices = test_val_indices[len(test_val_indices)//2:]

    return [train_indices], [val_indices], [test_indices]

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, discriminator, optimizer, local_optimizer, loader, mi_weight, pp_weight, inner_loop):
    model.train()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        data.y = data.y.long()
        try:
            data.x = data.x.float()
            out, all_pos_embedding, all_graph_embedding, all_pos_penalty = model(data)
            for j in range(0, inner_loop):
                local_optimizer.zero_grad()
                local_loss = - MI_Est(discriminator, all_graph_embedding.detach(), all_pos_embedding.detach())
                local_loss.backward()
                local_optimizer.step()
    
    
            optimizer.zero_grad()
            loss = F.nll_loss(out, data.y.view(-1))
    
            mi_loss = MI_Est(discriminator, all_graph_embedding, all_pos_embedding)
    
            loss = (1-pp_weight) * (loss + mi_weight*mi_loss) + pp_weight * all_pos_penalty #(1-mi_weight)*
    
            loss.backward()
            total_loss += loss.item() * num_graphs(data)
            optimizer.step()
        except Exception as e:
            print(e)
            continue



        
    return total_loss / len(loader.dataset)



def MI_Est(discriminator, embeddings, positive):

    batch_size = embeddings.shape[0]

    shuffle_embeddings = embeddings[torch.randperm(batch_size)]

    joint = discriminator(embeddings,positive)

    margin = discriminator(shuffle_embeddings,positive)

    #Donsker
    mi_est = torch.mean(joint) - torch.clamp(torch.log(torch.mean(torch.exp(margin))),-100000,100000)
    #JSD
    #mi_est = -torch.mean(F.softplus(-joint)) - torch.mean(F.softplus(-margin)+margin)
    #x^{2}
    #mi_est = torch.mean(joint**2) - 0.5* torch.mean((torch.sqrt(margin**2)+1.0)**2)
    return mi_est


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        try:
            data = data.to(device)
            with torch.no_grad():
                data.x = data.x.float()
                pred,_,_,_ = model(data)
                pred = pred.max(1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()
        except: continue
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        data.y = data.y.long()
        with torch.no_grad():
            try:
                data.x = data.x.float()
                out,_,_,_ = model(data)
                loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
            except Exception as e: 
                print(e)
                continue
        
    return loss / len(loader.dataset)
