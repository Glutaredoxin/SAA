import torch


def initial_indices(Qx_model, train_num):
    num_samples = len(Qx_model)
    num_gap = int(num_samples / train_num)
    indices = []
    for n in range(train_num):
        indices.append(n * num_gap)

    return indices

def initial_indices1(Qx_model, train_num):
    num_samples = len(Qx_model)
    num_gap = int(num_samples / train_num)
    indices = []
    for n in range(train_num):
        indices.append(n * num_gap)

    initialize = []
    for j in indices:
        temp_init = Qx_model[j:j+num_gap]
        temp_init = torch.mean(temp_init, dim=0)
        initialize.append(temp_init)
    initialize = torch.stack(initialize)
    return initialize

def sort(Qy, config, desc='train'):
    temp_list = []
    temp_num = -1
    for p in Qy:
        if p.item() == temp_num:
            if desc == 'train':
                temp_list.append((p.item()) * config.train_k_query + q)
            elif desc == 'val':
                temp_list.append((p.item()) * config.val_k_query + q)
            else:
                temp_list.append((p.item()) * config.test_k_query + q)
            q += 1
        else:
            q = 0
            if desc == 'train':
                temp_list.append((p.item()) * config.train_k_query + q)
            elif desc == 'val':
                temp_list.append((p.item()) * config.val_k_query + q)
            else:
                temp_list.append((p.item()) * config.test_k_query + q)
            q += 1
            temp_num = p.item()
    return temp_list

def sort_S(Sy, config, desc='train'):
    temp_list = []
    temp_num = -1
    for p in Sy:
        if p.item() == temp_num:
            if desc == 'train':
                temp_list.append((p.item()) * config.train_k_shot + q)
            elif desc == 'val':
                temp_list.append((p.item()) * config.val_k_shot + q)
            else:
                temp_list.append((p.item()) * config.test_k_shot + q)
            q += 1
        else:
            q = 0
            if desc == 'train':
                temp_list.append((p.item()) * config.train_k_shoty + q)
            elif desc == 'val':
                temp_list.append((p.item()) * config.val_k_shot + q)
            else:
                temp_list.append((p.item()) * config.test_k_shot + q)
            q += 1
            temp_num = p.item()
    return temp_list

def compute_acc_k(Qy, dis, config, desc='train'):
    temp_list = sort(Qy, config, desc)
    dis = dis[temp_list]
    acc = torch.eq(dis.argmax(dim=1), Qy).sum().item()
    return acc

if __name__ == '__main__':
    Qx_model = torch.randn(50, 640)
    initial_indices1(Qx_model, 10)
