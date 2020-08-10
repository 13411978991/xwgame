from data_processing.processing import *
from torch.utils.data import Dataset,DataLoader,TensorDataset
from moudel import models
import torch
import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):

        x = torch.FloatTensor([i[0] for i in datas]).to(self.device)
        z = torch.FloatTensor([i[1] for i in datas]).to(self.device)
        try:
            y = torch.LongTensor([i[2] for i in datas]).to(self.device)
        except:
            return (x, z)
        # pad前的长度(超过pad_size的设为pad_size)
        #        seq_len = torch.LongTensovr([_[2] for _ in datas]).to(self.device)
        return (x, z), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, batch_size, device):
    iter = DatasetIterater(dataset, batch_size, device)
    return iter



def data_iter(data_,batch_size):
    x,y = data_
    x,y = torch.FloatTensor(x),torch.LongTensor(y)
    data_ = TensorDataset(x,y)
    data_ = DataLoader(data_, batch_size=batch_size,shuffle=True)
    return data_


def get_k_fold_data(k, i, X):  ###此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    train_copy = X.fragment_id.drop_duplicates().to_list()
    random.seed(133)
    random.shuffle(train_copy)
    fold_size = len(train_copy) // 5  # 每份的个数:数据总条数/折数（组数）

    X_train = None
    for j in range(5):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        ##idx 为每组 valid
        X_part = train_copy[j * fold_size: (j + 1) * fold_size]
        if j == i:  ###第i折作valid
            X_valid = X_part
        elif X_train is None:
            X_train = X_part
        else:

            X_train.extend(X_part)
    X_train = pd.DataFrame(X_train, columns=['fragment_id'])
    X_valid = pd.DataFrame(X_valid, columns=['fragment_id'])
    X_train = X.merge(X_train, on='fragment_id', how='inner')
    X_valid = X.merge(X_valid, on='fragment_id', how='inner')

    X_train = strong(X_train, 0)             #数据增强


    train_x, train_y = feature(X_train)
    sent = []
    for i in range(train_x[0].shape[0]):
      sent.append((train_x[0][i],train_x[1][i],train_y[i]))

    train_x,train_y=feature(X_valid)
    sent_dev=[]
    for i in range(train_x[0].shape[0]):
      sent_dev.append((train_x[0][i],train_x[1][i],train_y[i]))

    random.shuffle(sent)
    random.shuffle(sent_dev)

    return sent, sent_dev


def k_fold(k, X_train, learning_rate,num_epochs,weight_decay,batch_size, sample_rate=None,test_file=None,sub=None):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    seed_list = [3, 337, 2020, 869, 87]
    for i in range(k):
        data = get_k_fold_data(k, i, X_train)  # 获取k折交叉验证的训练和验证数据
        feature_size = data[0][0][0].shape[0]
        co_feature = len(data[0][0][1])
        print('feature_size', feature_size, 'co_feature', co_feature)
        net = models.Model(filters=256, feature_size=feature_size, seq_len=61, co_feature=co_feature).to(device)  ### 实例化模型

        if sample_rate:
            pesudo_label = sample(sample_rate, sub, seed_list[i], test_file,batch_size)
            ### 每份数据进行训练,体现步骤三####
            train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size, i, sample_rate, pesudo_label)
        else:
            train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size, i, sample_rate)

        print('*' * 25, '第', i + 1, '折', '*' * 25)
        print('train_loss:%.6f' % min(train_ls[0]), 'train_acc:%.4f\n' % max(train_ls[1]),'valid loss:%.6f' % min(valid_ls[0]), 'valid_acc:%.4f' % max(valid_ls[1]))
        train_loss_sum += min(train_ls[0])
        valid_loss_sum += min(valid_ls[0])
        train_acc_sum += max(train_ls[1])
        valid_acc_sum += max(valid_ls[1])
        # print(train_ls,valid_ls)
    print('#' * 10, '最终k折交叉验证结果', '#' * 10)
    ####体现步骤四#####
    print('train_loss_sum:%.4f' % (train_loss_sum / k), 'train_acc_sum:%.4f\n' % (train_acc_sum / k),'valid_loss_sum:%.4f' % (valid_loss_sum / k), 'valid_acc_sum:%.4f' % (valid_acc_sum / k))



#########训练函数##########
def train(net, train_features, test_features, num_epochs, learning_rate, weight_decay, batch_size, i, sample_rate=None,
          pesudo_label=None):
    train_ls, test_ls = [[], []], [[], []]  ##存储train_loss,test_loss
    #
    train_iter = build_iterator(train_features, batch_size=8, device=device)
    dev_iter = build_iterator(test_features,batch_size=8, device=device)
    ### 将数据封装成 Dataloder 对应步骤（2）

    # 这里使用了Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_loss = float('inf')
    best_acc = 0
    count = 0
    glo = 0
    with tqdm(range(num_epochs)) as t:
        for epoch in t:
            for X, y in train_iter:  ###分批训练
                output = net(X)
                net.zero_grad()
                loss = F.cross_entropy(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            ### 得到每个epoch的 loss 和 accuracy
            loss_acc = evaluate(net, train_iter)

            if sample_rate:
                for X, y in pesudo_label:  ###分批训练
                    output = net(X)
                    net.zero_grad()
                    if epoch >= 10 and epoch <= 40:
                        alpha = (epoch - 10) / (30)
                    elif epoch > 40:
                        alpha = 1
                    else:
                        alpha = 0
                    loss = alpha * F.cross_entropy(output, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                ### 得到每个epoch的 loss 和 accuracy
                loss_acc_pesudo = evaluate(net, pesudo_label)

                train_ls[0].append((loss_acc[0] * 7292 * 0.8 + loss_acc_pesudo[0] * int(7500 * sample_rate)) / (
                            7292 * 0.8 + 7500 * sample_rate))
                train_ls[1].append((loss_acc[1] * 7292 * 0.8 + loss_acc_pesudo[1] * int(7500 * sample_rate)) / (
                            7292 * 0.8 + 7500 * sample_rate))
            else:
                train_ls[0].append(loss_acc[0])
                train_ls[1].append(loss_acc[1])
            if test_features is not None:
                loss_acc = evaluate(net, dev_iter)
                test_ls[0].append(loss_acc[0])
                test_ls[1].append(loss_acc[1])
                if best_acc <= loss_acc[1]:

                    best_loss = loss_acc[0]
                    best_acc = loss_acc[1]
                    if sample_rate:
                        torch.save(net.state_dict(), f'save_model/save_ssl{i}.model')
                    else:
                        torch.save(net.state_dict(), f'save_model/save{i}.model')
            if epoch % 10 == 0:
                if glo < best_acc:
                    glo = best_acc
                    count = 0
                else:
                    count += 1
                if count >= 3:
                    t.close()
                    break
                print('train_loss', train_ls[0][-1], 'train_acc', train_ls[1][-1])
                print('best_loss', best_loss, 'best_acc', best_acc, 'epoch', epoch, 'fold', i, '/n')
    return train_ls, test_ls


#从pseudo label抽样
def sample(sample_rate,pseudle_label,seed,test,batch_size):
    n_samples = int(pseudle_label.shape[0] * sample_rate)
    sampled_pseudo_data = pseudle_label.sample(n=n_samples,random_state=seed)
    sample_label = test.merge(sampled_pseudo_data,on='fragment_id',how='inner')
    sample_label = strong(sample_label,0)
    t,y = feature(sample_label)
    ln = sample_label['fragment_id'].drop_duplicates().shape[0]
    sent_test=[]
    for i in tqdm(range(ln)):
        sent_test.append((t[0][i],t[1][i],y[i]))
    random.seed(133)
    random.shuffle(sent_test)
    test_iter=build_iterator(sent_test, batch_size=batch_size,device=device)
    return test_iter




#评分函数
def acc_combo(y, y_pred):
    # 数值ID与行为编码的对应关系
    mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3',
        4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5',
        8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6',
        12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6',
        16: 'C_2', 17: 'C_5', 18: 'C_6'}
    # 将行为ID转为编码
    code_y, code_y_pred = mapping[y], mapping[y_pred]
    if code_y == code_y_pred: #编码完全相同得分1.0
        return 1.0
    elif code_y.split("_")[0] == code_y_pred.split("_")[0]: #编码仅字母部分相同得分1.0/7
        return 1.0/7
    elif code_y.split("_")[1] == code_y_pred.split("_")[1]: #编码仅数字部分相同得分1.0/3
        return 1.0/3
    else:
        return 0.0

def evaluate( model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    score = 0
    for item in zip(*[labels_all,predict_all]):
      score+=acc_combo(item[0],item[1])

    return loss_total / len(data_iter),acc