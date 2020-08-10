import torch
from data_processing import processing
from k_fold import *
from moudel import models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#读取数据
train_data = pd.read_csv(r'data/sensor_train.csv',nrows=6100)
test = pd.read_csv(r'data/sensor_test.csv')
sub = pd.read_csv(r'data/sub.csv')

#定义参数
class Config():
  def __init__(self):
    self.batch_size = 8  # mini-batch大小
    self.learning_rate = 1e-3  # 学习率
    self.num_epochs = 1#最大轮数
    self.weight_decay = 0#正则化系数，默认0
    self.sample_rate = 0.1#半监督采样率，需监督学习数据有较好的准确率，才会有提升，建议0.3
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = Config()

#预测函数
def eva(test,sub,SSL=False):
  # 预测
  t = feature(test)
  sent_test = []
  for i in range(sub.shape[0]):
    sent_test.append((t[0][i], t[1][i]))
  test_iter = build_iterator(sent_test, batch_size=config.batch_size, device=config.device)

  predict_output = [_ for _ in range(5)]
  for i in tqdm(range(5)):
    pre = np.zeros(shape=(0, 19))
    feature_size = sent_test[0][0].shape[0]
    co_feature = len(sent_test[0][1])
    net = models.Model(filters=256, feature_size=feature_size, seq_len=61, co_feature=co_feature).to(device)
    if SSL:
      net.load_state_dict(torch.load(f'save_model/save{i}.model'))
    net.eval()
    with torch.no_grad():
      flag = 0
      for X in test_iter:  ###分批训练
        output = net(X)
        pre = np.vstack((pre, output.cpu().numpy()))
    predict_output[i] = pre
  out = F.softmax(torch.tensor(predict_output[0]), dim=1) + F.softmax(torch.tensor(predict_output[1]),
                                                                      dim=1) + F.softmax(
    torch.tensor(predict_output[2]), dim=1) + F.softmax(torch.tensor(predict_output[3]), dim=1) + F.softmax(
    torch.tensor(predict_output[4]), dim=1)
  out = out.numpy()

  fina = []
  for item in range(sub.shape[0]):
    counts = np.argmax(out[item])
    # 返回众数
    fina.append(counts)
  sub['behavior_id'] = fina
  sub.columns = ['fragment_id', 'behavior_id']
  return sub

#五折交叉检验
k_fold(5, train_data, learning_rate=config.learning_rate,num_epochs=config.num_epochs,weight_decay=config.weight_decay,batch_size=config.batch_size)

sub = eva(test,sub)

#半监督学习五折
k_fold(5, train_data, learning_rate=config.learning_rate,num_epochs=config.num_epochs,weight_decay=config.weight_decay,batch_size=config.batch_size, sample_rate=config.sample_rate
      ,test_file=test,sub=sub)

sub = eva(test,sub,SSL=True)
