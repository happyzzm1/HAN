import torch
import torch.nn as nn
import torch.optim as optim
import time
import data_preprocess
import model
import train_model
import utiles

debug = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_EPOCHS = 10
vocab_max_size = 50000
batch_size = 64
emb_dim = 100
hidden_dim = 256
output_dim = 4
N_LAYERS = 2
DROPOUT = 0.5
label_tag = 'LOC_1'
tag_str = 'L1'

train_file_pkl = 'md_train_examples.pkl' if debug else 'sa_train_examples.pkl'
train_file_csv = 'md_train.csv' if debug else 'sa_train.csv'
train_ids_pkl = 'sa_train_ids.pkl'
valid_ids_pkl = 'sa_validation_ids.pkl'
train_mask_pkl = 'sa_train_mask.pkl'
valid_mask_pkl = 'sa_validation_mask.pkl'
validation_file_csv = 'sa_validation.csv'
prefix = './data/'

# 初始化fields
fields = data_preprocess.build_fields()
# 构建迭代器
train = data_preprocess.read_train_data_from_pkl(fields, train_file_pkl, prefix)
# 根据数据构建词典
data_preprocess.build_vocab(train, fields, vocab_max_size)
# 获取训练文本，初始化填充字符和未知字符
TEXT = fields[1][1]
vocab = TEXT.vocab
TEXT.vocab.stoi['<pad>'] = 0
TEXT.vocab.itos[0] = '<pad>'
TEXT.vocab.stoi['<unk>'] = 1
TEXT.vocab.itos[1] = '<unk>'
vocab_size = len(vocab)
PAD_IDX = TEXT.vocab.stoi['<pad>']
UNK_IDX = TEXT.vocab.stoi['<unk>']

# 获取文本的ids矩阵
train_ids, valid_ids = data_preprocess.read_ids_from_pkl(train_ids_pkl, valid_ids_pkl, prefix)
# 获取文本的mask矩阵
print("---------------开始读取mask矩阵---------------")
train_mask, valid_mask = data_preprocess.read_mask_from_pkl(train_mask_pkl, valid_mask_pkl, prefix)
# 构建Dataset
print("---------------开始构建数据集---------------")
train_dataset_han = data_preprocess.DatasetForHAN(prefix + train_file_csv, train_ids, train_mask, tag_str)
valid_dataset_han = data_preprocess.DatasetForHAN(prefix + validation_file_csv, valid_ids, valid_mask, tag_str)
# 构建DataLoader
print("---------------开始构建迭代器---------------")
train_data_loader = data_preprocess.build_data_loader_for_han(train_dataset_han, batch_size)
valid_data_loader = data_preprocess.build_data_loader_for_han(valid_dataset_han, batch_size)
# 初始化模型
my_model = model.HAN(vocab_size=vocab_size, embed_dim=emb_dim, hidden_dim=hidden_dim, num_layers=N_LAYERS,
                              output_dim=output_dim, pad_idx=PAD_IDX, dropout=DROPOUT, device=device)
my_model = my_model.to(device)
# 设置词嵌入矩阵中填充字符和未知字符的向量为0
my_model.emb.weight.data[UNK_IDX] = torch.zeros(emb_dim)
my_model.emb.weight.data[PAD_IDX] = torch.zeros(emb_dim)
# 初始化优化器和损失函数
optimizer = optim.Adam(my_model.parameters())
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

print(f'The model has {utiles.count_parameters(my_model):,} trainable parameters')
print("---------------开始训练---------------")
# 开始训练
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_f1 = train_model.train_for_han(my_model, train_data_loader, optimizer, criterion, label_tag)
    valid_loss, valid_f1 = train_model.evaluate_for_han(my_model, valid_data_loader, criterion, label_tag)

    end_time = time.time()

    epoch_mins, epoch_secs = utiles.epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(my_model.state_dict(), 'HAN-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_f1 * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_f1 * 100:.2f}%')



