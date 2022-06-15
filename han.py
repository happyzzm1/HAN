import torch
import torch.nn as nn
import torch.nn.functional as F


class HAN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, output_dim, pad_idx, dropout, device):
        super(HAN, self).__init__()
        self.pad_idx = pad_idx
        self.device = device
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_idx)
        # 词注意力
        self.word_gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True,
                               batch_first=True, dropout=dropout)
        self.word_query = nn.Parameter(torch.randn(2 * hidden_dim, 1), requires_grad=True)
        # 公式中的u(w), (2 * hidden_dim, 1)
        self.word_fc = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        # 句子注意力
        self.sentence_gru = nn.GRU(input_size=2 * hidden_dim, hidden_size=hidden_dim, num_layers=num_layers,
                                   bidirectional=True, batch_first=True)
        self.sentence_query = nn.Parameter(torch.randn(2 * hidden_dim, 1), requires_grad=True)
        # 公式中的u(s),(2 * hidden_dim, 1)
        self.sentence_fc = nn.Linear(2 * hidden_dim, 2 * hidden_dim)

        # 文档分类
        self.output = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x, mask):
        # x, mask = (batch_size, sen_num, sen_len)
        sen_num = x.size(1)
        sen_len = x.size(2)
        x = x.view(-1, sen_len)  # (batch_size * sen_num, sen_len)
        word_mask = mask.view(-1, sen_len).unsqueeze(2)  # (batch_size * sen_num, sen_len, 1)
        sen_mask = mask[:, :, 0].unsqueeze(2)  # (batch_size, sen_num, 1)
        embedded = self.emb(x)  # (batch_size * sen_num , sen_len, embed_dim)
        word_output, word_hidden = self.word_gru(embedded)
        word_output = torch.mul(word_output, word_mask)  # 手动mask，将padding处置为0
        word_attention = torch.tanh(self.word_fc(word_output))  # u_it = (batch_size * sen_num, sen_len, 2 * hidden_dim)
        weights = F.softmax(torch.matmul(word_attention, self.word_query), dim=1)  # 计算词权重a_it
        # (batch_size * sen_num, sen_len, 2 * hidden_dim) * (2 * hidden_dim, 1), 第二个矩阵会通过广播机制扩展第一维为batch
        # 对每一句话进行softmax得到归一化权重, (batch_size * sen_num, sen_len, 1)
        sentence_vector = torch.sum(weights * word_output, dim=1).view(-1, sen_num, word_output.size(2))
        # sentence_vector = (batch_size, sen_num, 2*hidden_dim)
        sentence_output, sentence_hidden = self.sentence_gru(sentence_vector)
        sentence_output = torch.mul(sentence_output, sen_mask)  # sentence_output: (batch_size, sen_num, 2 * hidden_dim)
        sentence_attention = torch.tanh(self.sentence_fc(sentence_output))
        sen_weights = F.softmax(torch.matmul(sentence_attention, self.sentence_query), dim=1)  # 计算句子权重a_i
        # sentence_weights: (batch_size, sen_num, 1)
        doc_vector = torch.sum(sen_weights * sentence_output, dim=1)  # 计算文档向量v
        # (batch_size, sen_num, 2 * hidden_dim) -> (batch_size, 2 * hidden_dim)
        output = self.output(doc_vector)  # (batch_size, output_dim)
        return output
