import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, output_dim, pad_idx, dropout):
        super().__init__()

        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_idx)
        self.bi_lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers,
                               dropout=dropout, bidirectional=True)
        self.output = nn.Linear(2 * hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = (sent_len, batch_size)
        embedded = self.emb(text)  # (sent_len, batch size, emb_dim)

        # pack sequence. lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), enforce_sorted=False)
        packed_output, (hidden, cell) = self.bi_lstm(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output = (sent_len, batch_size, hid_dim * D)
        # output over padding tokens are zero tensors

        # hidden = (num_layers * D, batch_size, hid_dim)
        # cell = (num_layers * D, batch_size, hid_dim)

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # [batch_size, hid_dim * D]

        # [batch_size, output_dim]
        return self.output(hidden)


class BiLSTMWithSelfAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, output_dim, pad_idx, dropout):
        super().__init__()

        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_idx)
        self.bi_lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers,
                               dropout=dropout, bidirectional=True)
        self.attention = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.inter_proj = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.output = nn.Linear(2 * hidden_dim, output_dim)
        self.LRelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = (sent_len, batch_size)
        embedded = self.emb(text)  # (sent_len, batch size, emb_dim)

        # pack sequence. lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), enforce_sorted=False)
        packed_output, (hidden, cell) = self.bi_lstm(packed_embedded)
        # hidden = (num_layers * D, batch_size, hid_dim)
        # cell = (num_layers * D, batch_size, hid_dim)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output = (sent_len, batch_size, hid_dim * D)
        # output over padding tokens are zero tensors

        output = self.dropout(output)

        attention_score = F.softmax(input=torch.tanh(self.attention(output)), dim=0).permute(1, 0, 2)
        # attention_score = (sent_len, batch_size, 1) -> (batch_size, sent_len, 1)

        raw_hidden = torch.bmm(output.permute(1, 2, 0), attention_score).squeeze(2)
        # raw_hidden = (batch_size, hid_dim * D)

        s = self.dropout(self.LRelu(self.inter_proj(raw_hidden)))

        # hidden = self.dropout(raw_hidden)
        # (batch_size, hid_dim * D)

        # (batch_size, output_dim)
        return self.output(s)


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

    def forward(self, x, mask, lengths, sen_nums):
        # x, mask = (batch_size, sen_num, sen_len)
        sen_num = x.size(1)
        sen_len = x.size(2)
        x = x.view(-1, sen_len)  # (batch_size * sen_num, sen_len)
        word_mask = mask.view(-1, sen_len).unsqueeze(2)  # (batch_size * sen_num, sen_len, 1)
        sen_mask = mask[:, :, 0].unsqueeze(2)
        # (batch_size, sen_num, 1)
        embedded = self.emb(x)  # (batch_size * sen_num , sen_len, embed_dim)
        # text_lengths （batch_size * sen_num）
        word_output, word_hidden = self.word_gru(embedded)
        word_output = torch.mul(word_output, word_mask)  # 手动mask，将padding处置为0
        # word_output: (batch_size * sen_num, sen_len, 2 * hidden_dim)

        word_attention = torch.tanh(self.word_fc(word_output))
        # word_attention: (batch_size * sen_num, sen_len, 2 * hidden_dim)
        # 计算u_it, (batch_size * sen_num, sen_len, 2 * hidden_dim)
        weights = F.softmax(torch.matmul(word_attention, self.word_query), dim=1)
        # 计算词权重weights: a_it
        # (batch_size * sen_num, sen_len, 2 * hidden_dim) * (2 * hidden_dim, 1), 第二个矩阵会通过广播机制扩展第一维为batch
        # 对每一句话进行softmax得到归一化权重, (batch_size * sen_num, sen_len, 1)

        # x = x.unsqueeze(2)  # (batch_size * sen_num, sen_len, 1)
        # weights = torch.where(x != int(self.pad_idx), weights, torch.full_like(x, 0, dtype=torch.float))
        # 去掉x中padding为0位置的attention比重, (batch_size * sen_num, sen_len, 1)

        # weights = weights / (torch.sum(weights, dim=1).unsqueeze(1)) + 1e-4
        # 去掉pad对应的attention比重后的重归一化。为了避免padding处的weights为0无法训练，加上一个极小值1e-4
        # (batch_size * sen_num, sen_len, 1)

        sentence_vector = torch.sum(weights * word_output, dim=1).view(-1, sen_num, word_output.size(2))
        # 哈达玛积与广播机制
        # weights: (batch_size * sen_num, sen_len, 1)
        # word_output: (batch_size * sen_num, sen_len, 2 * hidden_dim)
        # sentence_vector : (batch_size * sen_num, 2 * hidden_dim) ->(batch_size, sen_num, 2*hidden_dim)

        sentence_output, sentence_hidden = self.sentence_gru(sentence_vector)
        sentence_output = torch.mul(sentence_output, sen_mask)
        # sentence_output: (batch_size, sen_num, 2 * hidden_dim)
        sentence_attention = torch.tanh(self.sentence_fc(sentence_output))
        # 计算u_i, (batch_size, sen_num, 2 * hidden_dim)

        sen_weights = F.softmax(torch.matmul(sentence_attention, self.sentence_query), dim=1)
        # 计算句子权重a_i
        # (batch_size, sen_num, 2 * hidden_dim) * (2 * hidden_dim, 1), 第二个矩阵会通过广播机制扩展第一维为batch
        # sentence_weights: (batch_size, sen_num, 1)

        # x = x.view(-1, sen_num, x.size(1))  # (batch_size, sen_num, sen_len)
        # x = torch.sum(x, dim=2).unsqueeze(2)  # (batch_size, sen_num, 1), pad需要是0，这样求和还是0

        # sen_weights = torch.where(x != int(self.pad_idx), sen_weights, torch.full_like(x, 0, dtype=torch.float))
        # sen_weights = sen_weights / (torch.sum(sen_weights, dim=1).unsqueeze(1)) + 1e-4
        # (batch_size, sen_num, 1)

        # 计算文档向量v
        doc_vector = torch.sum(sen_weights * sentence_output, dim=1)
        # sen_weights: (batch_size, sen_num, 1)
        # sen_output: (batch_size, sen_num, 2 * hidden_dim)
        # (batch_size, sen_num, 2 * hidden_dim) -> (batch_size, 2 * hidden_dim)
        output = self.output(doc_vector)  # (batch_size, output_dim)
        return output
