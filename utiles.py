def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cut_sentences(content):  # 实现分句的函数，content参数是传入的文本字符串
    end_flag = ['?', '!', '.', '？', '！', '。', '\n', ';']  # 结束符号，包含中文和英文的
    content_len = len(content)
    sentences = []  # 存储每一个句子的列表
    tmp_char = ''
    for idx, char in enumerate(content):
        tmp_char += char  # 拼接字符
        if (idx + 1) == content_len:  # 判断是否已经到了最后一位
            sentences.append(tmp_char.strip().replace('\ufeff', ''))  # 去除非法字符和空白字符
            break
        if char in end_flag:  # 判断此字符是否为结束符号
            # 再判断下一个字符是否为结束符号，如果不是结束符号，则切分句子
            next_idx = idx + 1
            if not content[next_idx] in end_flag:
                sentences.append(tmp_char.strip().replace('\ufeff', ''))  # 去除非法字符和空白字符
                tmp_char = ''

    return sentences  # 函数返回一个包含分割后的每一个完整句子的列表


def cut_sentences_by_list(content, max_len, max_sen_num, pad_token='<pad>'):  # 实现分句的函数，content参数是传入的单词列表
    # todo:进一步处理句子，去除无用符号
    end_flag = ['?', '!', '.', '？', '！', '。', '\n']  # 结束符号，包含中文和英文的
    content_len = len(content)
    tmp_char = []
    raw_sentences = []  # 存储每一个句子的列表
    sentences = []
    lengths = []  # 存储每个句子的长度

    for idx, char in enumerate(content):
        tmp_char.append(char)  # 拼接字符
        if (idx + 1) == content_len:  # 判断是否已经到了最后一位
            raw_sentences.append(tmp_char)
            break
        if char in end_flag:  # 判断此字符是否为结束符号
            # 再判断下一个字符是否为结束符号，如果不是结束符号，则切分句子
            next_idx = idx + 1
            if not content[next_idx] in end_flag:
                raw_sentences.append(tmp_char)
                tmp_char = []

    for i in range(len(raw_sentences)):
        if i + 1 > max_sen_num:
            break
        if len(raw_sentences[i]) >= max_len:
            lengths.append(max_len)
            sentences.append(raw_sentences[i][: max_len])
        else:
            lengths.append(len(raw_sentences[i]))
            sentences.append(raw_sentences[i] + [pad_token] * (max_len - len(raw_sentences[i])))
    return sentences, lengths  # 函数返回一个包含分割后的每一个完整句子的列表

