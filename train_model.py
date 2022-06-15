import time

import torch
from sklearn.metrics import f1_score
from tqdm import tqdm


def get_f1_score(preds, y):
    y = y.cpu().data
    preds = preds.cpu().data
    top_pred = preds.argmax(1, keepdim=True)
    return f1_score(y, top_pred, average='macro', zero_division=0)


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, iterator, optimizer, criterion, label_tag):
    epoch_loss = 0
    model.train()
    predictions_list = []
    y_list = []
    for batch in tqdm(iterator):
        label = {'LOC_1': batch.LOC_1, 'LOC_2': batch.LOC_2, 'LOC_3': batch.LOC_3, 'SER_1': batch.SER_1,
                 'SER_2': batch.SER_2, 'SER_3': batch.SER_3, 'SER_4': batch.SER_4, 'PRI_1': batch.PRI_1,
                 'PRI_2': batch.PRI_2, 'PRI_3': batch.PRI_3, 'ENV_1': batch.ENV_1, 'ENV_2': batch.ENV_2,
                 'ENV_3': batch.ENV_3, 'ENV_4': batch.ENV_4, 'DISH_1': batch.DISH_1, 'DISH_2': batch.DISH_2,
                 'DISH_3': batch.DISH_3, 'DISH_4': batch.DISH_4, 'OTH_1': batch.OTH_1, 'OTH_2': batch.OTH_2}
        optimizer.zero_grad()
        text, text_lengths = batch.TEXT
        predictions = model(text, text_lengths)
        loss = criterion(predictions, label[label_tag])

        predictions_list.append(predictions)
        y_list.append(label[label_tag])

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    predictions = torch.cat(predictions_list, 0)
    y = torch.cat(y_list, 0)

    return epoch_loss / len(iterator), get_f1_score(predictions, y)


def evaluate(model, iterator, criterion, label_tag):
    epoch_loss = 0
    # epoch_acc = 0

    model.eval()
    predictions_list = []
    y_list = []
    with torch.no_grad():
        for batch in iterator:
            label = {'LOC_1': batch.LOC_1, 'LOC_2': batch.LOC_2, 'LOC_3': batch.LOC_3, 'SER_1': batch.SER_1,
                     'SER_2': batch.SER_2, 'SER_3': batch.SER_3, 'SER_4': batch.SER_4, 'PRI_1': batch.PRI_1,
                     'PRI_2': batch.PRI_2, 'PRI_3': batch.PRI_3, 'ENV_1': batch.ENV_1, 'ENV_2': batch.ENV_2,
                     'ENV_3': batch.ENV_3, 'ENV_4': batch.ENV_4, 'DISH_1': batch.DISH_1, 'DISH_2': batch.DISH_2,
                     'DISH_3': batch.DISH_3, 'DISH_4': batch.DISH_4, 'OTH_1': batch.OTH_1, 'OTH_2': batch.OTH_2}
            text, text_lengths = batch.TEXT
            predictions = model(text, text_lengths)
            loss = criterion(predictions, label[label_tag])
            # acc = categorical_accuracy(predictions, batch.SER_2)
            # acc = get_f1_score(predictions, batch.SER_2)
            predictions_list.append(predictions)
            y_list.append(label[label_tag])

            epoch_loss += loss.item()
            # epoch_acc += acc.item()

    predictions = torch.cat(predictions_list, 0)
    y = torch.cat(y_list, 0)

    return epoch_loss / len(iterator), get_f1_score(predictions, y)


def train_for_han(model, data_loader, optimizer, criterion, label_tag):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch_loss = 0
    model.train()
    predictions_list = []
    y_list = []
    for batch in tqdm(data_loader):

        optimizer.zero_grad()
        predictions = model(batch['ids'].to(device), batch['mask'].to(device), batch['sen_nums'], batch['sen_nums'])
        # start = time.time()
        loss = criterion(predictions, batch['L1'].to(device))

        predictions_list.append(predictions)
        y_list.append(batch['L1'])

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        # print(time.time() - start)

    predictions = torch.cat(predictions_list, 0)
    y = torch.cat(y_list, 0)

    return epoch_loss / len(data_loader), get_f1_score(predictions, y)


def evaluate_for_han(model, data_loader, criterion, label_tag):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch_loss = 0
    model.eval()
    predictions_list = []
    y_list = []
    with torch.no_grad():
        for batch in data_loader:
            predictions = model(batch['ids'].to(device), batch['mask'].to(device), batch['sen_nums'], batch['sen_nums'])
            loss = criterion(predictions, batch['L1'].to(device))

            predictions_list.append(predictions)
            y_list.append(batch['L1'])

            epoch_loss += loss.item()

    predictions = torch.cat(predictions_list, 0)
    y = torch.cat(y_list, 0)

    return epoch_loss / len(data_loader), get_f1_score(predictions, y)
