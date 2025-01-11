#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Neural ranking homework solution"""

import argparse
from timeit import default_timer as timer

import os
import numpy as np
import pandas as pd
import pathlib
import gc
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.retrieval import RetrievalNormalizedDCG
from transformers import AutoTokenizer, AutoModel, XLMRobertaTokenizer, XLMRobertaTokenizer
from sklearn.metrics import roc_auc_score

# global tokenizer
temp_data_dir = pathlib.Path("temp_data")
if not os.path.exists(temp_data_dir):
    os.mkdir(temp_data_dir)
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', cache_dir=temp_data_dir)


class config:
    EPOCHS=1
    LR=1e-4
    WD=0.01
    ACCUM_BS=1
    DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    LOG_INTERVAL=60


def datareader_to_df(path_list, column_names, data_dir, separator='\t'):
    n = len(path_list)
    result = []
    for j in range(n):
        file_path = data_dir.joinpath(path_list[j])
        result.append(pd.read_csv(file_path, sep=separator, header=None, names=column_names))
        print(f'Shape dataframe {path_list[j]}: {result[-1].shape}')
    return result


def read_line(docs_file, offset_list, n):
    with open(docs_file, 'rb') as file:
        file.seek(offset_list[n-1])
        line = file.readline()
    return line.decode('utf-8').rstrip('\n').split('\t')


def compose_batch(batch):
    texts = [x for x, _ in batch]
    ys = torch.tensor([y for _, y in batch]).reshape((-1, 1)).float()
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
    return tokens, ys


class RankDataset(Dataset):
    def __init__(self, data, doc_path, offset_list, neg_p=1.0):
        self.doc_path = doc_path
        self.offset_list = offset_list
        self.neg_p = neg_p
        if self.neg_p < 1.:
            self.data = pd.concat([data[data['label'] != 1],
                                   data[data['label'] == 1].sample(frac=self.neg_p, random_state=13)])
        else:
            self.data = data

    def __getitem__(self, index):
        query, docid, label = self.data.iloc[index, [3, 1, 2]]
        
        documentid, url, title, body = read_line(self.doc_path, self.offset_list, int(docid[1:]))
        # just in case - just to make sure we find the docs, but this should not get any error
        assert docid == documentid, "Document not found"
        text = title + ' ' + body
        # we do minmax scale from 0 to 3 to 0 to 1 in the labels
        return [query.lower(), text.lower()], label/3

    def __len__(self):
        return len(self.data)


class RankBert(nn.Module):
    def __init__(self, train_layers_count=2):
        super(RankBert, self).__init__()

        self.bert = AutoModel.from_pretrained("xlm-roberta-base")
        self.config = self.bert.config

        # freeze all the layers without bias and LN
        for name, par in self.bert.named_parameters():
            if 'bias' in name or 'LayerNorm' in name:
                continue
            par.requires_grad = False

        # unfreeze some of the layers
        layer_count = self.config.num_hidden_layers
        for i in range(train_layers_count):
            for par in self.bert.encoder.layer[layer_count - 1 - i].parameters():
                par.requires_grad = True

        # map cls token embedding to relevance score
        self.head = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        x = self.bert(input_ids=input_ids,
                      token_type_ids=token_type_ids,
                      attention_mask=attention_mask
                      )[0][:, 0, :]   # hidden_state of [CLS]
        x = self.head(x)
        return x


def move_batch_to_device(batch, device):
    batch_x, y = batch
    for key in batch_x:
        batch_x[key] = batch_x[key].to(device)
    y = y.to(device)
    return batch_x, y


def train_one_epoch(epoch_index, train_dataloader, optimizer, model, loss_fn, scheduler):
    running_loss = 0.
    running_auc = 0.
    last_loss = 0.

    device = config.DEVICE
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, batch in enumerate(train_dataloader):
        # Every data instance is an input + label pair
        batch_x, y = move_batch_to_device(batch, device)
        # batch_x, y = batch    # alternate way of not sending to device

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(**batch_x)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, y)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        scheduler.step()

        # Gather data to report
        running_loss += loss.item()

        y = y.cpu().int().numpy()
        if y.sum() > 0:
            # compute metric
            with torch.no_grad():
                auc = roc_auc_score(y,
                                    outputs.cpu().numpy(),
                                    average=None)
            running_auc += np.mean(auc)
        else:
            running_auc += 1

        if i % config.LOG_INTERVAL == config.LOG_INTERVAL - 1:
            last_loss = running_loss / config.LOG_INTERVAL    # loss per batch
            last_auc = running_auc / config.LOG_INTERVAL    # loss per batch
            print('  batch {} loss: {}, auc: {}'.format(i + 1, last_loss, last_auc))

            running_loss = 0.
            running_auc = 0.

        if (i+1) % 10 == 0:    # clean up memory
            gc.collect()
            torch.cuda.empty_cache()

    return last_loss, last_auc


def get_test_preds(model, test_dataloader):
    model.eval()   # eval mode
    y_test = []

    for i, batch in enumerate(test_dataloader):    # итерируемся по батчам
        batch_x, _ = move_batch_to_device(batch, config.DEVICE)
        with torch.no_grad():
            preds = model(**batch_x)
            y_test += [preds]
        if (i + 1) % 15 == 0:
            print("Done batch number", i+1)

    y_test = torch.cat(y_test).view(-1).cpu().numpy()
    return y_test


def main():
    # Парсим опции командной строки
    parser = argparse.ArgumentParser(description='Neural ranking homework solution')
    parser.add_argument('--submission_file', required=True, help='output Kaggle submission file')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    # Будем измерять время работы скрипта
    start = timer()
    
    # Creating a base folder
    checkpoint_dir = pathlib.Path("cross_encoder_checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
   
    # Тут вы, скорее всего, должны проинициализировать фиксированным seed'ом генератор случайных чисел для того, чтобы ваше решение было воспроизводимо.
    # Например (зависит от того, какие конкретно либы вы используете):
    #
    # random.seed(42)
    # np.random.seed(42)
    torch.manual_seed(18)
    # transformers.set_seed(422)
    # и т.п.

    # Дальше вы должны:
    # - загрузить датасет VK MARCO из папки args.data_dir
    data_dir = pathlib.Path(args.data_dir)
    # Loading queries
    path_list = ["vkmarco-doctrain-queries.tsv", "vkmarco-docdev-queries.tsv", "vkmarco-doceval-queries.tsv"]
    column_names = ['QueryId', 'query']
    queries_train_df, queries_val_df, queries_test_df = datareader_to_df(path_list, column_names, data_dir)

    # Loading qrels + sample submission
    path_list_qrels = ["vkmarco-doctrain-qrels.tsv", "vkmarco-docdev-qrels.tsv"]
    column_names_qrels = ['QueryId', 'unused', 'DocumentId', 'label']
    qrels_train_df, qrels_val_df = datareader_to_df(path_list_qrels, column_names_qrels, data_dir, separator=' ')

    samplesub_file = data_dir.joinpath("sample_submission.csv")
    samplesub_df = pd.read_csv(samplesub_file)

    # We don't load the documents to save memory ram - we create a list of addresses instead
    docs_file = data_dir.joinpath("vkmarco-docs.tsv")
    offset_list = []
    offset = 0
    with open(docs_file, 'rb') as file:
        for line in file:           
            offset_list.append(offset)
            offset = file.tell()

    # Adapting the data for training
    train_data = pd.merge(qrels_train_df, queries_train_df, how="left", on="QueryId").drop(columns="unused")
    val_data = pd.merge(qrels_val_df, queries_val_df, how="left", on="QueryId").drop(columns="unused")

    samplesub_df['label'] = 0
    test_data = pd.merge(samplesub_df, queries_test_df, how="left", on="QueryId")

    dataset_train = RankDataset(train_data, docs_file, offset_list, neg_p=0.55)
    dataset_valid = RankDataset(val_data, docs_file, offset_list)
    
    # - обучить модель с использованием train- и dev-сплитов датасета
    # train- и dev-сплитов датасета
    dataset_train = RankDataset(train_data, docs_file, offset_list, neg_p=0.55)    # 0.55 decided after analysing the distribution of labels
    dataset_valid = RankDataset(val_data, docs_file, offset_list)
    train_dataloader = DataLoader(dataset_train, shuffle=True, batch_size=128, collate_fn=compose_batch)
    valid_dataloader = DataLoader(dataset_valid, shuffle=False, batch_size=128, collate_fn=compose_batch)

    # ndcg@10 to eval the model
    ndcg = RetrievalNormalizedDCG(top_k=10)
    # start the model
    model = RankBert(train_layers_count=2)
    model.to(config.DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WD)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    pct_start=0.1,
                                                    max_lr=config.LR,
                                                    epochs=config.EPOCHS,
                                                    steps_per_epoch=len(train_dataloader))

    epoch_number = 0

    # best_vloss to start with
    best_vloss = float('inf')

    for epoch in range(config.EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        # avg_loss, avg_auc = train_one_epoch(epoch_number, writer)
        avg_loss, avg_auc = train_one_epoch(epoch_number, train_dataloader, optimizer, model, loss_fn, scheduler)
        
        running_vloss = 0.0
        running_vauc = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            preds = []
            for i, batch in enumerate(valid_dataloader):
                batch_x, y = move_batch_to_device(batch, config.DEVICE)
                voutputs = model(**batch_x)
                vloss = loss_fn(voutputs, y)
                running_vloss += vloss

                y = y.cpu().int().numpy()
                if y.sum() > 0:
                    #compute metric
                    with torch.no_grad():
                        auc = roc_auc_score(y,
                                            voutputs.cpu().numpy(),
                                            average=None)
                    running_vauc += np.mean(auc)
                else:
                    running_vauc += 1

                preds.append(voutputs)
                if (i + 1) % 10 == 0:
                    print("\rvalid %d" % (i + 1), end = '', flush=True)
            
        preds = torch.cat(preds).view(-1).cpu()
        
        # compute valid ndcg@10
        val_ndcg = ndcg(preds, torch.tensor(val_data['label'].values), torch.LongTensor(val_data['QueryId'].values)).item()

        avg_vloss = running_vloss / (i + 1)
        avg_vauc = running_vauc / (i + 1)
        print('\nLOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('AUC train {} valid {}'.format(avg_auc, avg_vauc))
        print('NDCG@10 valid {}'.format(val_ndcg))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

            checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_vloss': best_vloss}
            checkpoint_file = checkpoint_dir.joinpath(f'ckpt_epoch_{epoch}_loss{best_vloss}.pt')
            torch.save(checkpoint, checkpoint_file)

        epoch_number += 1

    # - загрузить пример сабмишна из args.data_dir/sample_submission.csv
    # - применить обученную модель ко всем запросам и документам из примера сабмишна
    test_dataset = RankDataset(test_data, docs_file, offset_list)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1024, collate_fn=compose_batch)
    y_test = get_test_preds(model.float(), test_dataloader)
    
    # - переранжировать документы из примера сабмишна в соответствии с предиктами вашей модели
    test_data['pred'] = y_test
    result_df = test_data.sort_values(by=['QueryId', 'pred'], ascending=[True, False])
    
    # - сформировать ваш сабмишн для заливки на Kaggle и сохранить его в файле args.submission_file
    result_df[['QueryId', 'DocumentId']].to_csv(args.submission_file, index=False)

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
