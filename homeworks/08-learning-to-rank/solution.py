#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Learning to Rank homework solution"""

import argparse
from timeit import default_timer as timer
import copy
import catboost
from catboost import utils
import pandas as pd
import pathlib
import os


EVAL_METRIC = 'NDCG:top=5;type=Exp'
DEFAULT_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.1,
    'l2_leaf_reg': 0.05,
    'depth': 7,
    'early_stopping_rounds': 100,  # stop if metric does not improve for N rounds
    'eval_metric': EVAL_METRIC,    # # metric used for early stopping
    'random_seed': 428,
    'verbose': 10
}


def create_column_desc_file(nb_features=573):
    # 'nb_features' obtained from opening the files
    feature_names = {j+1: f'feature_{j}' for j in range(1, nb_features+1)}
    utils.create_cd(
        label=0,
        group_id=1,
        feature_names=feature_names,
        output_path='tmp/train.cd'
    )


def create_model(loss_function, device_mode="CPU"):
    params = copy.deepcopy(DEFAULT_PARAMS)

    # Temporary directory that is used by catboost to store additional information
    catboost_info_dir = f"tmp/catboost_info.{loss_function.lower()}"

    params.update({
        'loss_function': loss_function,
        'train_dir': str(catboost_info_dir),
    })

    if device_mode == "GPU":
        params.update({
            'task_type': "GPU",
            'devices': '0'
        })
    
    return catboost.CatBoost(params)


def main():
    # Парсим опции командной строки
    parser = argparse.ArgumentParser(description='Learning to Rank homework solution')
    parser.add_argument('--train', action='store_true', help='run script in model training mode')
    parser.add_argument('--model_file', help='output ranking model file')
    parser.add_argument('--submission_file', help='output Kaggle submission file')
    parser.add_argument('--device', help='GPU for running on graphic card 0')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    # Будем измерять время работы скрипта
    start = timer()

    # Тут вы, скорее всего, должны проинициализировать фиксированным seed'ом генератор случайных чисел для того, чтобы ваше решение было воспроизводимо.
    # Например (зависит от того, какие конкретно либы вы используете):
    #
    # random.seed(42)
    # np.random.seed(42)
    # и т.п.
    # some pre-steps
    device = args.device if args.device and args.device == "GPU" else "CPU"
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    if not pathlib.Path('tmp/train.cd').is_file():
        create_column_desc_file()

    # Какой у нас режим: обучения модели или генерации сабмишна?
    if args.train:
        # Тут вы должны:
        # - загрузить датасет VKLR из папки args.data_dir
        data_dir = pathlib.Path(args.data_dir)
        # data_dir = root_data_dir.joinpath("Fold1")
        file_names_list = ['train.txt', 'vali.txt']
        file_path_list = []
        for file_name in file_names_list:
            file_path = data_dir.joinpath(file_name)
            file_path_list.append(file_path)
        
        # uploading datasets
        print("Uploading datasets")
        datasets = {}
        for file_name, file_path in zip(file_names_list, file_path_list):
            datasets[file_name[:-4]] = catboost.Pool(f'libsvm://{file_path}', column_description='tmp/train.cd')
        
        # - обучить модель с использованием train- и dev-сплитов датасета
        print("Training model on device:", device)
        model = create_model('YetiRank', device_mode=device)
        model.fit(datasets['train'], eval_set=datasets['vali'], use_best_model=True)
 
        # - при необходимости, подобрать гиперпараметры
        # Вручную попробовал несколько парматров и наиулучие указаны в DEFAULT_PARAMS словаре
        # - сохранить обученную модель в файле args.model_file
        model_file = args.model_file
        model.save_model(model_file, format="cbm")

    else:
        # Тут вы должны:
        # - загрузить тестовую часть датасета VKLR из папки args.data_dir
        data_dir = pathlib.Path(args.data_dir)
        file_path = data_dir.joinpath('test.txt')
        test_pool = catboost.Pool(f'libsvm://{file_path}', column_description='tmp/train.cd')
        # - загрузить модель из файла args.model_file
        model = catboost.CatBoost()
        model.load_model(args.model_file, format="cbm")
        # - применить модель ко всем запросам и документам из test.txt
        y_pred = model.predict(test_pool)
        nb_docs = y_pred.size
        # - переранжировать документы по каждому из запросов так, чтобы более релевантные шли сначала
        result_df = pd.DataFrame({
            'QueryId': test_pool.get_group_id_hash(), 
            'DocumentId': pd.Series(range(nb_docs)), 
            'prediction': y_pred})
        
        # - сформировать ваш сабмишн для заливки на Kaggle и сохранить его в файле args.submission_file
        result_df = result_df.sort_values(by=['QueryId', 'prediction'], ascending=[True, False])
        result_df[['QueryId', 'DocumentId']].to_csv(args.submission_file, index=False)

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
