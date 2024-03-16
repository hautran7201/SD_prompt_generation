import os
import evaluate
import datetime
import pytz
import pandas as pd
import json
from datasets import (
  load_dataset, 
  concatenate_datasets, 
  Dataset, 
  DatasetDict,
  load_from_disk
)


def save_to_json(data: dict, file_path: str):
    # Lưu dictionary vào file JSON
    with open(file_path, "w") as json_file:
        json.dump(data, json_file)


def load_prompt_data(
        loading_path,        
        instruction_column_name: str,
        prompt_column_name: str,
        test_ratio=0.1,
        validation_ratio=0.1,
        saving_path='',
        load_data_disk=False
    ):

    # Load dataset
    if load_data_disk:
        dataset = load_from_disk(loading_path)
    else:
        dataset = load_dataset(loading_path)    

    assert 'train' in dataset.keys(), "Train key do not exists !!!"

    # Remove column
    column_names = dataset['train'].column_names    
    if 'instruction' not in column_names:
        dataset = dataset.rename_column(instruction_column_name, 'instruction')
    if 'prompt' not in column_names:
        dataset = dataset.rename_column(prompt_column_name, 'prompt')

    columns_to_select = ['instruction', 'prompt']
    for split in dataset.keys():
        splited_dataset = pd.DataFrame(dataset[split])
        splited_dataset = splited_dataset.dropna()
        dataset[split] = Dataset.from_pandas(
            splited_dataset
        ).select_columns(column_names=columns_to_select)
    
    if 'test' not in dataset.keys():
        dataset['test'] = dataset['train'].train_test_split(
            test_size=test_ratio,
            shuffle=True
        ).pop('test')

    if 'validation' not in dataset.keys():
        dataset['validation'] = dataset['train'].train_test_split(
            test_size=validation_ratio,
            shuffle=True
        ).pop('test')

    # Save dataset to disk 
    if saving_path!='':
        dataset.save_to_disk(saving_path)

    return dataset


def get_run_name(extra_info=None):
    time_zone = pytz.timezone('Asia/Ho_Chi_Minh')
    current_datetime = datetime.datetime.now(time_zone)
    if extra_info:
        current_time = current_datetime.strftime(f"%D-%I:%M-%p")
    else:
        current_time = current_datetime.strftime(f"%D-%I:%M-%p")+f"-{extra_info}"
    return current_time