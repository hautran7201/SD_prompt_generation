import os
import evaluate
import datetime
import pytz
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict

def load_data_from_huggingface(loading_path, saving_path=''):
    dataset = load_dataset(loading_path)

    if saving_path!='':
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)

        for split, split_dataset in dataset.items():
            split_dataset.to_json(f'{saving_path}/{split}.json')

    return dataset

def get_run_name(extra_info=None):
    time_zone = pytz.timezone('Asia/Ho_Chi_Minh')
    current_datetime = datetime.datetime.now(time_zone)
    if extra_info:
        current_time = current_datetime.strftime(f"%D-%I:%M-%p")
    else:
        current_time = current_datetime.strftime(f"%D-%I:%M-%p")+f"-{extra_info}"
    return current_time