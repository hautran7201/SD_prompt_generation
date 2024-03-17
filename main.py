import pandas as pd
import os
import torch
from omegaconf import OmegaConf
from opt import config_parser
from model import prompt_model
from utils import load_prompt_data, get_run_name
from data.prompt_dataset import prompt_dataset
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader


def train(config):
    # === Model ===
    model_path = config.model.model_path
    model = prompt_model(model_path)
    tokenizer = model.tokenizer
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # === Load data ===
    datasets = []
    for data_config in config.data.datasets:
        dataset = load_prompt_data(
            loading_path=data_config.data_path,
            instruction_column_name=data_config.instruction_column,
            prompt_column_name=data_config.prompt_column,
            saving_path=data_config.saving_path,
            load_data_disk=data_config.load_from_disk
        )
        datasets.append(dataset)

    # === Concat dataset ===
    train_dataset = concatenate_datasets([dataset['train'] for dataset in datasets])
    eval_dataset = concatenate_datasets([dataset['validation'] for dataset in datasets])


    # === Create dataset ===
    dataset = prompt_dataset(
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        pad_token_id = model.pad_token_id,
        decoder_start_token_id = model.decoder_start_token_id,
        tokenizer = tokenizer,
        data_collator = data_collator
    )

    # === Data loader ===
    train_dataloader, eval_dataloader = dataset.get_train_test_split(config.model.batch_size)

    
    # === Wandb ===
    if config.wandb.token:
        # === wandb args ===
        project = config.wandb.project
        training_group = config.wandb.training_group
        eval_group = config.wandb.eval_group

        # === Import ===
        import wandb
        # === Wandb login === 
        os.environ['WANDB_API_KEY'] = config.wandb.token
        # === Logs ===
        training_run_log = wandb.init(project=project, group=training_group, name=get_run_name())
        eval_run_log = wandb.init(project=project, group=eval_group, name=get_run_name())
    else:
        training_run_log = None
        eval_run_log = None

    if config.huggingface.write_token:
        os.environ['HF_TOKEN'] = config.huggingface.write_token

    model.training(
        epochs            = config.model.epochs,
        train_dataloader  = train_dataloader, 
        eval_dataloader   = eval_dataloader, 
        num_warmup_steps  = config.model.num_warmup_steps,
        lr                = config.model.learning_rate,
        training_run_log  = training_run_log,
        eval_run_log      = eval_run_log,
        out_dir           = config.model.model_out_dir,
        hub_id            = config.huggingface.hub_id
    )

    return model

def evaluation(config):
    # === Model ===
    model_path = config.model.model_path
    model = prompt_model(model_path)
    tokenizer = model.tokenizer
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # === Load data ===
    datasets = []
    for data_config in config.data.datasets:
        dataset = load_prompt_data(
            loading_path=data_config.data_path,
            instruction_column_name=data_config.instruction_column,
            prompt_column_name=data_config.prompt_column,
            saving_path=data_config.saving_path,
            load_data_disk=data_config.load_from_disk
        )
        datasets.append(dataset)

    # === Concat dataset ===
    test_dataset = concatenate_datasets([dataset['test'] for dataset in datasets])

    # === Data loader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.model.batch_size,
        shuffle=True
    )

    # === Wandb ===
    if config.wandb.token:
        # === wandb args ===
        project = config.wandb.project
        training_group = config.wandb.training_group
        eval_group = config.wandb.eval_group

        # === Import ===
        import wandb
        # === Wandb login === 
        os.environ['WANDB_API_KEY'] = config.wandb.token
        # === Logs ===
        eval_run_log = wandb.init(project=project, group=eval_group, name=get_run_name())
    else:
        eval_run_log = None

    # === Evaluation ===
    result = model.evaluation(
        eval_dataloader=test_dataloader,
        eval_run_log=eval_run_log
    )

    return result

def inference(config, data):
    # === Model ===
    model_path = config.model.model_path
    model = prompt_model(model_path)
    tokenizer = model.tokenizer
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # === Inference ===
    result = model.inference(data)

    return result

if __name__ == '__main__':
    # Config
    args = config_parser()    
    config = OmegaConf.load(args.config)

    # === Training ===
    if args.train_only:
        train(config)

    # === Evaluation ===
    if args.eval_only:
        result = evaluation(config)
        print(result)

    # === Inference ===
    data = {'instruction': args.infer_data}
    if args.infer_only:
        result = inference(config, data)
        print(result)