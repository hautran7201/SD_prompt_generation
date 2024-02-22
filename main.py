import pandas as pd
import wandb
import os
from opt import config_parser
from model import prompt_model
from utils import load_data_from_huggingface, get_run_name
from data.prompt_dataset import prompt_dataset
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import DataCollatorForSeq2Seq


# === Wandb login === 
os.environ['WANDB_API_KEY'] = 'token'


# === Model ===
model_path = 'merve/chatgpt-prompt-generator-v12'
model = prompt_model(model_path)
tokenizer = model.tokenizer
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# === sd dataset ====
sd_huggingface_path = 'MadVoyager/stable_diffusion_instructional_dataset'
sd_path = r'data/sd_dataset'

sd_dataset = load_data_from_huggingface(
    loading_path=sd_huggingface_path,
    saving_path=sd_path
)

# select and rename columns
sd_dataset = sd_dataset.rename_column('INSTRUCTION', 'instruction')
sd_dataset = sd_dataset.rename_column('RESPONSE', 'prompt')
sd_dataset = sd_dataset['train'].train_test_split(train_size=0.9)
sd_dataset['validation'] = sd_dataset.pop('test')
sd_dataset = sd_dataset.remove_columns(['SOURCE'])


# === mj dataset ===
mj_huggingface_path = 'digitalwas-solutions/midjourney-prompts'
mj_path = r'data/mj_dataset'

mj_dataset = load_data_from_huggingface(
    loading_path=mj_huggingface_path,
    saving_path=mj_path
)

# select and rename columns
mj_dataset = mj_dataset.rename_column('autotrain_text', 'instruction')
mj_dataset = mj_dataset.rename_column('Prompt', 'prompt')
train_df_mj_dataset = pd.DataFrame(mj_dataset['train'])
test_df_mj_dataset = pd.DataFrame(mj_dataset['validation'])

# dropna row
train_df_mj_dataset = train_df_mj_dataset.dropna()
test_df_mj_dataset = train_df_mj_dataset.dropna()

train_mj_dataset = Dataset.from_pandas(train_df_mj_dataset)
test_mj_dataset = Dataset.from_pandas(test_df_mj_dataset)

mj_dataset = DatasetDict(
    {
        'train': train_mj_dataset,
        'validation': test_mj_dataset
    }
)


# === Concat dataset ===
train_dataset = concatenate_datasets([sd_dataset['train'], mj_dataset['train']])
eval_dataset = concatenate_datasets([sd_dataset['validation'], mj_dataset['validation']])


# === Create dataset ===
dataset = prompt_dataset(
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    pad_token_id = model.pad_token_id,
    decoder_start_token_id = model.decoder_start_token_id,
    tokenizer = tokenizer,
    data_collator = data_collator
)

train_dataloader, eval_dataloader = dataset.get_train_test_split(batch_size=8)


if __name__ == '__main__':
    # Config
    args = config_parser()

    project = 'prompt-generator-t5'
    training_group = 'training_group'
    eval_group = 'eval_group'
    training_run_log = wandb.init(project=project, group=training_group, name=get_run_name())
    eval_run_log = wandb.init(project=project, group=eval_group, name=get_run_name())

    # === Training ===
    if args.train_only:
        model.training(
            epochs            = args.epochs,
            train_dataloader  = train_dataloader, 
            eval_dataloader   = eval_dataloader, 
            num_warmup_steps  = args.num_warmup_steps,
            lr                = args.learning_rate,
            training_run_log  = training_run_log,
            eval_run_log      = eval_run_log,
            out_dir           = args.model_out_dir
        )

    # === Eval ===
    if args.eval_only:
        model.evaluate(
            eval_dataloader = eval_dataloader,
            eval_run_log    = eval_run_log
        )

    # === Infer ===
    if args.infer_only:
        model.inference_model(
            [args.infer_data]
        )
