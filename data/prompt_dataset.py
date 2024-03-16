import torch 
from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from transformers.models.bart.modeling_bart import shift_tokens_right

class prompt_dataset:
    def __init__(
        self, 
        train_dataset,
        eval_dataset,
        pad_token_id,
        decoder_start_token_id,
        tokenizer,
        data_collator
    ):

        # Dataset
        self.dataset = DatasetDict(
            {
                'train': train_dataset.shuffle(seed=42).select(range(int(len(train_dataset)))),
                'validation': eval_dataset.shuffle(seed=42).select(range(int(len(eval_dataset))))
            }
        )
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.tokenizer = tokenizer
        self.data_collator = data_collator

        self.dataset.set_format(type='torch')
        self.tokenized_dataset = self.dataset.map(
            self.convert_to_feature,
            batched=True,
            remove_columns=self.dataset['train'].column_names
        )

    def convert_to_feature(self, example):
        input_encoding = self.tokenizer.batch_encode_plus(example['instruction'], padding='max_length', max_length=128, truncation=True)
        target_encoding = self.tokenizer.batch_encode_plus(example['prompt'], padding='max_length', max_length=128, truncation=True)

        labels = torch.Tensor(target_encoding['input_ids']).long()
        decode_input_ids = shift_tokens_right(
            labels,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id).long()

        labels[labels[:, :] == self.pad_token_id] == -100
        input_ids = torch.tensor(input_encoding['input_ids']).long()

        encodings = {
            'input_ids': input_ids,
            'attention_mask': input_encoding['attention_mask'],
            'decoder_input_ids': decode_input_ids,
            'labels': labels
        }

        return encodings
    
    def get_train_test_split(self, batch_size):

        train_dataloader = DataLoader(
            self.tokenized_dataset['train'],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=batch_size
        )

        eval_dataloader = DataLoader(
            self.tokenized_dataset['validation'],
            collate_fn=self.data_collator,
            batch_size=batch_size
        )

        return train_dataloader, eval_dataloader