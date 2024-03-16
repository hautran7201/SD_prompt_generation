import torch
import numpy as np
import evaluate
import os
import json

from utils import get_run_name, save_to_json
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.optim import AdamW
from datasets import DatasetDict, load_metric, Dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader

class prompt_model:
    def __init__(self, model_path):
        model = BartForConditionalGeneration.from_pretrained(model_path, from_tf=True)
        
        # accelerator
        self.accelerator = Accelerator()

        # model, optimizer
        self.model = self.accelerator.prepare(model)
        self.tokenizer = BartTokenizer.from_pretrained(model_path, return_tensors='pt')        

        # Metric
        self.metric  = evaluate.load('sacrebleu', trust_remote_code=True)
        self.sacrebleu_metric = evaluate.load('sacrebleu', trust_remote_code=True)
        # self.rouge_metric = evaluate.load('rouge', trust_remote_code=True)

        # special token id 
        self.pad_token_id = model.config.pad_token_id
        self.decoder_start_token_id = model.config.decoder_start_token_id

    def training(
        self, 
        epochs,
        train_dataloader: DataLoader, 
        eval_dataloader: DataLoader, 
        num_warmup_steps=0,
        lr=2e-5,
        training_run_log=None,
        eval_run_log=None,
        out_dir=None,
        hub_id=''
    ):
        # Data
        train_dataloader, eval_dataloader, optimizer= self.accelerator.prepare(
            train_dataloader, 
            eval_dataloader,
            self.accelerator.prepare(
                AdamW(self.model.parameters(), lr=lr)
            )
        )

        # Learning rate scheduler
        lr_scheduler = get_scheduler(
            'linear',
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=epochs*len(train_dataloader),
        )
        process_bar = tqdm(range(epochs*len(train_dataloader)))

        # Start training
        for epoch in range(epochs):
            
            self.model.train()
            for batch in train_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                process_bar.update()

                if training_run_log:
                    training_run_log.log(
                        {
                            'loss': loss
                        }
                    )

            # Evaluation
            result = self.evaluate(
                eval_dataloader,
                eval_run_log=eval_run_log
            )
            print(f"epoch {epoch}, BLEU score: {result['score']:.2f}")

            # Save to disk 
            if out_dir:
                self.accelerator.wait_for_everyone()
                unwarped_model = self.accelerator.unwrap_model(self.model)
                unwarped_model.save_pretrained(out_dir, save_function=self.accelerator.save)
                if self.accelerator.is_main_process:
                    self.tokenizer.save_pretrained(out_dir)
                
                save_to_json(result, os.path.join(out_dir, 'log'))

            # Push to huggingface
            if hub_id:
                self.accelerator.wait_for_everyone()
                # Save model
                unwarped_model = self.accelerator.unwrap_model(self.model)
                unwarped_model.push_to_hub(hub_id)
        
        if training_run_log:
            training_run_log.finish()

    def evaluate(
        self, 
        eval_dataloader: DataLoader, 
        eval_run_log=None
    ):

        process_bar = tqdm(range(len(eval_dataloader)))

        for batch in eval_dataloader:
            self.model.eval()
            with torch.no_grad():
                generated_tokens = self.accelerator.unwrap_model(self.model).generate(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=128,
                    num_beams=4
                )

            generated_tokens = self.accelerator.pad_across_processes(
                generated_tokens,
                dim=1,
                pad_index=self.tokenizer.pad_token_id
            )

            labels = batch['labels']
            labels = self.accelerator.pad_across_processes(
                labels,
                dim=1,
                pad_index=-100
            )

            predictions_gathered = self.accelerator.gather(
                generated_tokens
            )

            labels_gathered = self.accelerator.gather(
                labels
            )

            decode_preds, decode_labels = self.postprocess(predictions_gathered, labels_gathered)

            self.metric.add_batch(predictions=decode_preds, references=decode_labels)
            sacrebleu_score = self.sacrebleu_metric.compute(predictions=decode_preds, references=decode_labels)
            # rouge_score = self.rouge_metric.compute(predictions=decode_preds, references=decode_labels)

            if eval_run_log:
                eval_run_log.log(
                    {
                        'blue_score': sacrebleu_score['score'],
                        # 'rouge1_precision': rouge_score['rouge1'].mid.precision,
                        # 'rouge2_recall':rouge_score['rouge1'].mid.recall,
                    }
                )

            process_bar.update(1)

        result = self.metric.compute()

        if eval_run_log:
            eval_run_log.finish()

        return result

    def inference(self, batch):
        batch = Dataset.from_dict(batch)

        def convert_to_feature_for_infer(example):
            input_encoding = self.tokenizer.batch_encode_plus(
                example['instruction'], 
                padding='max_length',
                max_length=128, 
                truncation=True
            )

            encodings = {
                'input_ids': input_encoding['input_ids'],
                'attention_mask': input_encoding['attention_mask'],
            }

            return encodings

        batch = batch.map(
            convert_to_feature_for_infer,
            batched=True,
            remove_columns=batch.column_names
        )

        self.model.eval()
        with torch.no_grad():
            generated_tokens = self.accelerator.unwrap_model(self.model).generate(
                torch.Tensor(batch['input_ids']).to(torch.int64).to(self.model.device), # batch['input_ids'], #
                attention_mask=torch.Tensor(batch['attention_mask']).to(torch.int64).to(self.model.device), # batch['attention_mask'],
                max_new_tokens=60,
                num_beams=2,
                no_repeat_ngram_size=2
            )

        generated_tokens = self.accelerator.pad_across_processes(
            generated_tokens,
            dim=1,
            pad_index=self.tokenizer.pad_token_id
        )

        predictions_gathered = self.accelerator.gather(
            generated_tokens
        )

        decode_preds, _ = self.postprocess(predictions_gathered)

        return decode_preds

    def postprocess(self, predictions, labels=None):
        predictions = predictions.cpu().numpy()
        decode_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decode_preds = [pred.strip() for pred in decode_predictions]

        decode_labels = None
        if labels != None:
            labels = labels.cpu().numpy()
            labels = np.where(labels != -100, labels, self.pad_token_id)
            decode_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            decode_labels = [[label.strip()] for label in decode_labels]

        return decode_preds, decode_labels