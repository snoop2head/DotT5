import multiprocessing

import numpy as np
import torch
import yaml
from easydict import EasyDict
from transformers import (
    T5Config,
    PreTrainedTokenizerFast,
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset, load_metric

from utils import postprocess_text, compute_metrics, preprocess_function

# run the script
if __name__ == "__main__":

    with open("config.yaml") as infile:
        SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
        CONFIG = EasyDict(SAVED_CFG["CFG"])
    CPU_COUNT = multiprocessing.cpu_count()

    # Load the metric, dataset and tokenizer
    metric = load_metric("sacrebleu")
    dset = load_dataset(
        "PoolC/AIHUB-parallel-ko-braille-short64", use_auth_token=True
    )  # or replace with 128 for more data
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "snoop2head/KoBrailleT5-small-v1"
    )
    tokenized_datasets = dset["train"].map(
        preprocess_function,
        batched=True,
        num_proc=CPU_COUNT,
        remove_columns=dset["train"].column_names,
    )
    tokenized_valid_datasets = dset["valid"].map(
        preprocess_function,
        batched=True,
        num_proc=CPU_COUNT,
        remove_columns=dset["valid"].column_names,
    )

    # Split the validation set in half (train: 0.8, validation: 0.1, test: 0.1)
    input_dset = tokenized_valid_datasets.train_test_split(test_size=0.5)
    input_dset["valid"] = input_dset["train"]
    del input_dset["train"]
    input_dset["train"] = tokenized_datasets

    if CONFIG.model_name == "scratch":
        # define model components to train from scratch
        t5 = T5ForConditionalGeneration(T5Config(**CONFIG.t5_config))
    elif CONFIG.model_name == "KETI-AIR/ke-t5-small":
        # Get the pretrained model on Korean-English https://github.com/AIRC-KETI/ke-t5
        model_name = "KETI-AIR/ke-t5-small"
        t5 = T5ForConditionalGeneration.from_pretrained(model_name)

        # override keti t5 config for custom use https://huggingface.co/KETI-AIR/ke-t5-small/blob/main/config.json
        t5.config.vocab_size = tokenizer.vocab_size
        t5.config.dropout_rate = CONFIG.t5_config.dropout_rate
        t5.config.pad_token_id = tokenizer.pad_token_id
        t5.config.eos_token_id = tokenizer.eos_token_id
        t5.config.decoder_start_token_id = tokenizer.pad_token_id

        # resize token embedding
        t5.resize_token_embeddings(tokenizer.vocab_size)

    # prepare trainer
    t5.train()
    training_args = Seq2SeqTrainingArguments(
        output_dir=CONFIG.train_config.output_dir,
        learning_rate=CONFIG.train_config.learning_rate,
        weight_decay=CONFIG.train_config.weight_decay,
        per_device_train_batch_size=CONFIG.train_config.per_device_train_batch_size,
        per_device_eval_batch_size=CONFIG.train_config.per_device_eval_batch_size,
        evaluation_strategy="steps",
        # eval_steps=CFG.eval_steps,
        save_steps=CONFIG.train_config.save_steps,
        num_train_epochs=CONFIG.train_config.num_train_epochs,
        save_total_limit=CONFIG.train_config.save_total_limit,
        predict_with_generate=CONFIG.train_config.predict_with_generate,
        fp16=CONFIG.train_config.fp16,
        gradient_accumulation_steps=CONFIG.train_config.gradient_accumulation_steps,
        logging_steps=CONFIG.train_config.logging_steps,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=t5)

    trainer = Seq2SeqTrainer(
        t5,
        training_args,
        train_dataset=input_dset["train"],
        eval_dataset=input_dset["valid"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.predict(test_dataset=input_dset["test"])
