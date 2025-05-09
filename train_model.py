import argparse
import json
import os
import random

import evaluate
import numpy as np
import torch
from rich import box
from rich.console import Console
from rich.table import Column, Table
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from train_utils.dataset import DialoconanDatasetWithGraph
from train_utils.model import T5GenerationWithGraph
from train_utils.utils_data import load_data_std_dialoconan, mk_dir, make_save_directory
from train_utils.utils_prompt import postprocess_text

console = Console(record=True)
os.environ["WANDB_PROJECT"] = "KG_CN_Generation"


def train_t5_model(config):
    setup_random_seeds(config)

    # Create output directories
    print('====Creating directories====')
    output_path = make_save_directory(config)
    mk_dir(config.output_dir)

    # Initialize tokenizer
    print(f'====Initializing tokenizer====')
    token_processor = AutoTokenizer.from_pretrained(config.model)
    token_processor.add_special_tokens({'additional_special_tokens': ['<s>']})
    vocabulary = token_processor.get_vocab()
    special_token_id = vocabulary["<s>"]
    data_collector = DataCollatorForSeq2Seq(token_processor)

    # Prepare datasets
    print('====Preparing dataset====')
    train_examples, dev_examples, test_examples = load_data_std_dialoconan(config, console=console)
    training_dataset = DialoconanDatasetWithGraph(train_examples, "train", token_processor, config.input_len, config.output_len, config)
    validation_dataset = DialoconanDatasetWithGraph(dev_examples, "dev", token_processor, config.input_len, config.output_len, config)

    # Initialize model
    print(f'====Initializing model: {config.model} ====')
    neural_model = T5GenerationWithGraph.from_pretrained(config.model, s_token_id=special_token_id)
    neural_model.resize_token_embeddings(len(token_processor))
    print("model parameters: ", neural_model.num_parameters())

    # Setup evaluation metric
    evaluation_metric = evaluate.load("rouge")

    def calculate_rouge_metrics(eval_predictions):
        predictions, references = eval_predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        processed_preds = np.where(predictions != -100, predictions, token_processor.pad_token_id)
        decoded_preds = token_processor.batch_decode(processed_preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_refs = token_processor.batch_decode(references, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        final_preds, final_refs = postprocess_text(decoded_preds, decoded_refs)

        scores = evaluation_metric.compute(predictions=final_preds, references=final_refs, use_stemmer=True)
        scores = {k: round(v * 100, 4) for k, v in scores.items()}
        pred_lengths = [np.count_nonzero(pred != token_processor.pad_token_id) for pred in processed_preds]
        scores["gen_len"] = np.mean(pred_lengths)
        return scores

    # Configure training arguments
    print('====Configuring training arguments====')
    train_config = Seq2SeqTrainingArguments(output_path,
                                         do_train=True,
                                         do_eval=True,
                                         eval_strategy="steps",
                                         logging_strategy="steps",
                                         logging_steps=10,
                                         save_strategy="steps",
                                         eval_steps=500,
                                         save_steps=500,
                                         save_total_limit=2,
                                         learning_rate=config.lr,
                                         eval_accumulation_steps=config.eval_acc,
                                         per_device_train_batch_size=config.bs,
                                         per_device_eval_batch_size=config.eval_bs,
                                         weight_decay=config.weight_decay,
                                         num_train_epochs=config.epoch,
                                         metric_for_best_model="rougeL",
                                         predict_with_generate=True,
                                         generation_max_length=config.output_len,
                                         load_best_model_at_end=True,
                                         report_to="none",
                                         bf16=config.bf16
                                         )

    print('====Setting up trainer====')
    model_trainer = Seq2SeqTrainer(model=neural_model,
                                args=train_config,
                                train_dataset=training_dataset,
                                eval_dataset=validation_dataset,
                                data_collator=data_collector,
                                tokenizer=token_processor,
                                compute_metrics=calculate_rouge_metrics,
                                preprocess_logits_for_metrics=None
                                )

    # Start training
    print('====Starting training====')
    model_trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    model_trainer.save_model(output_path)

    # Run evaluation
    print('====Running evaluation====')
    performance_metrics = model_trainer.evaluate(eval_dataset=validation_dataset, max_length=config.output_len)
    model_trainer.log_metrics("eval", performance_metrics)
    model_trainer.save_metrics("eval", performance_metrics)

    def create_model_outputs(dataset):
        results = model_trainer.predict(test_dataset=dataset, max_length=config.output_len)
        model_outputs, target_outputs = results.predictions, results.label_ids
        model_outputs = np.where(model_outputs != -100, model_outputs, token_processor.pad_token_id)
        decoded_outputs = token_processor.batch_decode(model_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_targets = token_processor.batch_decode(target_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        cleaned_outputs = [output.strip() for output in decoded_outputs]

        return cleaned_outputs, decoded_targets

    # Generate and save predictions
    print('====Generating predictions====')
    torch.cuda.empty_cache()
    if model_trainer.is_world_process_zero():
        generated_outputs, reference_outputs = create_model_outputs(validation_dataset)
        results_data = {"preds": generated_outputs,
                       "labels": reference_outputs}

        # Save prediction results
        prediction_file_path = os.path.join(output_path, "predictions_ans_eval.json")
        with open(prediction_file_path, "w") as writer:
            writer.write(json.dumps(results_data, indent=4))


def setup_random_seeds(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./../data')
    parser.add_argument('--dataset', type=str, default='DIALOCONAN')
    parser.add_argument('--got_root', type=str, default='got/')
    parser.add_argument('--output_dir', type=str, default='./../experiments')
    parser.add_argument('--model', type=str, default='declare-lab/flan-alpaca-base')
    parser.add_argument('--exclude_context', action='store_true', help='remove dialogue history from the prompt')
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--eval_bs', type=int, default=4)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='resume from checkpoint')
    parser.add_argument('--eval_strategy', type=str, default="steps", help='evaluation strategy',
                        choices=['steps', 'epoch'])
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')
    parser.add_argument('--bf16', action='store_true', help='use bf16 dtype')
    args = parser.parse_args()

    print("args", args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    return args


def main():
    print(f"\n\n\nCUDA AVAILABLE? {torch.cuda.is_available()}\n\n\n")

    # Setup training progress logger
    training_progress = Table(Column("Epoch", justify="center"),
                            Column("Steps", justify="center"),
                            Column("Loss", justify="center"),
                            title="Training Status",
                            pad_edge=False,
                            box=box.ASCII)

    args = parse_args()

    train_t5_model(args)


if __name__ == '__main__':
    main()
