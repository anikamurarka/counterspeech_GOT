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

from eval_utils.utils_evaluate import get_scores_conan
from train_utils.dataset import DialoconanDatasetWithGraph, DialoconanDatasetNoGraph
from train_utils.model import T5GenerationWithGraph, T5GenerationNoGraph
from train_utils.utils_data import load_data_std_dialoconan
from train_utils.utils_prompt import postprocess_text

console = Console(record=True)


os.environ["WANDB_PROJECT"] = "KG_CN_Generation"


def evaluate_model(args):
    configure_random_seeds(args)

    # Create output directories
    print('====Creating output directories====')
    results_dir = args.evaluate_dir
    print(results_dir)

    # Initialize tokenizer
    print(f'====Initializing tokenizer====')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<s>']})
    vocabulary = tokenizer.get_vocab()
    special_token_id = vocabulary["<s>"]
    data_collator = DataCollatorForSeq2Seq(tokenizer)

    # Load datasets
    print('====Loading datasets====')
    train_samples, dev_samples, test_samples = load_data_std_dialoconan(args, console=console)

    # Create appropriate dataset objects based on configuration
    if args.text_only:
        train_dataset = DialoconanDatasetNoGraph(train_samples, tokenizer, args.input_len, args.output_len,
                                                args.exclude_context)
        eval_dataset = DialoconanDatasetNoGraph(dev_samples, tokenizer, args.input_len, args.output_len,
                                               args.exclude_context)
        test_dataset = DialoconanDatasetNoGraph(test_samples, tokenizer, args.input_len, args.output_len,
                                               args.exclude_context)
    else:
        train_dataset = DialoconanDatasetWithGraph(train_samples, "train", tokenizer, args.input_len, args.output_len,
                                                  args)
        eval_dataset = DialoconanDatasetWithGraph(dev_samples, "dev", tokenizer, args.input_len, args.output_len,
                                                 args)
        test_dataset = DialoconanDatasetWithGraph(test_samples, "test", tokenizer, args.input_len, args.output_len,
                                                 args)

    # Load pretrained model
    args.model = args.evaluate_dir
    print(f'====Loading model from: {args.model} ====')
    if args.text_only:
        model = T5GenerationNoGraph.from_pretrained(args.model, s_token_id=special_token_id)
    else:
        model = T5GenerationWithGraph.from_pretrained(args.model, s_token_id=special_token_id)
    model.resize_token_embeddings(len(tokenizer))
    print("Total model parameters:", model.num_parameters())

    # Initialize evaluation metric
    rouge_metric = evaluate.load("rouge")

    def calculate_rouge_metrics(eval_preds):
        predictions, references = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        processed_preds = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(processed_preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_refs = tokenizer.batch_decode(references, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        formatted_preds, formatted_refs = postprocess_text(decoded_preds, decoded_refs)

        metrics_result = rouge_metric.compute(predictions=formatted_preds, references=formatted_refs, use_stemmer=True)
        metrics_result = {k: round(v * 100, 4) for k, v in metrics_result.items()}
        pred_lengths = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in processed_preds]
        metrics_result["gen_len"] = np.mean(pred_lengths)
        return metrics_result

    # Configure evaluation settings
    print('====Configuring evaluation parameters====')
    eval_args = Seq2SeqTrainingArguments(results_dir,
                                         do_train=False,
                                         do_eval=True,
                                         evaluation_strategy="steps",
                                         logging_strategy="steps",
                                         logging_steps=20,
                                         save_strategy="steps",
                                         eval_steps=20,
                                         save_steps=100,
                                         save_total_limit=2,
                                         learning_rate=args.lr,
                                         eval_accumulation_steps=args.eval_acc,
                                         per_device_train_batch_size=args.bs,
                                         per_device_eval_batch_size=args.eval_bs,
                                         weight_decay=args.weight_decay,
                                         num_train_epochs=args.epoch,
                                         metric_for_best_model="rougeL",
                                         predict_with_generate=True,
                                         generation_max_length=args.output_len,
                                         load_best_model_at_end=True,
                                         report_to="wandb",
                                         )

    print('====Initializing trainer====')
    trainer = Seq2SeqTrainer(model=model,
                             args=eval_args,
                             train_dataset=train_dataset,
                             eval_dataset=eval_dataset,
                             data_collator=data_collator,
                             tokenizer=tokenizer,
                             compute_metrics=calculate_rouge_metrics,
                             preprocess_logits_for_metrics=None
                             )

    def generate_model_outputs(dataset):
        prediction_results = trainer.predict(test_dataset=dataset, max_length=args.output_len)
        predictions, references = prediction_results.predictions, prediction_results.label_ids
        processed_preds = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(processed_preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_refs = tokenizer.batch_decode(references, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        cleaned_preds = [pred.strip() for pred in decoded_preds]

        return cleaned_preds, decoded_refs

    # Run prediction and evaluation
    print('====Generating predictions====')
    predictions, references = generate_model_outputs(test_dataset)
    evaluation_scores, results_df = get_scores_conan(predictions, references, train_samples)
    output_results = {"scores": evaluation_scores,
                      "preds": predictions,
                      "labels": references}

    # Save evaluation results
    output_file_path = os.path.join(results_dir, "predictions_ans_test.json")
    with open(output_file_path, "w") as output_file:
        output_file.write(json.dumps(output_results, indent=4))

    results_df.to_csv(os.path.join(results_dir, "predictions_ans_test.csv"))


def configure_random_seeds(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./../data')
    parser.add_argument('--dataset', type=str, default='DIALOCONAN')
    parser.add_argument('--got_root', type=str, default='got/')
    parser.add_argument('--output_dir', type=str, default='./../experiments/DIALOCONAN')
    parser.add_argument('--model', type=str, default='declare-lab/flan-alpaca-base')
    parser.add_argument('--text_only', action='store_true', help='remove graphs')
    parser.add_argument('--exclude_context', action='store_true', help='remove dialogue history from the prompt')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--eval_bs', type=int, default=4)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=64)
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--evaluate_dir', type=str,
                        default="./../experiments/DIALOCONAN/declare-lab-flan-alpaca-base_lr5e-05_bs32_op256_ep50_useGTrue_2024-06-01-04-17/checkpoint-3500",
                        help='the directory of model for evaluation')
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

    # Create training progress logger
    training_logger = Table(Column("Epoch", justify="center"),
                            Column("Steps", justify="center"),
                            Column("Loss", justify="center"),
                            title="Training Status",
                            pad_edge=False,
                            box=box.ASCII)

    args = parse_args()

    evaluate_model(args)


if __name__ == '__main__':
    main()