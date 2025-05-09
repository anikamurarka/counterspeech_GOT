conda activate kg-counter-narratives

python format_dataset.py

python dialogue_to_got.py

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 \
  train_model.py \
  --data_root data --dataset DIALOCONAN --got_root got/ \
  --output_dir experiments/DIALOCONAN \
  --model declare-lab/flan-alpaca-base \
  --epoch 20 --lr 5e-5 --bs 8 --eval_bs 16 \
  --input_len 512 --output_len 256 \
  --bf16

CUDA_VISIBLE_DEVICES=0 python \
  evaluate_model.py \
  --data_root data --dataset DIALOCONAN --got_root got/ \
  --output_dir experiments/DIALOCONAN \
  --model declare-lab/flan-alpaca-base \
  --epoch 20 --lr 5e-5 --bs 8 --eval_bs 16 \
  --input_len 512 --output_len 256 \
  --bf16 \
  --evaluate_dir experiments/DIALOCONAN/declare-lab-flan-alpaca-base_lr5e-05_bs8_op256_ep20_2025-05-09-14-59