python train.py \
  --train_data './valid_debertav3.json' \
  --valid_data './valid_debertav3.json' \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 2e-6 \
  --lr_end 8e-7 \
  --warmup_ratio 0.1 \
  --num_train_epochs 2 \
  --gradient_accumulation_steps 8 \
  --logging_steps 5 \
  --eval_steps 20 \
  --save_steps 20 \
  --weight_decay 1e-4 \
  --MAX_INPUT 384 \
  --MODEL 'microsoft/deberta-v3-large' \
  --VER 5 \
  --dropout_rate 0 \
  --awp_lr 0.1 \
  --awp_eps 1e-4 \
  --awp_start_epoch 0.5 \
  --label_smoothing_factor 0.1 