python main.py \
--cuda "0, 1, 2, 3" \
--batch_size 8 \
--train_epochs 2 \
--lora_rank 64 \
--arch te \
--precision baseline \
--model Qwen/Qwen3-8B