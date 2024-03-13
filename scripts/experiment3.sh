python src/train.py \
    --config-name experiment2.yaml \
    -m \
    trainer.max_epochs=1200 \
    data.limit=-1 \
    data.train_val_test_split="[1.0,0,0]" \
    model.optimizer.lr=0.001,0.0001

# I just reuse the experiment2 config for easier experiment starting here.