# Perform training w/ callbacks to use them for inference subsequently

time python src/train.py \
    --config-name experiment_w_callbacks.yaml \
    -m \
    data.batch_size=64 \
    trainer.max_epochs=30 \
    data.limit=200 \
    data.train_val_test_split="[0.8,0.2,0]" \
    model.optimizer.lr=0.0001 \
    data.pin_memory=True \
    data.num_workers=4 \
    data.transform="carbune2020_xyn" \
    callbacks.model_checkpoint.every_n_epochs=10 \
    tags="experiment8"