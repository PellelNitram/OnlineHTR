# Perform experiment5 and experiment6 for fewer epochs
# but with validation enabled.

time python src/train.py \
    --config-name experiment2.yaml \
    -m \
    data.batch_size=64 \
    trainer.max_epochs=2000 \
    data.limit=-1 \
    data.train_val_test_split="[0.8,0.2,0]" \
    model.optimizer.lr=0.0001 \
    data.pin_memory=True \
    data.num_workers=4 \
    data.transform="carbune2020_xytn","carbune2020_xyn" \
    tags="experiment7"