# Run data in format that is available in Xournal(++)

time python src/train.py \
    --config-name experiment_LitModule1.yaml \
    -m \
    trainer.max_epochs=3000 \
    data.limit=-1 \
    data.train_val_test_split="[0.8,0.2,0]" \
    data.transform="iam_SimpleNormalise_xyn" \
    tags="experiment11"