# Run speed test to see if avoiding re-computing string labels is faster

# Old version to run this script at: a18972f197f8ffae396874b4d7e184919d40b229
# New version to run this script at: 5a056f10c6009de5a34440e4da80a24c9720886a

# I run those two experiments using this script here at the two git commits
# manually as it's easier.

time python src/train.py \
    --config-name experiment_LitModule1.yaml \
    -m \
    trainer.max_epochs=1000 \
    data.limit=-1 \
    data.train_val_test_split="[0.8,0.2,0]" \
    data.transform="carbune2020_xytn" \
    model.dropout=0.0 \
    model.number_of_layers=3 \
    model.nodes_per_layer=64 \
    model.optimizer.lr=0.0001 \
    model.optimizer.weight_decay=0.0 \
    tags="experiment13"