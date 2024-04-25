# 1. Run speed test to see if avoiding re-computing string labels is faster
#   - Old version to run this script at: a18972f197f8ffae396874b4d7e184919d40b229
#   - New version to run this script at: 5a056f10c6009de5a34440e4da80a24c9720886a
# 2. Run speed test to see if removing text decoding, WER/CER computations and logging
#    thereof is faster
#   - Old version to run this script at: b464132e91e4f6ac644954347773b7e60754c6e8
#   - New version to run this script at: e1337475baea1e937237c6d5f4b6b59343e3cf10

# I run those two speed tests using this script here at the two corresponding old and
# new git commits manually.

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
    tags="experiment13" \
    experiment_name="experiment13" \
    trial_name="trial1"