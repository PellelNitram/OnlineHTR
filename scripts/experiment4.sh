# This is just a quick speed test

python src/train.py \
    --config-name experiment2.yaml \
    -m \
    data.batch_size=64,128,256 \
    trainer.max_epochs=200 \
    data.limit=-1 \
    data.train_val_test_split="[1.0,0,0]" \
    model.optimizer.lr=0.0001 \
    data.pin_memory=True,False \
    data.num_workers=0,4,8 \
    tags="experiment4"

# First default outpath, then tweak it, maybe?
# this is w/ small data.limit -> increase later!