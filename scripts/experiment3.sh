python src/train.py \
    --config-name experiment2.yaml \
    -m \
    trainer.max_epochs=1200 \
    data.limit=-1 \
    data.train_val_test_split="[1.0,0,0]" \
    model.optimizer.lr=0.001,0.0001

python src/train.py \
    --config-name experiment2.yaml \
    -m \
    data.batch_size=128,256 \
    trainer.max_epochs=1200 \
    data.limit=-1 \
    data.train_val_test_split="[1.0,0,0]" \
    model.optimizer.lr=0.001,0.0001

python src/train.py \
    --config-name experiment2.yaml \
    -m \
    data.batch_size=64 \
    trainer.max_epochs=3000 \
    data.limit=-1 \
    data.train_val_test_split="[1.0,0,0]" \
    model.optimizer.lr=0.0001

python src/train.py \
    --config-name experiment2.yaml \
    -m \
    data.batch_size=128 \
    trainer.max_epochs=3000 \
    data.limit=-1 \
    data.train_val_test_split="[1.0,0,0]" \
    model.optimizer.lr=0.001

# I just reuse the experiment2 config for easier experiment starting here.