python src/train.py \
    --config-name experiment1.yaml \
    -m \
    trainer.max_epochs=10000,20000,30000 \
    trainer.accelerator=gpu,cpu