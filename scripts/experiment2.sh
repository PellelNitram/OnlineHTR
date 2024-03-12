for VALUE in 4 8 16; do

    python src/train.py \
        --config-name experiment2.yaml \
        -m \
        trainer.max_epochs=30000 \
        data.limit=${VALUE} \
        data.train_val_test_split="[${VALUE},0,0]"
        # See this answer on using a tuple with Hydra's CLI: https://stackoverflow.com/a/71213905

done
