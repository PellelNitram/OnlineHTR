# This script allows training of a model that is used for inference.

# Hyperparameters are chosen so that they are optimal within the scope
# of my hyperparameter sweep.

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
    tags="train_inference_model" \
    experiment_name="train_inference_model" \
    trial_name="final"
