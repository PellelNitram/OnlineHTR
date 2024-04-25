# Run short parameter sweep for carbune2020_xytn w/o scheduler.
# This is to start getting intuition for best parameters.

time python src/train.py \
    --config-name experiment_LitModule1.yaml \
    -m \
    trainer.max_epochs=1500 \
    data.limit=-1 \
    data.train_val_test_split="[0.8,0.2,0]" \
    data.transform="carbune2020_xytn" \
    model.dropout=0.0,0.25,0.5,0.75 \
    model.number_of_layers=2,3,4 \
    model.nodes_per_layer=64 \
    model.optimizer.lr=0.0001 \
    model.optimizer.weight_decay=0.0,1e-4 \
    tags="experiment12" \
    experiment_name="experiment12" \
    trial_name="trial1"

time python src/train.py \
    --config-name experiment_LitModule1.yaml \
    -m \
    trainer.max_epochs=1500 \
    data.limit=-1 \
    data.train_val_test_split="[0.8,0.2,0]" \
    data.transform="carbune2020_xytn" \
    model.dropout=0.0,0.25,0.5,0.75 \
    model.number_of_layers=2,3,4 \
    model.nodes_per_layer=64 \
    model.optimizer.lr=0.00005 \
    model.optimizer.weight_decay=0.0,1e-4 \
    tags="experiment12" \
    experiment_name="experiment12" \
    trial_name="trial2"