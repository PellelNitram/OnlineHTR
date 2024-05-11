# Experiment to understand how useful a scheduler is.
# Does it improve validation loss?

time python src/train.py \
    --config-name experiment_LitModule1_w_scheduler.yaml \
    -m \
    trainer.max_epochs=100 \
    data.limit=10 \
    data.train_val_test_split="[0.8,0.2,0]" \
    data.transform="carbune2020_xytn" \
    model.dropout=0.0 \
    model.number_of_layers=3 \
    model.nodes_per_layer=64 \
    model.optimizer.lr=0.0001 \
    model.optimizer.weight_decay=0.0 \
    model.scheduler.patience=0,1,5 \
    model.scheduler.factor=0.1,0.5 \
    tags="experiment14" \
    experiment_name="experiment14" \
    trial_name="trial1_first_test"