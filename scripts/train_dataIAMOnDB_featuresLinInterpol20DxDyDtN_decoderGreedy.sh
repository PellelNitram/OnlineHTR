# This script allows training of the model called
# "dataIAMOnDB_featuresLinInterpol20DxDyDtN_decoderGreedy".

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
    tags="dataIAMOnDB_featuresLinInterpol20DxDyDtN_decoderGreedy" \
    experiment_name="dataIAMOnDB_featuresLinInterpol20DxDyDtN_decoderGreedy" \
    trial_name="final"
