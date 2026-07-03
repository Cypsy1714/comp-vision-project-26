# cnn options

everything is in the cnn section of config.yaml, no code changes needed.

train with `uv run python src/cnn.py --train`, that saves the weights to output/models.
score with `uv run python src/cnn.py`, that reads output/views and writes the .npy score files to output/scores.

the general options: labels points to the training csv, checkpoints is where weights go.
train_on decides what the models train on, crops means the clean crops, views means all
the filtered views (so the filters work like augmentation). val_split is how much data we
hold out for validation, the best epoch by macro f1 is the one that gets saved.
batch_size, label_smoothing, warmup_epochs and weight_decay are the usual training knobs,
weighted_loss turns on class weights if the classes are imbalanced.

models is a list. one entry = one model, two entries = both run side by side and their
scores land in the same file for the meta part. score_layout controls that file: stacked
gives one (models*K, classes) array which works with the current meta.py, 3d gives
(models, K, classes) if the meta part wants it separated.

per model: name is just the checkpoint filename. type timm gives a real backbone like
efficientnet_b0 (set backbone for the name and pretrained true/false for imagenet or
random init). type custom is our own small cnn, channels sets the conv block sizes and
hidden the dense layers. head and full are the two training phases, in head the base is
frozen and only the head trains, in full everything trains. set full epochs to 0 to keep
the imagenet base locked completely.

stuff worth testing for the report: pretrained true vs false on the same backbone,
locked base (full epochs 0) vs full finetuning, the timm backbone vs our custom cnn,
training on crops vs views, and one model vs two models with the meta classifier
deciding which model/view combo to trust.
