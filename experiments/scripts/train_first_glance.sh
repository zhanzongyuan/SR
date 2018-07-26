#!/bin/bash
# Initial logs
rm -rf ./experiments/logs/train_first_glance

################## Train arguments ###############
# Train epoch
epoch=100
# Learning rate
lr=0.001
# Weight decay
weight_decay=0.0001
# Batch size for train
batch_size=8
# momentum
momentum=0.9
# Number of classes
num=6
# Worker number
worker=7

################## Dataset arguments ###############
# Path to Images
ImagePath="data/PISC/image"
# Path to object boxes
ObjectsPath="data/objects/PISC_objects/"
# Path to test list
TrainList="data/list/PISC_fine_level_train.txt"
# Path to test list
TestList="data/list/PISC_fine_level_test.txt"


################## Record arguments ###############
# Path to save scores
ResultPath="experiments/logs/train_first_glance"
# Print frequence
print_freq=100


################## Model arguments ###############
# Path to save model
ModelPath="models/First_Glance_checkpoint.pth"
# Path to load model
CheckpointPath=""

python ./tools/train_first_glance.py \
    $ImagePath \
    $ObjectsPath \
    $TrainList \
    $TestList \
    -n $num \
    -b $batch_size \
    --lr $lr \
    -m $momentum \
    --wd $weight_decay \
    -e $epoch \
    -j $worker \
    --print-freq $print_freq \
    --result-path $ResultPath \
    --checkpoint $CheckpointPath \
    --weights $ModelPath

