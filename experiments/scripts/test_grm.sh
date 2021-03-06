#!/bin/bash

# Path to Images
ImagePath="data/PISC/image"
# Path to object boxes
ObjectsPath="data/objects/PISC_objects/"
# Path to test list
TestList="data/list/PISC_fine_level_test.txt"
# Path to adjacency matrix
AdjMatrix="data/adjacencyMatrix/PISC_fine_level_matrix.npy"
# Number of classes
num=6
# Path to save scores
ResultPath="experiments/logs/test"

# Path to model
ModelPath="models/PISC_fine_level.pth.tar"

CUDA_VISIBLE_DEVICES=0 python ./tools/test_grm.py \
    $ImagePath \
    $ObjectsPath \
    $TestList \
    --weights "" \
    --adjacency-matrix $AdjMatrix \
    -n $num -b 4 \
    --print-freq 100 \
    --write-out --result-path $ResultPath

