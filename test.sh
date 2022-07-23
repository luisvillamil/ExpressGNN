#!/bin/bash                                                                     

for ((i = 0; i <= 12; i++))
do
    python -m main.train -data_root data/fb15k-237\
    -rule_filename cleaned_rules_weight_larger_than_0.9.txt\
    -slice_dim 16 -batchsize $((i*16)) -use_gcn 1 -num_hops 1 -embedding_size 128\
    -gcn_free_size 127 -patience 20 -lr_decay_patience 100 -entropy_temp 1\
    -load_method 1 -exp_folder exp -exp_name freebase -device cuda > output/test_$i.txt
done