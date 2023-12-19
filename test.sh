#! /bin/bash
echo "Start running..."

seq_len=(32 64 128 256 512 1024)
feature_len=(32 64 128 256 512 1024 2048)
kd=(0 1)

for ((k=0; k<2; k++))
do
    for ((i=0; i<6; i++))
    do
        for ((j=0; j<7; j++))
        do
            echo "seq_len is ${seq_len[$i]}, feature_size is ${feature_len[$j]}, kd is ${kd[$k]}"
            ./bin/attention ${seq_len[$i]} ${feature_len[$j]} ${kd[$k]}  
        done
    done
done