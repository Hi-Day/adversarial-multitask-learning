#!/bin/bash



for SOURCE in "amazon" "dslr" "webcam"
do
for TARGET in "amazon" "dslr" "webcam"
do

if [[ $SOURCE == $TARGET ]]; then
    continue
fi

# W_CATEGORICAL=1 
# W_ADVERSARIAL=1
# W_DOMAIN=1

    for W_CATEGORICAL in 0 0.1 0.3 0.6 0.01
        do
        for W_ADVERSARIAL in 0 0.1 0.3 0.6 0.01
            do
            for W_DOMAIN in 0 0.1 0.3 0.6 0.01
                do
                python main2.py $W_CATEGORICAL $W_ADVERSARIAL $W_DOMAIN $SOURCE $TARGET
                done
            done
        done



done
done