#!/bin/bash



for W_CATEGORICAL in 0.6 0.01
    do
    for W_ADVERSARIAL in 0 0.1 0.3 0.6 0.01
        do
        for W_DOMAIN in 0 0.1 0.3 0.6 0.01
            do
            python main.py $W_CATEGORICAL $W_ADVERSARIAL $W_DOMAIN
            done
        done
    done
