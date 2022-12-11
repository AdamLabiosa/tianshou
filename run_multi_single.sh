#!/bin/bash
clear;
rm -rf outputs_single

for i in {1..10}
do
   python dqntest.py --test-num $i --rank 0 --dist 0;
done