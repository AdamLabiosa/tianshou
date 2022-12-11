#!/bin/bash
clear;
rm -rf outputs_rank{$1}

for i in {1..3}
do
   python dqntest.py --test-num $i --rank $1;
done