#!/bin/bash
clear;
for i in {1..3}
do
   python dqntest.py --test-num $i --rank $1;
done