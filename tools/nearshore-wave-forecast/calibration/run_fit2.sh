#!/bin/bash
# usage: run_fit2.sh job -> stage1 (strip y100-250), stage2 (y250-520), stage3 (y700-1000), full-domain given-data check
cd /Users/dev/.tb-graders/k-ocean/hidden
j=$1
SIM_Y0=100 SIM_Y1=250 ../venv/bin/python -W ignore fit2.py $j 1 >> fits2/$j.log 2>&1
SIM_Y0=250 SIM_Y1=520 ../venv/bin/python -W ignore fit2.py $j 2 >> fits2/$j.log 2>&1
SIM_Y0=700 SIM_Y1=1000 ../venv/bin/python -W ignore fit2.py $j 3 >> fits2/$j.log 2>&1
../venv/bin/python -W ignore fit2.py $j 0 >> fits2/$j.log 2>&1
echo DONE >> fits2/$j.log
