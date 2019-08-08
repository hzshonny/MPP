#!/bin/bash
LOG=./log/INFO-4c-W-low-`date  +%Y-%m-%d-%H-%M-%S`.log
/home/caffe-ssd/build/tools/caffe train --solver=./solve_4c_W_low.prototxt 2>&1 | tee $LOG
