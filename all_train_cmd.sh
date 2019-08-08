# !/bin/bash


python makelist.py
python cpsomefile.py
python createhd5_4channels.py low


cd ./01000
sh train_regnet_4c_W_low.sh
cd ../02000
sh train_regnet_4c_W_low.sh
cd ../03000
sh train_regnet_4c_W_low.sh
cd ../04000
sh train_regnet_4c_W_low.sh
cd ../05000
sh train_regnet_4c_W_low.sh
cd ../06000
sh train_regnet_4c_W_low.sh
cd ../07000
sh train_regnet_4c_W_low.sh
cd ../08000
sh train_regnet_4c_W_low.sh
cd ../09000
sh train_regnet_4c_W_low.sh
cd ../10000
sh train_regnet_4c_W_low.sh

cd ..
python saveres_4c_W_low.py

python createhd5_4channels.py high


cd ./01000
sh train_regnet_4c_W_high.sh
cd ../02000
sh train_regnet_4c_W_high.sh
cd ../03000
sh train_regnet_4c_W_high.sh
cd ../04000
sh train_regnet_4c_W_high.sh
cd ../05000
sh train_regnet_4c_W_high.sh
cd ../06000
sh train_regnet_4c_W_high.sh
cd ../07000
sh train_regnet_4c_W_high.sh
cd ../08000
sh train_regnet_4c_W_high.sh
cd ../09000
sh train_regnet_4c_W_high.sh
cd ../10000
sh train_regnet_4c_W_high.sh

cd ..
python saveres_4c_W_high.py
