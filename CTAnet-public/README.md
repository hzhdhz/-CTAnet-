# Convolutional Transformer-inspired Autoencoder for Hyperspectral Anomaly Detection

This repository provides a [PyTorch](https://pytorch.org/) implementation of the *CTAnet* method presented in our paper
 ”Convolutional Transformer-inspired Autoencoder for Hyperspectral Anomaly Detection”.

how to test?
abu-airport-2 ----test
activate pytorch
python main_test.py abu-airport-2 abu-airport-2  ./Datasets_HSI/ 50 10 random 9 [2] False False False True full  ctanet_hz ./log/DeepSAD/abu-airport-2   ./data --load_model  ./log/DeepSAD/abu-airport-2/model.tar --ratio_known_outlier 0.01 --ratio_pollution 0.1 --lr 0.0001 --n_epochs 150 --lr_milestone 50 


how to train?
abu-airport-2
activate pytorch
python main.py abu-airport-2 abu-airport-2 ./Datasets_HSI/ 50 10 random 9 [2] False False False True full ctanet_hz ./log/DeepSAD/abu-airport-2 ./data --ratio_known_outlier 0.01 --ratio_pollution 0.1 --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 32 --weight_decay 0.5e-6 --ae_lr 0.0001 --ae_n_epochs 150 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class 0 --known_outlier_class 1 --n_known_outlier_classes 1



The code is written based on the DSAD (https://github.com/lukasruff/Deep-SAD-PyTorch).
