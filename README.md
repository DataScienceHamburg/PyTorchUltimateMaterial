# PytorchUltimate

This repo holds material for the Udemy course "PyTorch Ultimate"

You can find the course under this link.

## Environment Installation from yml file

We work with anaconda and use conda environments. You can replicate my environment by running:

conda env create -f pytorch.yml

## Environment Installation from scratch

If the installation from yml file fails, you can install the environment manually by running these commands.

conda create -n pytorch python=3.10
conda activate pytorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install ipykernel
conda install -c anaconda seaborn
conda install scikit-learn
conda install -c conda-forge detecto