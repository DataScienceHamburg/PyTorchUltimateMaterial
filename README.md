# PytorchUltimate

This repo holds material for the Udemy course _**PyTorch Ultimate: From Basics to Cutting-Edge**_. You can find the course under [this link](https://www.udemy.com/course/pytorch-ultimate/).

## Environment Installation from yml file

We work with [Anaconda](https://www.anaconda.com/) and use [conda environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#). You can replicate my environment by running:

```
C:\...> conda env create -f pytorch.yml
```

## Environment Installation from scratch

If the installation from yml file fails, you can install the environment manually by running these commands:

```
C:\...> conda create -n pytorch python=3.10
C:\...> conda activate pytorch
(pytorch) C:\...> conda install pytorch torchvision torchaudio cpuonly -c pytorch
(pytorch) C:\...> conda install ipykernel
(pytorch) C:\...> conda install -c anaconda seaborn
(pytorch) C:\...> conda install scikit-learn
(pytorch) C:\...> $ conda install -c conda-forge detecto
```