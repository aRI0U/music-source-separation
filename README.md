# Improving DNN-based Music Source Separation using Phase Features

This repository contains the implementation of [Improving DNN-based Music Source Separation using Phase Features](http://arxiv.org/abs/1807.02710), from Muth *et al*.

## Preparation

1. Clone this repository

  ```sh
  git clone https://github.com/aRI0U/music-source-separation
  ```

2. Install all Python dependencies

  ```sh
  cd music-source-separation
  pip install -r requirements.txt
  ```

3. Download the MUSDB18 dataset. Access on request [here](https://zenodo.org/record/1117372).

4. Put the archive containing MUSDB18 in a folder named `datasets` and extract it with the provided script.

  ```sh
  mkdir datasets
  mv /path/to/musdb18.zip datasets
  ./utils/setup_musdb.sh
  ```

5. The architecture of the model contains a bias and scale layers which are initialized respectively with the mean and standard deviation per frequency bin over the training set. In order to compute these statistics, run the following:

  ```sh
  python3 utils/compute_stats.py
  ```

  ***Warning:*** *This script computes the STFT of the tracks in the dataset. The parameters of the STFT given by default are the same as the default options of `train.py`. If you want to change the STFT options in `train.py`, you will have to change the parameters of this `compute_stats.py` consequently.*


## Usage

- Train a model

  ```sh
  python3 train.py
  ```

  A lot of options can be configured through command-line arguments. Type `python3 train.py --help` for an exhaustive detailed list of those options.


### Folder structure

When you run `train.py`, a new folder is created in folder `runs`. You can specify the name of the experiment with option `--name`. Otherwise a timestamp will be used.

Each folder of `runs` contains three subfolders:

- A `checkpoints` directory containing the saved models along training.
- A `logs` directory containing the Tensorboard logs. One can thus track the evolution of the loss along the training by running on a separate terminal

  ```sh
  tensorboard --logdir runs
  ```
  Samples of separed audio from the validation set can also be listened along training with Tensorboard.

- A `results` directory containing signals separed by the model.

Since saving checkpoints and audio extracts is a quite slow operation, it is done only every 5 epochs. You can change this behaviour with option `--save_freq`.


## Citation

This work implements the model presented in [Improving DNN-based Music Source Separation using Phase Features](http://arxiv.org/abs/1807.02710). To our knowledge, no official implementation of this paper is available.

To cite the original paper:
```
@article{Muth18,
  archivePrefix = {arXiv},
  arxivId       = {1807.02710},
  author        = {Muth, Joachim and
                   Uhlich, Stefan and
                   Perraudin, Nathanael and
                   Kemp, Thomas and
                   Cardinaux, Fabien and
                   Mitsufuji, Yuki},
  eprint        = {1807.02710},
  title         = {{Improving DNN-based Music Source Separation using Phase Features}},
  url           = {http://arxiv.org/abs/1807.02710},
  year          = {2018}
}
```

A part of the code of this repository (the major part of `data.py` and maybe a few lines somewhere else) come from [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch).

For all experiments, the model of this repository uses the [MUSDB18](https://sigsep.github.io/datasets/musdb.html#sisec-2018-evaluation-campaign) dataset.

```
@misc{musdb18,
  author       = {Rafii, Zafar and
                  Liutkus, Antoine and
                  Fabian-Robert St{\"o}ter and
                  Mimilakis, Stylianos Ioannis and
                  Bittner, Rachel},
  title        = {The {MUSDB18} corpus for music separation},
  month        = dec,
  year         = 2017,
  doi          = {10.5281/zenodo.1117372},
  url          = {https://doi.org/10.5281/zenodo.1117372}
}
```
