# Digital Image Processing: Final Project
## Tom Dana and Fayrouz Azem

## Paper: [SNIPS: Solving Noisy Inverse Problems Stochastically](http://arxiv.org/abs/2105.14951) by Bahjat Kawar, Gregory Vaksman, and Michael Elad

This repo is base on [snips_torch](https://github.com/bahjat-kawar/snips_torch).

## Running Experiments

### Dependencies

Run the following conda line to install all necessary python packages for our code and set up the snips environment.

```bash
conda create --name <env> --file requirements.txt
```

### Project structure

`main.py` is the file that you should run for sampling. Execute ```python main.py --help``` to get its usage description:

```
usage: main.py [-h] [--config CONFIG] [--seed SEED] [--exp EXP] [--doc DOC] [--comment COMMENT] [--verbose VERBOSE] [-i IMAGE_FOLDER] [-n NUM_VARIATIONS] [-s SIGMA_0] [--sp_amount AMOUNT] [-d DEG] [--noise_type NOISE]
               [--num_samples NUM_SAMPLES] [--overwrite]

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the config file
  --seed SEED           Random seed
  --exp EXP             Path for saving running related data.
  --doc DOC             A string for documentation purpose. Will be the name of the log folder
  --comment COMMENT     A string for experiment comment
  --verbose VERBOSE     Verbose level: info (default) | debug | warning | critical
  -i IMAGE_FOLDER, --image_folder IMAGE_FOLDER
                        The folder name of samples
  -n NUM_VARIATIONS, --num_variations NUM_VARIATIONS
                        Number of variations to produce for each sample
  -s SIGMA_0, --sigma_0 SIGMA_0
                        Noise std to add to observation (used in `noise_type=[gaussian | speckle]`). Default: 0.1
  --sp_amount AMOUNT    Probability of each pixel to become 1 or 0 (used in `noise_type=salt_and_pepper`). Default: 0.05
  -d DEG, --degradation DEG
                        Degradation: inp | deblur_uni | deblur_gauss | sr2 | sr4 (default) | cs4 | cs8 | cs16
  --noise_type NOISE    Noise type: gaussian (default) | poisson | salt_and_pepper | speckle
                        Gaussian:        output = input + gauss_noise(std=sigma_0)
                        Poisson:         output = poisson(input * WHITE_LEVEL) / WHITE_LEVEL
                        Salt and pepper: each pixel is converted to 0/1 with probability sp_amount
                        Speckle:         output = input + input * gauss_noise(std=sigma_0)
  --num_samples NUM_SAMPLES
                        Number of samples to generate. Default value is in the config file (sampling.batch_size)
  --overwrite           Overwrite image folder if already exists
```

Configuration files are in `config/`. You don't need to include the prefix `config/` when specifying  `--config` . All files generated when running the code is under the directory specified by `--exp`. They are structured as:

```bash
<exp> # a folder named by the argument `--exp` given to main.py
├── datasets # all dataset files
│   ├── celeba # all CelebA files
│   └── lsun # all LSUN files
├── logs # contains models checkpoints
│   └── <doc> # a folder named by the argument `--doc` specified to main.py
│      └── checkpoint_x.pth # the checkpoint file will be used
├── image_samples # contains generated samples
│   └── <i>
│       ├── stochastic_variation.png # samples generated from checkpoint_x.pth, including original, degraded, mean, and std   
│       ├── results.pt # the pytorch tensor corresponding to stochastic_variation.png
│       ├── y_0.pt # the pytorch tensor containing the input y of SNIPS
│       └── description.txt # text file with all of the experiment's configs and results
```

The file `runners/ncsn_runner.py` implements the high level pipeline of generating samples - getting the raw images, applying degradation and adding noise, calling a function to get the output samples and saving them.

The `models` module implements the `NCSNv2` Unet model used in the algorithm, and the `general_anneal_Langevin_dynamics()` function which holds the main logic for generating samples given a degraded image `y_0` and degradation operator `H`.

The `datasets` modules implements the handling of different datasets.

The `filter_builder.py` file used to generate uniform and gaussian deblurring kernels.

### Downloading data

You can download the aligned and cropped CelebA files from their official source [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). The LSUN files can be downloaded using [this script](https://github.com/fyu/lsun). For our purposes, only the validation sets of LSUN bedroom and tower need to be downloaded.

### Running SNIPS

If we want to run SNIPS on CelebA for the problem of super resolution by 2, with added gaussian noise of standard deviation 0.1, and obtain 3 variations, we can run the following

```bash
python main.py -i celeba --config celeba.yml --doc celeba -n 3 --degradation sr2 --noise_type gaussian --sigma_0 0.1
```

Samples will be saved in `<exp>/image_samples/celeba`.

The available degradations are: Inpainting (`inp`), Uniform deblurring (`deblur_uni`), Gaussian deblurring (`deblur_gauss`), Super resolution by 2 (`sr2`) or by 4 (`sr4`), Compressive sensing by 4 (`cs4`), 8 (`cs8`), or 16 (`cs16`). The sigma_0 can be any value from 0 to 1.

## Pretrained Checkpoints

Link: https://drive.google.com/drive/folders/1217uhIvLg9ZrYNKOR3XTRFSurt4miQrd?usp=sharing

These checkpoint files are provided as-is from the authors of [NCSNv2](https://github.com/ermongroup/ncsnv2). You can use the CelebA, LSUN-bedroom, and LSUN-tower datasets' pretrained checkpoints. We assume the `--exp` argument is set to `exp`. The checkpoint needs to be located at `<exp>/logs/<doc>/`, e.g. `exp/logs/celeba/checkpoint_210000.pth` for the CelebA dataset.

## Acknowledgement

This repo is largely based on the [NCSNv2](https://github.com/ermongroup/ncsnv2) repo, and uses modified code from [this repo](https://github.com/alisaaalehi/convolution_as_multiplication) for implementing the blurring matrix.

## References

If you find the code/idea useful for your research, please consider citing

```bib
@article{kawar2021snips,
  title={{SNIPS}: Solving noisy inverse problems stochastically},
  author={Kawar, Bahjat and Vaksman, Gregory and Elad, Michael},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={21757--21769},
  year={2021}
}
```

