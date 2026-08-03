# Foam-dynamics-prediction

This project contains the python code used for the resaerch paper: Predicting Plasticity in Two-Dimensional Foam Channel Flow Around an Obstacle. This work was done by myself, Alexandre Stepanetz and Bahaa Mazloum, Benjamin Dollet and Misaki Ozawa at the LIPhy laboratory in Grenoble.

The goal is to predict the future dynamics of a dense and amorphous simulated particle system using machine learning techniques. you can find data at the folloing link:
https://zenodo.org/records/21774620?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjM3M2JjNDgyLWFkYjAtNDhjOS05ZTU3LWE2MmRkYWJmYjk0NyIsImRhdGEiOnt9LCJyYW5kb20iOiIwZmY5YTdkYWRjNTk5YTMwMDhiMzRiMTNhMDY2NGQ3NiJ9.FnfvtRxgKFgVrNHiCaiMj4yTVWuJFE2IYt8GaXd_P1l9-9hDfMtLJcqgUPYsSHtAi2_UH2-2wOqdkMAZzxTbpw

## The environnement

The code was created under python 3 and you will need some packages to run the different python files: 

- time
- os
- pathlib
- math
- tqdm
- mpl_toolkits
- numpy
- matplotlib
- sklearn
- scipy
- multiprocessing


## Data format

Download the data (link above). The files are text files. There is a header: number of particle, whidth and height of the simulation box. And the data themselves (900*1000) x 5: position x and y, size of the particle and the two targets (neighbor rearrangement and non affine displacement) for the 900 particles of 1000 independant configurations. Each file correspond to a timescale studied ($\Delta t$): 50, 100, 200, 300, 600, 900, 1200 and 1500.
