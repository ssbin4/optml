# OptML Final Project

### Prepare data

```
mkdir CARS196
cd CARS196
wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz
wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
tar -xvzf cars_train.tgz
tar -xvzf car_devkit.tgz
```

### Prepare virtual environment

```
conda create --name cbm python=3.6.13
conda env create --file environment.yaml
```

To activate the environment,
```
source activate cbm
```

### Training

Use optml.sh file.
Just change the environment name.

### Testing

Not completed yet.