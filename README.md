![Python 3.8.5](https://img.shields.io/badge/Python-3.8.5-blue.svg)

# OTAnalysis

## Tool for managing the results of optical tweezers
TODO

## Install MiniConda && Create conda environment
```{}
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
conda create -n newenv
conda activate newenv

```
## Dependencies
```{}
conda install numpy
conda install pandas
conda install scipy
pip install PyQt5
conda install pyqt qtpy
conda install matplotlib
```

## Launch project
```{}
python main.py
```

## Documentation

### Update
```{}
make html
```

### Visualization
```{}
firefox docs/_build/html/index.html
```
