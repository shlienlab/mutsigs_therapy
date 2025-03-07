
## Unique patterns of mutations in childhood cancer highlight chemotherapy’s disease-defining role at relapse

#### Scripts and Notebooks to reproduce figures in the above paper.


### v 0.1

This repository includes

* 5 Notebooks to reproduce Main Figures
* 10 Notebooks to reproduce Extended Data Figures
* 1 Notebook to reproduce Supplementary Figures
* 1 helper script with all the plotting functions
* 2 helper scripts with 10s of analysis functions
* 1 source data directory containing all the data needed to reproduce the figures (data will be added after manuscript decision)


### Dependencies

```
- scikit-learn==1.3.0
- shap==0.46.0
- ipykernel==6.25.2
- ipython==8.15.0
- pandas==2.1.0
- numpy==1.24.4
- scipy==1.11.2
- matplotlib==3.7.3
- seaborn==0.13.2
- plotly==5.16.1
- pywaffle==1.1.0
- xgboost==1.7.6
- catboost==1.2.1
- UpSetPlot==0.9.0
- nbformat==5.10.4
- patsy==1.0.1
- networkx==3.4.2
- statsmodels==0.14.4
```


### Installation

To install you can download this repository by running 
 
    git clone https://github.com/shlienlab/mutsigs_therapy

Or directly download the repository from the Github webpage.

Navigate to the root directory of cloned or downloaded repository and type:
```sh
pip install -r requirements.txt
```

We recommended installing the required packages into a python virtual environment.
To do so:

## Create the virtual environment in the specified path of your choice
```sh
python3 -m venv /your_path/your_env_name
```

## Activate the virtual environment
```sh
source/your_path/your_env_name
```

## Install requirements
```sh
pip install -r requirements.txt
```

### Usage

After installation, you can open the notebooks in the software of your choice (e.g., VS Code).
Make sure you select the created virtual environment as your kernel.


### Citation

When using this library, please cite

> Layeghifard M., ..., and Shlien A., "Unique patterns of mutations in childhood cancer highlight chemotherapy’s disease-defining role at relapse" (under review).


### Contributions

This library is still a work in progress.
Contributions are always welcome.
