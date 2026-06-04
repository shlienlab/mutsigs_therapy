
## Prior Therapy Defines Mutation Profiles in Childhood Cancer at Relapse

#### Scripts and Notebooks to reproduce figures in the paper above.


---
### v 1.0

This repository includes

* 5 Notebooks to reproduce Main Figures
* 10 Notebooks to reproduce Extended Data Figures
* 1 Notebook to reproduce Supplementary Notes 2.1-9 & Supplementary Figures 1-8
* 1 helper script with all the plotting functions
* 2 helper scripts with 10s of analysis functions
* 1 script to run the logistic regression model
* 1 source data directory containing all the data needed to reproduce all the figures in the manuscript


---
### Dependencies

```
- scikit-learn==1.3.0
- shap==0.46.0
- ipykernel==6.25.2
- ipython==8.15.0
- pandas>2.1.0
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
- pydeseq2==0.5.1
- ipykernel
- jupyter
```

---
### Download

To install, you can directly download the zipped folder from menu above or run this command:
 
    git clone https://github.com/shlienlab/mutsigs_therapy


---
### Installation

#### We recommend installing the required packages into a python virtual environment.

1. Create the virtual environment in the specified path of your choice
```sh
python3 -m venv /your_path/your_env_name
```

2. Activate the virtual environment
```sh
source /your_path/your_env_name/bin/activate
```

3. Install requirements (from inside the root directory of the repository)
```sh
pip install -r requirements.txt
```

4. Add the new environment to kernels list
```sh
python3 -m ipykernel install --name "your_env_name" --user
```

Otherwise, skip the virtual environment steps, navigate to the root directory and type the following:
```sh
pip install -r requirements.txt
```

---
### Usage

After installation, you can open and run the Jupyter notebooks in any compatible (e.g., VS Code).
If you created a virtual environment, make sure to select it as your kernel.

---
### Source files

All the data needed to reproduce figures are located in the `source_data/` directory.<br>
The code to produce each panel (in every main, extended data, and supplementary figure) is self-contained and automatically loads the necessary source data.

---
### Citation

When using this library, please cite:
> Layeghifard M., ..., and Shlien A., "Prior Therapy Defines Mutation Profiles in Childhood Cancer at Relapse" (under review).

---
### Contributions

This library is still a work in progress.
Contributions are always welcome.
