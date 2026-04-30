# Fast-and-Certified-Bounding-of-Security-Constrained-DCOPF-via-Interval-Bound-Propagation

## Setup Instructions for Python code

### 1. Clone the repository
```bash
git clone <repo-url>
cd DCOPF_abC_Python
```

### 2. Create and activate Conda environment
```bash
conda create -n SCDCOPF python=3.12.13
conda activate SCDCOPF
```

### 3. Install the main package (editable mode)
```bash
pip install -e .
```

### 4. Install `auto_LiRPA`
Go to https://github.com/Verified-Intelligence/auto_LiRPA and clone the repository into the main directory. Then install the requirements:
```bash
cd auto_LiRPA
pip install .
```

### 5. Create symbolic link
```bash
cd DCOPF_abC_Python
ln -s ../auto_LiRPA
```

## Setup Instructions for Julia code
Julia version 1.10.5 is used in this project. Please make sure you have this version available. When you are in the main directory, run the following code:
```bash
julia
using Pkg
Pkg.activate("DCOPF_abC_Julia")
Pkg.instantiate()
```

## Run the code
In main directory run the following in the terminal:
```bash
python -m DCOPF_abC_Python.reproduce_results
```