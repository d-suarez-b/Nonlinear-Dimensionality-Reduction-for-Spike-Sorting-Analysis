### Environment Setup

This project requires a specific set of Python packages. You can set up a suitable Conda environment using one of the following methods:

**Method 1: Using `environment.yml` (Recommended)**

This method recreates the Conda environment that was used during development, including the exact Python version and package versions from the correct channels.

1.  Ensure you have Anaconda or Miniconda installed.
2.  Open your terminal or Anaconda Prompt.
3.  Navigate to the root directory of this project (where `environment.yml` is located).
4.  Create and activate the environment using the following commands:
    
    conda env create -f environment.yml

    conda activate <environment_name_from_yml>
    
**Method 2: Using `env_requirements.txt` with Pip**

This method uses `pip` to install the required packages into a manually created Conda environment.

1.  Ensure you have Anaconda or Miniconda installed.
2.  Open your terminal or Anaconda Prompt.
3.  Create a new Conda environment with a compatible Python version (e.g., Python 3.12).

    (Replace <your_env_name> and choose the python version)
    
    conda create --name <your_env_name> python=3.12
    
5.  Activate the new environment:
    
    conda activate <your_env_name>
    
6.  Navigate to the root directory of this project (where `env_requirements.txt` is located).
7.  Install the required packages using pip:
    
    pip install -r env_requirements.txt
    
    *Note: Some packages might have system dependencies (like Git) that need to be installed separately if `pip` encounters errors.*

**Final Step: Install Sorting (core scripts)**

After setting up and activating the environment using **either Method 1 or Method 2**, install Sorting (core scripts) itself in development mode. 
Navigate to the project's root directory in your terminal and run:


python setup.py develop


This makes the Sorting package importable while allowing you to edit the source code directly.

