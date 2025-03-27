# TopoSort
Algorithm for the classification of neural action potentials.

We utilize UMAP, a non-linear dimensionality reduction technique, to provide a low dimensional representation of neuronal action potentials. In this low dimensional representation, leveraging the fact that UMAP preserves local structures, to cluster with a hierarchal density based clustering algorithm (HDBSCAN). 


### Standard Data Format

The toposort function is designed to take a very specific set of inputs to work properly. This is done to optimize the code and facilitate the sharing of results inbetween different labs. The idea is that each lab or researcher create their own data extraction function that formats the data into the following structure:

**spikes:** array, shape (n_spikes, n_sample_points), dtype float64 <br/>
&emsp;&emsp;&emsp;&emsp; Array of all spike waveforms you wish to analyze. Contains one spike per row.

**spike_times:** array, shape (n_spikes, ), dtype float64 <br/>
&emsp;&emsp;&emsp;&emsp; Array indicating the time at which each spike was registered within its specific trial.
        
**spike_trial:** array, shape (n_spikes, ), dtype int <br/>
&emsp;&emsp;&emsp;&emsp;Array indicating the trial in which each spike was registered.

**trial_class:** array, shape (n_trials, ), dtype int <br/>
&emsp;&emsp;&emsp;&emsp; Array indicating the class of every trial.

### Dependencies

- [numpy](https://numpy.org/doc/)
- [pandas](https://pandas.pydata.org/docs/)
- [umap](https://umap-learn.readthedocs.io/en/latest/)
- [sklearn](https://scikit-learn.org/stable/)
- [hdbscan](https://hdbscan.readthedocs.io/en/latest/)
- [matplotlib](https://matplotlib.org/)
- [bokeh](https://docs.bokeh.org/en/latest/index.html)
- [plotly](https://plotly.com/)
- [holoviz](https://holoviz.org/)      

These can be installed in an Anaconda environment by running:

conda install -c pyviz holoviz

conda install -c conda-forge umap-learn, scikit-learn, numpy, pandas, bokeh, plotly, hdbscan, matplotlib


### Installation
To install, simply cd into the root toposort directory and run 

python setup.py develop

This will develop the package instead of installing it. Allowing for easier updates.

### Data 
Sample data for the notebooks can be found in the following [link](https://drive.google.com/drive/folders/1iz90i7GaNXxXxHqRyaSwI2cIezbmv-BS?usp=sharing)





### Figure reproduction of the article: **Relevance of Nonlinear Dimensionality Reduction for Efficient and
Robust Spike Sorting**

syntetic dataset: download at: 
Figure reproduction and da
