# python-resources

This repo git contains different functions for climate data access, post-processing and analysis. 

To date there are the following modules:

- **Loader**: basic files and code for data access to the Santander Climate Data Service (SCDS) THREDDS Data Server, which provides user-friendly access to a variety of remote climate data sources. This service is maintained by the Santander Meteorology Group (University of Cantabria - CSIC).
- **Interpolater**: wrapping packages for regridding based ond the [xESMF](https://xesmf.readthedocs.io/en/latest/) python package.
- **Climindx**: basic code for climate index calculation and wrapping function based on [icclim](https://icclim.readthedocs.io/en/stable/index.html) **[IN PROGRESS]**.
- **Downscaler**: wrapping funcions for bias adjusting based on ISIMIP3 and [Ibicus](https://ibicus.readthedocs.io/en/latest/index.html) **[IN PROGRESS]**.


## Requirements

Scripts and (jupyter) notebooks are provided in [Python](https://www.python.org/) to ensure reproducibility and reusability of the results. The simplest way to match all these requirements is by using a dedicated [conda](https://docs.conda.io) environment, which can be easily installed by issuing:

```sh
conda create -n python-resources
conda activate python-resources
conda env update --file environmet.yml --prune
```
