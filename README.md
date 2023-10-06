# Seg2Map :mag_right: :milky_way:

_An interactive web map app for applying Doodleverse/Zoo models to geospatial imagery_

![](https://user-images.githubusercontent.com/3596509/194389595-82ade668-daf0-4d24-b1a0-6ecf897f40fe.gif)

![separate_seg_controls_demo (1)](https://github.com/Doodleverse/seg2map/assets/61564689/d527fe8c-c3f2-4c62-b448-e581162e8475)

## Overview:

- Seg2Map facilitates application of Deep Learning-based image segmentation models and apply them to high-resolution (~1m or less spatial footprint) geospatial imagery, in order to make high-resolution label maps. Please see our [wiki](https://github.com/Doodleverse/seg2map/wiki) for more information.

- The principle aim is to generate time-series of label maps from a time-series of imagery, in order to detect and assess land use/cover change. This project also demonstrates how to apply generic models for land-use/cover on publicly available high-resolution imagery at arbitrary locations.

- Imagery comes from Google Earth Engine via [s2m_engine](https://github.com/Doodleverse/s2m_engine). Initially, we focus on [NAIP](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-national-agriculture-imagery-program-naip) time-series, available for the conterminious United States since 2003. In the future, [Planetscope](https://developers.planet.com/docs/data/planetscope/) imagery may also be made available (for those with access, such as federal researchers).

- We offer a set of [Segmentation Zoo](https://github.com/Doodleverse/segmentation_zoo) models, especially created and curated for this project based on a publicly available datasets. These datasets have been selected because they are public, large (several hundred to several thousand labeled images), and provide broad class labels for generic land use/cover mapping needs.

# Installation Instructions

In order to use seg2map you need to install Python packages in an environment. We recommend you use [Anaconda](https://www.anaconda.com/products/distribution) to install the python packages in an environment for seg2map. After you install Anaconda on your PC, open the Anaconda prompt or Terminal in Mac and Linux and use the `cd` command (change directory) to go the folder where you have downloaded the seg2map repository.

1. Create an Anaconda environment

- This command creates an anaconda environment named `seg2map` and installs `python 3.10` in it
- You can also use `python 3.10`
- We will install the seg2map package and its dependencies in this environment.
  ```bash
  conda create --name seg2map python=3.10 -y

2. Activate your conda environment

   ```bash
   conda activate seg2map
   ```

- If you have successfully activated seg2map you should see that your terminal's command line prompt should now start with `(seg2map)`.

3. Install Conda Dependencies

- seg2map requires `jupyterlab` and `geopandas` to function properly so they will be installed in the `seg2map` environment.
- [Geopandas](https://geopandas.org/en/stable/) has [GDAL](https://gdal.org/) as a dependency so its best to install it with conda.
- Make sure to install geopandas from the `conda-forge` channel to ensure you get the latest version.
- Make sure to install both jupyterlab and geopandas from the conda forge channel to avoid dependency conflicts
  
```bash
  conda install -c conda-forge geopandas jupyterlab -y
  ```

4. Install the seg2map from PyPi
   ```bash
   pip install seg2map
   ```
5. Uninstall the h5py installed by pip and reinstall with conda-forge
   ```bash
   pip uninstall h5py -y
   conda install -c conda-forge h5py -y
   ```
## **Having Installation Errors?**

Use the command `conda clean --all` to clean old packages from your anaconda base environment. Ensure you are not in your seg2map environment or any other environment by running `conda deactivate`, to deactivate any environment you're in before running `conda clean --all`. It is recommended that you have Anaconda prompt (terminal for Mac and Linux) open as an administrator before you attempt to install `seg2map` again.

#### Conda Clean Steps

```bash
conda deactivate
conda clean --all
```

# How to Use Seg2Map

1. Sign up to use Google Earth Engine Python API

First, you need to request access to Google Earth Engine at https://signup.earthengine.google.com/. It takes about 1 day for Google to approve requests.

2. Activate your conda environment

   ```bash
   conda activate seg2map
   ```

- If you have successfully activated seg2map you should see that your terminal's command line prompt should now start with `(seg2map)`.

3. Install the seg2map from PyPi
   ```bash
   cd <location you downloaded seg2map>
   ex: cd C:\1_repos\seg2map
   ```
4. Launch Jupyter Lab

- make you run this command in the seg2map directory so you can choose a notebook to use.
  ```bash
  jupyter lab
  ```

## Features

### 1. Download Imagery from Google Earth Engine

Use google earth engine to download multiple years worth of imagery.
![download_imagery_demo](https://github.com/Doodleverse/seg2map/assets/61564689/a36421de-e6d2-4a3f-8c08-2e47be99e3e0)

### You can download multiple ROIs and years of data at lighting speeds 🌩�
