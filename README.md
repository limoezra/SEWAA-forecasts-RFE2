# cGAN Rainfall Downscaling with RFE2

Retraining a conditional GAN (WGAN-GP) to downscale IFS forecasts using RFE2 satellite rainfall as truth data for the ICPAC/SEWAA region.

## Overview

This project adapts the [dsrnngan](https://github.com/Fenwick-Cooper/SEWAA-forecasts) cGAN framework to use **RFE2 daily rainfall** instead of IMERG 6-hourly data. The model takes IFS ensemble forecast fields and generates high-resolution probabilistic rainfall predictions over the Greater Horn of Africa.

## Domain

- **Region:** ICPAC / Greater Horn of Africa
- **Grid:** 384 x 352 pixels, 0.1 deg resolution
- **Lat:** -13.65 to 24.65, **Lon:** 19.15 to 54.25
- **Countries:** Kenya, Uganda, Tanzania, Ethiopia, Somalia, Djibouti, Eritrea, Sudan, South Sudan, Rwanda, Burundi

## Data

### IFS Forecast Fields (4 predictors)
| Variable | Description |
|----------|-------------|
| tp | Total precipitation |
| t2m | 2-metre temperature |
| tcwv | Total column water vapour |
| sp | Surface pressure |

- Source: rain.physics.ox.ac.uk
- Years: 2018-2021
- Each field provides 2 channels (ensemble mean + std) = **8 input channels**

### RFE2 Truth Data
- Source: NOAA CPC FTP (ftp.cpc.ncep.noaa.gov)
- Daily rainfall estimates (mm/day), converted to mm/hr for training
- Regridded onto the exact IFS grid (384x352) using bilinear interpolation
- Grid coordinates copied from elev.nc for cell-for-cell alignment

### Constants
- **elev.nc** - Surface elevation (divided by 10,000 for normalization)
- **lsm.nc** - Land-sea mask (0-1)

## Training Setup

| Parameter | Value |
|-----------|-------|
| Mode | GAN (WGAN-GP) |
| Architecture | forceconv |
| Training years | 2018, 2019, 2020, 2021 |
| Validation year | 2020 |
| Test period | 2023 onwards |
| Patch size | 128 x 128 pixels |
| Input channels | 8 (4 fields x 2) |
| Constant fields | 2 (elevation + LSM) |
| Content loss | ensmeanMSE |

## Repository Structure

```
CGAN-RFE2/
  SEWAA-forecasts/
    24h_accumulations/cGAN/
      dsrnngan/           # Core model code
        data.py           # Data loading (modified for RFE2)
        data_generator.py # Batch generator
        tfrecords_generator.py  # TFRecords creation
        main.py           # Training entry point
        train.py          # Training loop
        predict.py        # Generate predictions
        evaluation.py     # CRPS, RMSE evaluation
        gan.py            # WGAN-GP model
        models.py         # Generator/Discriminator architectures
        config.yaml       # Training configuration
      FCSTNorm2018.pkl    # Forecast normalization constants
    cGAN_data/
      elev.nc             # Elevation
      lsm.nc              # Land-sea mask
  cGAN_setup/             # Data preparation notebooks
  Scripts/                # Download and utility scripts
  Train_cGAN_RFE2_Colab.ipynb  # Colab training notebook
```

## Key Changes from Original (IMERG)

1. `all_fcst_fields`: 13 fields -> 4 (tp, t2m, tcwv, sp)
2. `HOURS`: 6 -> 24 (daily instead of 6-hourly)
3. `get_dates`: Rewritten for RFE2 daily files (checks D+1)
4. `load_truth_and_mask`: Reads pre-regridded 384x352, handles NaN, converts mm/day -> mm/hr
5. Rainfall bins recalibrated: [0.006, 0.036, 0.076] mm/hr
6. TF imports moved inside functions to avoid HDF5 DLL conflicts

## How to Run

### Generate TFRecords
```python
from tfrecords_generator import write_data
write_data(2018)
write_data(2019)
write_data(2020)
write_data(2021)
```

### Train
```bash
cd SEWAA-forecasts/24h_accumulations/cGAN/dsrnngan
python main.py --config config.yaml
```

### Evaluate
```bash
python main.py --config config.yaml --no_train --evaluate --eval_blitz
```

### Generate Predictions
```bash
python predict.py --log_folder ../logs_RFE2_run03 --model_number 0015872 --num_samples 5 --pred_ensemble_size 3
```

## Data Not Included (too large for GitHub)

- `IFS_training/` - IFS forecast NetCDF files (~120 GB)
- `RFE/` - RFE2 daily rainfall files (~50 GB)
- `rfe_tfrecords/` - Training TFRecords (~12 GB)
- `logs_RFE2_run*/` - Model weights and training logs

## Credits

- **dsrnngan framework:** Fenwick Cooper, University of Oxford
- **SEWAA project:** ICPAC / WMO
- **RFE2 adaptation:** Ezra Kiplimo, ICPAC
