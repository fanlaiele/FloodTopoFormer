## FloodTopoFormer

Updated on 7/4/2026

FloodTopoFormer is an integrated geospatial tool for urban flood extraction, topographic correction, and exposure assessment using multi-source remote sensing data. It combines Sentinel-1/2 imagery, an enhanced permanent water mask, and Transformer-based DEM correction to support urban flood analysis.


## Requirements

### Google Earth Engine (GEE)
- A registered [Google Earth Engine](https://earthengine.google.com/) account
- Access to Sentinel-1, Sentinel-2, and GSW v1.4 datasets via GEE

### Python
- Python >= 3.8
- Required packages:
```bash


## GEE scripts
- `EPWM.txt`: Generates the Enhanced Permanent Water Mask (EPWM).
- `SARwater.txt`: Extracts water bodies from Sentinel-1 SAR imagery.
- `Accuracy Assessment.txt`: Assesses the accuracy of SAR-derived 

## Python scripts
- `transformer.py`: Trains and validates the Transformer-based DEM correction model.
- `pop_exposure.py`: Calculates flood-exposed population.
- `building_exposure.py`: Calculates flood-exposed buildings.


