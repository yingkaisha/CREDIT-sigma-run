# Investigating the contribution of terrain-following coordinates and conservation schemes in AI-driven precipitation forecasts

Yingkai Sha, John S. Schreck, William Chapman, David John Gagne II

NSF National Center for Atmospheric Research, Boulder, Colorado, USA

## Abstract

Artificial Intelligence (AI) weather prediction (AIWP) models often produce ``blurry'' precipitation forecasts that overestimate drizzles 
and underestimate extremes. This study provides a novel solution to tackle this problem---integrating terrain-following coordinates with 
global mass and energy conservation schemes into AIWP models. Forecast experiments are conducted to evaluate the effectiveness of this 
solution using FuXi, an example AIWP model, adapted to 1.0$^\circ$ grid spacing data. Verification results show large performance gains. 
The conservation schemes are found to reduce drizzle bias, whereas using terrain-following coordinates improves the estimation of extreme 
events and precipitation intensity spectra. Furthermore, a case study revealed that terrain-following coordinates capture near-surface 
winds better over mountains, offering AIWP models more accurate information on understanding the dynamics of precipitation processes. 
The proposed solution of this study can benefit a wide range of AIWP models and bring insights into how atmospheric domain knowledge 
can support the development of AIWP models.

## Resources

* NSF NCAR Research Data Archive, [ERA5 Reanalysis (0.25 Degree Latitude-Longitude Grid)](https://rda.ucar.edu/datasets/d633000/)
  * Note: this repository visits RDA internally, e.g., `/glade/campaign/collections/rda/data/d633000/e5.oper.an.pl/197901/`

* Google Research, Analysis-Ready, Cloud Optimized (ARCO) ERA5 [[link](https://cloud.google.com/storage/docs/public-datasets/era5)]

* Goddard Earth Science Data and Information Science Center (GES-DISC), NASA, Integrated Multi-satellitE Retrievals for Global Precipitation Measurement (IMERG) Final Precipitation L3 daily product (GPM\_3IMERGDF) version 7.0 [[link](https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGDF_07/summary)]

* IFS-HRES and ERA5 climatology from the Weatherbench2 project [[link](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)]

* NSF NCAR MILES Community Research Earth Digital Intelligence Twin (CREDIT) [[link](https://github.com/NCAR/miles-credit)]

* The implementation of conservation schemes in CREDIT [[link](https://github.com/NCAR/miles-credit/blob/main/credit/postblock.py)]

## Navigation

* Derivations of conservation schemes: [Pytorch integration](https://github.com/yingkaisha/CREDIT-sigma-run/blob/main/physics/DEV00_pytorch_model_level_physcis.ipynb)
* Results: [TS and SEEPS](https://github.com/yingkaisha/CREDIT-sigma-run/blob/main/visualization/FIG02_TS_SEEPS.ipynb), [quantile-based verification](https://github.com/yingkaisha/CREDIT-sigma-run/blob/main/visualization/FIG03_Histogram.ipynb), [case study](https://github.com/yingkaisha/CREDIT-sigma-run/blob/main/visualization/FIG04_example.ipynb)

## Contact


