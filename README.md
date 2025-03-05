# Company Scaling

This is the schama code for the work [Predicting the size-based growth of companies across the global market.](https://arxiv.org/abs/2109.10379). 

## Overview

(abstract)

## Data Description

Datasets for the US and Europe are derived from Compustat. It contains the financial information obtained from the income statements and balance sheets of publicly traded companies. We have 44,256 North America companies from 1950 to 2023, and 15.601 European companies within 51 countries from 1987 to 2023. Classification of companies by sector is performed according to the Global Industry Classification Standard (GICS), developed by the developed by Standard & Poor’s (S&P) for use by the global financial community.

Dataset for China is derived from the Shanghai Stock Exchange (SSE) & and Shen Zhen Stock Exchange (SZSE) Stock Database, collected by Wind Information Co.,Ltd, which is the leading financial data, information, and software services company in China. It contains the financial information obtained from the income statements and balance sheets of publicly traded companies in China from 1996 to 2022 for our sample of 3,160 companies. Classification of companies by sector is performed using the Global Industries Classification Standard (GICS) codes, the Wind industry
classification standard is fine adjusted based on the situation in China.


## System requirements

### OS Requirements
This package is supported for macOS and Linux. The package has been tested on the following systems:

macOS: 13.0.1
Linux: Ubuntu 16.04

Python(3.7.13) Dependencies

```
scipy
numpy
matplotlib
pandas
seaborn
```

## Demo

We showed some samples data of our whole datasets here as example. 

`fit_params.ipynb`  is used for fitting scaling parameters

`unified_growth_curve.ipynb` is used for calculting growth curve

