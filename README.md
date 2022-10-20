# License

Copyright (C) 2022 Li Peng (plpeng@hnu.edu.cn), Yuan Tu (ty09060418@163.com)

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, see http://www.gnu.org/licenses/.

# DAESTB

## Method Description 
DAESTB is a effective method to predict associations between small molecule and miRNA, which is based on deep autoencoder and a scalable tree boosting model. The code and details of the paper will be released in this page after the paper accepted.


## Required Packages

* python==3.6 (or a compatible version)
* tensorflow==2.1.0 (cuda==11.4)
* keras==2.3.1
* numpy==1.19.5
* sklearn==0.0
* matplotlib==3.3.4
* xgboost==1.2.0

## File Description
* dataset: this folder contains two datasets of small molecules and miRNAs.
* dataprocessing.py: this file is used for data pre-processing and calculating performance evaluation metrics.
* DeepAE.py: feature dimensionality reduction using deep autoencoder.
* DAESTB.py: the core model proposed in the paper.

### Contact
* If you have any problem or find mistakes, please feel free to contact me: plpeng@hnu.edu.cn and ty09060418@163.com



