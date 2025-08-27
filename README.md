# Research on the prediction of CO<sub>2</sub> absorbent properties based on machine learning

## Overview of the project
> This study proposes a data generation regression prediction model based on small data samples for predicting the performance of carbon dioxide adsorbents in direct air capture. The study employs a tabular variational autoencoder (TVAE) and a conditional tabular generative adversarial network (CTGAN) as the data generation framework, enhancing the training dataset with synthetic data to improve the model's generalization ability and prediction accuracy. Experimental results show that the model using generated data achieves a 9% to 12% reduction in prediction error compared to the model without generated data.
> 
<p align="center">
<img src="image\1.png"/>
</p>

The upside is the theoretical process of the Tabular-VAE model, downside part is the theoretical process of the CTGAN(Conditional Tabular GAN), The rightside part is the predictive model training and testing process

## Background of the project
> As global climate change intensifies, reducing greenhouse gas emissions has become a shared goal of the international community. Carbon dioxide capture technology is one of the key technologies for addressing climate change, and the development of efficient carbon dioxide adsorbents is at its core. Traditional adsorbent development relies on experimental screening, which is a cumbersome and costly process. To overcome the issue of data scarcity, this study combines machine learning methods and data generation techniques to propose a framework for predicting adsorbent properties suitable for small sample data.

## Method and Tech
> This study employs two generative data models: the Table Variational Autoencoder (TVAE) and the Conditional Table Generative Adversarial Network (CTGAN), combined with classical machine learning regression models (such as random forests, linear regression, and decision trees) for prediction. Synthetic data is generated to expand the training set, thereby enhancing the model's generalization ability in small-sample scenarios. The specific steps are as follows:
- **Data preparation and preprocessing**: Data is collected from public databases and literature, followed by missing value handling, outlier detection, and feature engineering.
- **Generative models**: TVAE and CTGAN are used to generate data, ensuring that the distribution of the generated data aligns with that of the original data.
- **Regression model training**: The regression models are retrained using the generated synthetic data, and their predictive performance is evaluated.
- **SHAP Analysis**: Use SHAP values to quantify the contribution of each feature to the model's prediction results.

## Data
- Database：The dataset used in this study comprises physical and chemical properties of carbon dioxide adsorbents along with experimental conditions collected from the literature. The data includes information such as the adsorbent's specific surface area, pore volume, nitrogen atom content, proportion of amine functional groups, temperature, and partial pressure of carbon dioxide.
<img src="image\README\label.png" width="200" />
The data was from the [this](https://www.sciencedirect.com/science/article/pii/S2666546825000096#sec0015) article

## Result
### The result of generating the data

<p float="left", align="center">
  <img src="image\README\rs1_1.png" width="200" />
  <img src="image\README\rs1_2.png" width="200" />
  <img src="image\README\rs2_1.png" width="200" />
  <img src="image\README\rs2_2.png" width="200" />
</p>

### The result of using the generate skill and without this skill
<div align="center">

|Model|MAE|MSE|R<sup>2</sup>|
|:---:|:---:|:---:|:---:|
|Random Forest|0.4504|0.4345|0.6895|
|TVAE-RF|0.4320|0.3483|0.7511|
|CTGAN-RF|0.4069|0.3247|0.7680|

</div>

## Package Installation
```bash
pip install -r requirements.txt
pandas
numpy
scipy
matplotlib
seaborn
statsmodels
scikit-learn
sdv
```
