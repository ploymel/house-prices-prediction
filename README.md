# House Prices: Advanced Regression Techniques

## Competition Description
Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.  
With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.
Competition link -> [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## Requirements
* Tensorflow 2.x

## To Obtain the Model
1. Download dataset from [kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and place all the file in `data` folder
2. Preprocess Data including dealing with missing data and dummies by running
```
python preprocessing.py
```
For more information please go to [preprocessing jupyter](./preprocessing.ipynb)
3. Train the Model by running
```
python train.py
```
This code is included training the model and saving the prediction for submission  
For validation please go to [train juptyer](./train.ipynb)

## Reference
* [COMPREHENSIVE DATA EXPLORATION WITH PYTHON](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python#4.-Missing-data)
* [Regularized Linear Models](https://www.kaggle.com/apapiu/regularized-linear-models)