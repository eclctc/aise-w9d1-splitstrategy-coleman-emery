# Team Information

## Team Members
- **Partner A**: Jumoke Emery - Random Holdout Strategy
- **Partner B**: Brett Coleman - Stratified/Time-Aware Strategy

## Dataset Choice
- **Dataset Name**: Wine Quality (Multiclass)
- **Dataset Number**: 2
- **Task Type**: Classification
- **Key Characteristic**: Balanced - Only three wine types

## Why We Chose This Dataset
We chose this particular data set because it was covered in class already. Therefore it was easy to go back and reference our notes without too much confusion when implementing the cross validation and train/test/split/fit/predict functionality.

## Agreed-Upon Metric
- **Metric**: Accuracy
- **Why this metric**: We wanted to see how the model’s cross-validated training scores compared to its final prediction results on the train/test split.

## Loading Code
from sklearn.datasets import load_wine  
data = load_wine(as_frame=True)  
X, y = data.data, data.target

## Package Versions
- Python: 3.12.7
- scikit-learn: 1.5.1
- pandas: 2.2.2
- numpy: 1.26.4