# LES Precipitation Forecasting - a multi modal approach to analyze temporal and satellite data

## Problem Statement
- Develop a hybrid deep learning model (an encoder decoder architecture) that integrates meteorological weather data and satellite imagery to predict future weather forecasts.

## Approach
- Utilized a provided notebook for initial data preparation, including cleaning and feature selection:
  - Removed features with high correlations and excessive NaN values.
- Implemented sliding windows of time-series for data input:
  - 24-hour meteorological data and 8-hour image data for next day precipitation prediction.
  - 48-hour meteorological data and 16-hour image data for subsequent day precipitation prediction.
- Addressed multi-class classification by distributing labels into four balanced classes to manage class imbalance.
- Explored multiple model architectures:
  1. ConvLSTM2D + LSTM Shallow
  2. ConvLSTM2D + LSTM Deep
  3. Conv3D + ConvLSTM2D + LSTM Shallow
  4. Conv3D + ConvLSTM2D + LSTM Deep
  - Additionally, a single Conv3D + ConvLSTM2D + LSTM Shallow model was used for the 48-hour window.
- Evaluated models through:
  - Publishing scores and classification reports.
  - Generating plots of Training and Validation Loss/Accuracy.

### Results

ConvLSTM2D + LSTM (shallow)

```
              precision    recall  f1-score   support

           0       0.63      0.44      0.52       149
           1       0.34      0.48      0.40        93
           2       0.00      0.00      0.00        21
           3       0.20      0.27      0.23        48

    accuracy                           0.40       311
   macro avg       0.29      0.30      0.29       311
weighted avg       0.43      0.40      0.40       311
```

### Conv2D + LSTM in an Encoder Decoder Architecture (Deep)

```
              precision    recall  f1-score   support

           0       0.63      0.44      0.52       149
           1       0.34      0.48      0.40        93
           2       0.00      0.00      0.00        21
           3       0.20      0.27      0.23        48

    accuracy                           0.40       311
   macro avg       0.29      0.30      0.29       311
weighted avg       0.43      0.40      0.40       311

```

### Conv3D_LSTM + LSTM in an Encoder Decoder Architecture (Shallow)

```
           0       0.60      0.59      0.60       149
           1       0.35      0.37      0.36        93
           2       0.10      0.10      0.10        21
           3       0.30      0.29      0.29        48

    accuracy                           0.44       311
   macro avg       0.34      0.34      0.34       311
weighted avg       0.45      0.44      0.44       311
```

Conv3D_LSTM + LSTM in an Encoder Decoder Architecture (Deep Network)

```
              precision    recall  f1-score   support

           0       0.48      1.00      0.65       149
           1       0.00      0.00      0.00        93
           2       0.00      0.00      0.00        21
           3       0.00      0.00      0.00        48

    accuracy                           0.48       311
   macro avg       0.12      0.25      0.16       311
weighted avg       0.23      0.48      0.31       311

```

### CONV3D + CONV2DLSTM + LSTM - 76M Params

```

              precision    recall  f1-score   support

           0       0.36      0.32      0.34       119
           1       0.32      0.20      0.25        95
           2       0.09      0.29      0.14        24
           3       0.27      0.26      0.27        72

    accuracy                           0.27       310
   macro avg       0.26      0.27      0.25       310
weighted avg       0.31      0.27      0.28       310

```

## Conclusion

Based on the limited training resources, we could only try a few variations in our architecture but still we saw quite different results. This shows the model architecture has a role to play when it comes to Deep neural network prediction/inference

Also, the satellite images were shrunk to 64x64 pixels which might've led to a loss of data that could've otherwise made a difference, if we had to optimize accuracy. But we traded off training time with trying different approaches, as model training took a significant time during the course of this project

We were able to achieve a maximum f1-score of 0.51. The results aren't the most path breaking. But we feel the motive wasn't that either. Executing the idea, understanding forecasting techniques and modelling SOTA deep neural networks using multi modal data was the crux of this project.

## Contact

If you want to reproduce results or want our model checkpoints, please feel free to reach out.