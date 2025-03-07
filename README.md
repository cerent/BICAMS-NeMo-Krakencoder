# BICAMS-NeMo-Krakencoder

In this repo, we provide the codes of linear regression with the ridge regularization method and related libraries that are applicable to R version 4.4.2. The codes can be used to predict any numerical continuous metrics.

The model was trained with outer and inner loops of k-fold cross-validation (k = 5) to optimize the hyperparameters and test model performance. The inner loop (repeated over 5 different partitions of the training dataset only) optimized the hyperparameter that maximized the correlation for the validation dataset. A model was then fitted using the entire training dataset and this hyperparameter, which was then assessed on the hold-out test set from the outer loop. The outer loop was repeated for 100 different random partitions of the data.

The prediction accuracy was obtained using Spearman's and Pearson's correlation coefficients. Parameter coefficients (beta values of the classification models) and their Haufe-transformed versions were used to assess the variable importance.

Please contact Ceren Tozlu (tozluceren@gmail.com) for any questions about the repository.
