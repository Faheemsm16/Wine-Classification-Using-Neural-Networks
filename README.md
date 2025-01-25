# Wine-Classification-Using-Neural-Networks

This project demonstrates how to train a neural network to classify wine samples based on their chemical properties using the Wine dataset. The Wine dataset consists of 178 samples with 13 features, representing different chemical properties of wines from three different cultivars.

## Procedure
**1. Dataset:**

  The Wine dataset contains 178 samples with 13 features:
   - Alcohol content
   - Malic acid
   - Ash content
   - Alcalinity of ash
   - Magnesium content
   - Total phenols
   - Flavanoids
   - Nonflavanoid phenols
   - Proanthocyanins
   - Color intensity
   - Hue
   - OD280/OD315 of diluted wines
   - Proline
     
  There are 3 classes (cultivars) that represent different types of wines.

**2. Preprocessing:**

One-Hot Encoding: The target labels are one-hot encoded to be compatible with the softmax activation in the output layer.

Feature Scaling: The features are standardized (zero mean and unit variance) to speed up the training process and improve convergence.

**3. Model:**

A simple feedforward neural network is constructed with two hidden layers, each using the ReLU activation function.

The output layer employs the softmax activation function to perform multi-class classification.

**4. Training:**

The model is trained using the Adam optimizer and categorical crossentropy as the loss function.

Training is performed for 50 epochs with a batch size of 8, and a validation split of 20% is used to monitor the model's performance during training.

**5. Evaluation:**

After training, the model is evaluated on a test set to check its accuracy.

## Output:

**1. Training Output:**

During training the model's loss and accuracy are displayed for each epoch:

Epoch 1/50

15/15 ━━━━━━━━━━━━━━━━━━━━ 1s 16ms/step - accuracy: 0.2815 - loss: 1.0890 - val_accuracy: 0.3448 - val_loss: 1.0659

Epoch 2/50

15/15 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.4626 - loss: 1.0042 - val_accuracy: 0.5517 - val_loss: 0.9762

...

Epoch 50/50

15/15 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.9956 - loss: 0.0320 - val_accuracy: 0.9655 - val_loss: 0.0776

**2. Test Accuracy:**

After training, the model's accuracy on the test dataset is reported:

Test Accuracy: 0.97

This indicates that the model achieved 100% accuracy on the test set.

**3. Predictions:**

The predicted classes for the test set are compared with the true classes:

Predicted classes: [0 0 2 0 1 0 1 2 1 2 0 2 0 2 0 1 1 1 0 1 0 1 1 2 2 2 1 1 1 0 0 1 2 0 0 0]

True classes:      [0 0 2 0 1 0 1 2 1 2 0 2 0 1 0 1 1 1 0 1 0 1 1 2 2 2 1 1 1 0 0 1 2 0 0 0]

## Explanation:

Training Output: The model's loss and accuracy for both training and validation sets are shown after each epoch.

Test Accuracy: The model's final accuracy when evaluated on the test dataset. Here, it achieved a test accuracy of 96%.

Predictions: The predicted classes for the test samples are displayed alongside the true classes.

## Notes:

The architecture of the model (number of layers, units, epochs, and batch size) can be fine-tuned to further improve performance.

The Wine dataset's features (chemical properties) are well-suited for classification tasks, and neural networks can perform very well on such datasets.
