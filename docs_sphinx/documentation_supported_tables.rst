.. list-table:: Algorithms supported
   :widths: 25 25 50
   :header-rows: 1

   * - Algorithm name
     - Algorithm Family
     - Description
   * - autoenocder_dense
     - autoenocder
     - DNN based encoder decoder architecture
   * - autoencoder_lstm
     - autoenocder
     - LSTM based encoder decoder architecture
   * - autoencoder_cnn
     - autoenocder
     - CNN based encoder decoder architecture
   * - autoencoder_cnn_variational
     - autoenocder
     - Variational autoenocder based encoder decoder architecture


.. list-table:: optimizer supported
  :widths: 25 25 50
  :header-rows: 1

  * - Optimizer name
    - Mapped Function
    - Description
  * - adam
    - tf.keras.optimizers.Adam()
    -
  * - adamax
    - tf.keras.optimizers.Adamax()
    -
  * - nadam
    - tf.keras.optimizers.Nadam()
    -
  * - adagrad
    - tf.keras.optimizers.Adagrad()
    -
  * - rmsprop
    - tf.keras.optimizers.RMSprop()
    -
  * - sgd
    - tf.keras.optimizers.SGD()
    -


.. list-table:: metrics supported
  :widths: 25 25 50
  :header-rows: 1

  * - Optimizer name
    - Mapped Function
    - Description
  * - mse
    - tf.keras.metrics.MeanSquaredError()
    -
  * - rmse
    - tf.keras.metrics.RootMeanSquaredError()
    -
  * - mae
    - tf.keras.metrics.MeanAbsoluteError()
    -
  * - mape
    - tf.keras.metrics.MeanAbsolutePercentageError()
    -
  * - msle
    - tf.keras.metrics.MeanSquaredLogarithmicError()
    -
  * - mean
    - tf.keras.metrics.Mean()
    -
  * - sum
    - tf.keras.metrics.Sum()
    -
  * - kldiv
    - tf.keras.metrics.KLDivergence()
    -

.. list-table:: loss functions supported
  :widths: 25 25 50
  :header-rows: 1

  * - Loss function name
    - Mapped Function
    - Description
  * - mse
    - tf.keras.losses.MeanSquaredError()
    -
  * - mae
    - tf.keras.losses.MeanAbsoluteError()
    -
  * - mape
    - tf.keras.losses.MeanAbsolutePercentageError()
    -
  * - mape
    - tf.keras.metrics.MeanAbsolutePercentageError()
    -
  * - msle
    - tf.keras.losses.MeanSquaredLogarithmicError()
    -
  * - kldiv
    - tf.keras.losses.KLDivergence()
    -
  * - cosine
    - tf.keras.losses.CosineSimilarity()
    -
