import tensorflow as tf
print("TensorFlow version:", tf.__version__)

#load dataset
mnist = tf.keras.datasets.mnist

#images are up to 255 pixels, so they're converted to a value between 0-1 by dividing by 255
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#build the ML model
#sequential is good for stacking layers that have 1 input and output tensor (multi dimensional array)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# model returns a vector of logits (non normalized predictions) for each class
predictions = model(x_train[:1]).numpy() for each class
predictions

#converts the logits to probabilities 
tf.nn.softmax(predictions).numpy()

#define loss funciton for training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#takes a vector of ground truth (ones we know are correct/true) values and logits and returns a scalar loss for each example 
#loss = negative log probability of true class. 0 means model is sure of correct class
# untrained models will give a value close to 2.3
loss_fn(y_train[:1], predictions).numpy()

#configure and compile model
model.compile(optimizer='adam',
              loss=loss_fn, #set loss to the loss function aleady defined 
              metrics=['accuracy']) #specify metric to evaluate the model by

#adjust model parameters and minimize loss
model.fit(x_train, y_train, epochs=5)

#check the models performance, usually against validation or test set
model.evaluate(x_test,  y_test, verbose=2)

#to return a probaility, wrap trained model and attach softmax to it
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
probability_model(x_test[:5])

