## Step 1
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

## Step 2
colors_df = pd.DataFrame(data=[['red'],['blue'],['green'],['blue']], columns=['color'])
print('Before One Hot Encoding:')
display(colors_df)

one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoder.fit(colors_df)

colors_df_encoded = one_hot_encoder.transform(colors_df)
colors_df_encoded = pd.DataFrame(data=colors_df_encoded, columns=one_hot_encoder.categories_)
print('\n\nAfter One Hot Encoding:')
display(colors_df_encoded)
## Step 3 these numbers correspond to Step 2 colors_df
labels = [0,1,2,1]

linear_regression = LinearRegression()
pipeline = Pipeline(steps=[('one_hot_encoder',one_hot_encoder), ('linear_regression', linear_regression)])
pipeline.fit(colors_df, labels)

print('Red Prediction:', pipeline.predict([['red']])[0])
print('Blue Prediction:', pipeline.predict([['blue']])[0])

## Step 4
category_indices = [0, 1, 2, 2, 1, 0]
unique_category_count = 3
inputs = tf.one_hot(category_indices, unique_category_count)
print(inputs.numpy())

## Step 5

text_vectorization = layers.experimental.preprocessing.TextVectorization(output_sequence_length=1)
text_vectorization.adapt(colors_df.values)

print('Red index:', text_vectorization.call([['red']]))
print('Blue index:', text_vectorization.call([['blue']]))
print('Green index:', text_vectorization.call([['green']]))

print(text_vectorization.get_vocabulary()) # prints [b'blue', b'red', b'green']

## Step 6
class OneHotEncodingLayer(layers.experimental.preprocessing.PreprocessingLayer):
    def __init__(self, vocabulary=None, depth=None, minimum=None):
        super().__init__()
        self.vectorization = layers.experimental.preprocessing.TextVectorization(output_sequence_length=1)

        if vocabulary:
            self.vectorization.set_vocabulary(vocabulary)
        self.depth = depth
        self.minimum = minimum

    def adapt(self, data):
        self.vectorization.adapt(data)
        vocab = self.vectorization.get_vocabulary()
        self.depth = len(vocab)
        indices = [i[0] for i in self.vectorization([[v] for v in vocab]).numpy()]
        self.minimum = min(indices)

    def call(self,inputs):
        vectorized = self.vectorization.call(inputs)
        subtracted = tf.subtract(vectorized, tf.constant([self.minimum], dtype=tf.int64))
        encoded = tf.one_hot(subtracted, self.depth)
        return layers.Reshape((self.depth,))(encoded)

    def get_config(self):
        return {'vocabulary': self.vectorization.get_vocabulary(), 'depth': self.depth, 'minimum': self.minimum}

    ## Step 7
colors_df = pd.DataFrame(data=[[5,'yellow'],[1,'red'],[2,'blue'],[3,'green'],[4,'blue'],[7,'purple']], columns=['id', 'color'])

categorical_input = layers.Input(shape=(1,), dtype=tf.string)
one_hot_layer = OneHotEncodingLayer()
one_hot_layer.adapt(colors_df['color'].values)
encoded = one_hot_layer(categorical_input)

numeric_input = layers.Input(shape=(1,), dtype=tf.float32)

concat = layers.concatenate([numeric_input, encoded])

## Step 8
model = models.Model(inputs=[numeric_input, categorical_input], outputs=[concat])
model.compile()
predicted = model.predict([colors_df['id'], colors_df['color']])
print(predicted)

## Step 9
config = model.get_config()
with tf.keras.utils.custom_object_scope({'OneHotEncodingLayer': OneHotEncodingLayer}):
    loaded_model = tf.keras.Model.from_config(config)

predicted = loaded_model.predict([colors_df['id'], colors_df['color']])
print(predicted)