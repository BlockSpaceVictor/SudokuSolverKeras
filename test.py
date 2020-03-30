import numpy as np 
import pandas as pd
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
import random



# sudokuSmall.csv contains 150 puzzles and solutions 
# sudoky.csv contains 1 million puzzles and solutions 
#sudokuMedium.csv conains 5300

quizzes = np.zeros((5300, 81), np.int32)
solutions = np.zeros((5300, 81), np.int32)
for i, line in enumerate(open('sudokuMedium.csv', 'r').read().splitlines()[1:]):
    quiz, solution = line.split(",")
    for j, q_s in enumerate(zip(quiz, solution)):
        q, s = q_s
        quizzes[i, j] = q
        solutions[i, j] = s
quizzes = quizzes.reshape((-1, 9, 9))
solutions = solutions.reshape((-1, 9, 9))

DATA_SIZE = 5300

## these are numpy arrays that look like this: 
# [[8 6 4 3 7 1 2 5 9]
#  [3 2 5 8 4 9 7 6 1]
#  [9 7 1 2 6 5 8 4 3]
#  [4 3 6 1 9 2 5 8 7]
#  [1 9 8 6 5 7 4 3 2]
#  [2 5 7 4 8 3 9 1 6]
#  [6 8 9 7 3 4 1 2 5]
#  [7 1 3 5 2 8 6 9 4]
#  [5 4 2 9 1 6 3 7 8]]

#print(quizzes[0])
#print(solutions[0])


def normalize(np_arr):
	np_arr = np.divide(np_arr, 9)
	return np_arr

def reverse_normalize(np_arr):
	np_arr = np.multiply(np_arr, 9)
	return np_arr

#print(normalize(quizzes[0]))
#print(reverse_normalize(normalize(quizzes[0])))

# normalize data:
n_quizzes = normalize(quizzes)
n_solutions = normalize(solutions)


## setup the model:
model = tf.keras.Sequential()
model.add(layers.Dense(81, input_shape=(9,9), activation='relu'))
model.add(layers.Dense(256, activation='relu'))	
model.add(layers.Dense(128, activation='relu'))	
# model.add(layers.Dense(64, activation='softmax'))	
# model.add(layers.Dense(32, activation='softmax'))	
# model.add(layers.Dense(820, activation='softmax'))	
# model.add(layers.Dense(820, activation='softmax'))	
# model.add(layers.Dense(820, activation='softmax'))	
# model.add(layers.Dense(256, activation='softmax'))	
model.add(layers.Dense(32, activation='relu'))	
model.add(layers.Dense(9, activation='softmax'))			

opt = SGD(lr=0.01)
model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(model.summary())


model.fit(n_quizzes, n_solutions, epochs=20, batch_size=30, validation_split=0.1)


n_predictions = model.predict(n_quizzes)

testvar = random.randint(0,DATA_SIZE)

print(reverse_normalize(n_predictions[testvar]))
print(reverse_normalize(n_solutions[testvar]))






