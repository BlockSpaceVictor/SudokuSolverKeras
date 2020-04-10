import numpy as np 
import pandas as pd
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
import random
import matplotlib.pyplot as plt



# sudokuSmall.csv contains 150 puzzles and solutions 
# sudoky.csv contains 1 million puzzles and solutions 
# sudokuMedium.csv conains 5300 puzzles and solutions
# sudokuBig.csv contains 432032 puzzles and solutions
# sudokus_test.csv has 4000 puzzles and solutions 

def their_data(filename, size):
	DATA_SIZE = size

	quizzes = np.zeros((DATA_SIZE, 81), np.int32)
	solutions = np.zeros((DATA_SIZE, 81), np.int32)
	for i, line in enumerate(open(filename, 'r').read().splitlines()[1:]):
	    quiz, solution = line.split(",")

	    for j, q_s in enumerate(zip(quiz, solution)):
	        q, s = q_s
	        quizzes[i, j] = q
	        solutions[i, j] = s
	quizzes = quizzes.reshape((-1, 9, 9))
	solutions = solutions.reshape((-1, 9, 9))
	return quizzes, solutions

def my_data(filename, size):
	DATA_SIZE = size

	quizzes = np.zeros((DATA_SIZE, 81), np.int32)
	solutions = np.zeros((DATA_SIZE, 81), np.int32)
	for i, line in enumerate(open(filename, 'r').read().splitlines()[1:]):
	    quiz, solution = line.split(",")
	    quiz = quiz[1:-1]
	    solution = solution[2:-1]

	    for j, q_s in enumerate(zip(quiz, solution)):
	        q, s = q_s
	        quizzes[i, j] = q
	        solutions[i, j] = s
	quizzes = quizzes.reshape((-1, 9, 9))
	solutions = solutions.reshape((-1, 9, 9))
	return quizzes, solutions


#qmillion, smillion = my_data("mixedSudokusBetter.csv", 1000000)
q80, s80 = my_data("s_make80.csv", 5000)
q78, s78 = my_data("s_make78.csv", 5000)
q75, s75 = my_data("s_make75.csv", 50000)
# q70, s70 = my_data("s_make70.csv", 5000)
# q65, s65 = my_data("s_make65.csv", 5000)
# q60, s60 = my_data("s_make60.csv", 5000)
# q47, s47 = my_data("s_make47.csv", 10000)
# q57, s57 = my_data("s_make57.csv", 10000)
# q59, s59 = my_data("s_make59.csv", 10000)
# q65b, s65b = my_data("s_make65b.csv", 10000)
# q66, s66 = my_data("s_make66.csv", 10000)
# q67, s67 = my_data("s_make67.csv", 10000)
# q68, s68 = my_data("s_make68.csv", 10000)
# q69, s69 = my_data("s_make47.csv", 10000)
# qbig, sbig = their_data("sudokuBig.csv", 432032)

SEED = 42

print("Done with file reading")

mega_quiz = np.concatenate((q75,q80,q78), axis=0)
mega_sol = np.concatenate((s75,s80,s78), axis=0)

print("Done with concatenate")

np.random.seed(SEED)
np.random.shuffle(mega_quiz)
np.random.seed(SEED)
np.random.shuffle(mega_sol)

print("Done with mega shuffle")


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
#n_quizzes = normalize(quizzes)
#n_solutions = normalize(solutions)
#set 2:
n_qmega = normalize(mega_quiz) 
n_smega = normalize(mega_sol)

print("Done with normalize")

# n_q80, n_s80 = normalize(q80), normalize(s80)
# n_q78, n_s78 = normalize(q78), normalize(s78)
# n_q75, n_s75 = normalize(q75), normalize(s75)
# n_q70, n_s70 = normalize(q70), normalize(s70)
# n_q65, n_s65 = normalize(q65), normalize(s65)
# n_q60, n_s60 = normalize(q60), normalize(s60)


#print(n_quizzes.shape)
#print(n_solutions.shape)

## setup the models:
model1 = tf.keras.Sequential()
model2 = tf.keras.Sequential()
model3 = tf.keras.Sequential()
model4 = tf.keras.Sequential()
model5 = tf.keras.Sequential()
model6 = tf.keras.Sequential()
model7 = tf.keras.Sequential()
model8 = tf.keras.Sequential()
model9 = tf.keras.Sequential()
model10 = tf.keras.Sequential()

## Build Model Layers 

## Test1: Increase Fatness of Model
model1.add(layers.Dense(81, input_shape=(9,9), activation='linear', bias_initializer='random_uniform'))	
model1.add(layers.Dense(32, activation='linear', bias_initializer='random_uniform'))
model1.add(layers.Dense(18, activation='linear', bias_initializer='random_uniform'))			
model1.add(layers.Dense(9, activation='linear'))	

model2.add(layers.Dense(162, input_shape=(9,9), activation='linear', bias_initializer='random_uniform'))	
model2.add(layers.Dense(81, activation='linear', bias_initializer='random_uniform'))		
model2.add(layers.Dense(32, activation='linear', bias_initializer='random_uniform'))	
model2.add(layers.Dense(9, activation='linear'))	

model3.add(layers.Dense(243, input_shape=(9,9), activation='linear', bias_initializer='random_uniform'))	
model3.add(layers.Dense(81, activation='linear', bias_initializer='random_uniform'))	
model3.add(layers.Dense(32, activation='linear', bias_initializer='random_uniform'))		
model3.add(layers.Dense(9, activation='linear'))	

model4.add(layers.Dense(324, input_shape=(9,9), activation='linear', bias_initializer='random_uniform'))	
model4.add(layers.Dense(162, activation='linear', bias_initializer='random_uniform'))	
model4.add(layers.Dense(81, activation='linear', bias_initializer='random_uniform'))		
model4.add(layers.Dense(9, activation='linear'))	

model5.add(layers.Dense(405, input_shape=(9,9), activation='linear', bias_initializer='random_uniform'))	
model5.add(layers.Dense(162, activation='linear', bias_initializer='random_uniform'))
model5.add(layers.Dense(81, activation='linear', bias_initializer='random_uniform'))			
model5.add(layers.Dense(9, activation='linear'))	

model6.add(layers.Dense(486, input_shape=(9,9), activation='linear', bias_initializer='random_uniform'))	
model6.add(layers.Dense(243, activation='linear', bias_initializer='random_uniform'))
model6.add(layers.Dense(81, activation='linear', bias_initializer='random_uniform'))			
model6.add(layers.Dense(9, activation='linear'))	

model7.add(layers.Dense(567, input_shape=(9,9), activation='linear', bias_initializer='random_uniform'))	
model7.add(layers.Dense(243, activation='linear', bias_initializer='random_uniform'))
model7.add(layers.Dense(81, activation='linear', bias_initializer='random_uniform'))		
model7.add(layers.Dense(9, activation='linear'))	

model8.add(layers.Dense(648, input_shape=(9,9), activation='linear', bias_initializer='random_uniform'))	
model8.add(layers.Dense(243, activation='linear', bias_initializer='random_uniform'))	
model8.add(layers.Dense(81, activation='linear', bias_initializer='random_uniform'))		
model8.add(layers.Dense(9, activation='linear'))	

model9.add(layers.Dense(729, input_shape=(9,9), activation='linear', bias_initializer='random_uniform'))	
model9.add(layers.Dense(243, activation='linear', bias_initializer='random_uniform'))
model9.add(layers.Dense(81, activation='linear', bias_initializer='random_uniform'))			
model9.add(layers.Dense(9, activation='linear'))	

model10.add(layers.Dense(810, input_shape=(9,9), activation='linear', bias_initializer='random_uniform'))	
model10.add(layers.Dense(324, activation='linear', bias_initializer='random_uniform'))	
model10.add(layers.Dense(162, activation='linear', bias_initializer='random_uniform'))		
model10.add(layers.Dense(45, activation='linear', bias_initializer='random_uniform'))	
model10.add(layers.Dense(9, activation='linear'))	
		

opt = keras.optimizers.Adam(lr=0.001)

## Compile Models
model1.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
model2.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
model3.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
model4.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
model5.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
model6.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
model7.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
model8.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
model9.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
model10.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

#print(model1.summary())

## Fit Models:

history1 = model1.fit(n_qmega, n_smega, epochs=3, batch_size=500, validation_split=0.001)
history2 = model2.fit(n_qmega, n_smega, epochs=3, batch_size=500, validation_split=0.001)
history3 = model3.fit(n_qmega, n_smega, epochs=3, batch_size=500, validation_split=0.001)
history4 = model4.fit(n_qmega, n_smega, epochs=3, batch_size=500, validation_split=0.001)
history5 = model5.fit(n_qmega, n_smega, epochs=3, batch_size=500, validation_split=0.001)
history6 = model6.fit(n_qmega, n_smega, epochs=3, batch_size=500, validation_split=0.001)
history7 = model7.fit(n_qmega, n_smega, epochs=3, batch_size=500, validation_split=0.001)
history8 = model8.fit(n_qmega, n_smega, epochs=3, batch_size=500, validation_split=0.001)
history9 = model9.fit(n_qmega, n_smega, epochs=3, batch_size=500, validation_split=0.001)
history10 = model10.fit(n_qmega, n_smega, epochs=3, batch_size=500, validation_split=0.001)

# ## "Predict always takes a list... [] hmmm"
# n_predictions = model1.predict(n_qmega)

# #n_predictions2 = model.predict(n_quizzes2)

# #testvar = random.randint(0,3000)

# print("given: ")
# print(reverse_normalize(n_qmega[44]))
# print("prediction: ")
# print(np.round(reverse_normalize(n_predictions[44]), decimals=0))
# print("solution: ")
# print(reverse_normalize(n_smega[44]))
# print("2: ")
# print("given: ")
# print(reverse_normalize(n_qmega[266]))
# print("prediction: ")
# print(np.round(reverse_normalize(n_predictions[266]), decimals=0))
# print("solution: ")
# print(reverse_normalize(n_smega[266]))

# history_dict = history.history
# print(history_dict.keys())


# ## Get Historgram of value occurences: 

# preds = np.round(reverse_normalize(n_predictions), decimals=0)
# uniqueValues, occurCount = np.unique(preds, return_counts=True)

# sols = np.round(reverse_normalize(n_smega),decimals=0)
# uniqueValuesS, occurCountS = np.unique(sols, return_counts=True)

# print()
# print("Unique Values in Solution: ", uniqueValues)
# print("Occurence Count in Solution: ", occurCount)
# print()
# print("Unique Values in Prediction: ", uniqueValues)
# print("Occurence Count in Prediction: ", occurCount)
# print()

# #histogramS = plt.bar(uniqueValuesS, occurCountS)
# #plt.xlabel('Distribution of Digits in Sudoku Solutions')
# #plt.ylabel('Frequency')
# #plt.show()
# histogramP = plt.bar(uniqueValues, occurCount, )
# plt.xlabel('Distribution of Digits in Sudoku Solutions')
# plt.ylabel('Frequency')
# plt.show()







# # Plot training & validation accuracy values
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()







