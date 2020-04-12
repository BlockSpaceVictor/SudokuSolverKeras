import numpy as np 
import pandas as pd
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import random
import matplotlib.pyplot as plt


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
q30k, s30k = my_data("30k_mixedSudokus75_80.csv", 30000)
q80, s80 = my_data("s_make80.csv", 5000)
#q78, s78 = my_data("s_make78.csv", 5000)
#q75, s75 = my_data("s_make75.csv", 50000)
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
#q1m, s1m = my_data("mixedSudokus4m.csv", 1000000)

SEED = 42

print("Done with file reading")

mega_quiz = np.concatenate((q80, q30k), axis=0)
mega_sol = np.concatenate((s80, s30k), axis=0)

SIZE = len(mega_quiz)

print("Done with concatenate")

np.random.seed(SEED)
np.random.shuffle(mega_quiz)
np.random.seed(SEED)
np.random.shuffle(mega_sol)

print("Done with mega shuffle")

n_qmega = mega_quiz / 9.
n_smega = mega_sol / 9.

print("Done with normalize")

## add extra dimension: 
n_qmega = np.expand_dims(n_qmega, -1)
n_smega = n_smega.reshape(SIZE, 81)  

print("added dimension")

#learning rate scheduler simple:
def scheduler(epoch):
	if epoch < 10:
		return 0.1
	elif epoch < 20:
		return 0.01
	elif epoch < 30:
		return 0.001
	elif epoch < 40:
		return 0.0005
	elif epoch < 50:
		return 0.0002
	elif epoch < 100:
		return 0.0001
	elif epoch < 300:
		return 0.00005
	elif epoch < 700:
		return 0.00002
	else:
		return 0.00001

## BUILD CNN: 
i = Input(shape=n_qmega[0].shape)
x = Conv2D(16, (3,3), strides=1, activation='relu',padding='same')(i)
x = Conv2D(32, (9,1), strides=1, activation='relu',padding='same')(x)
x = Conv2D(64, (1,9), strides=1, activation='relu',padding='same')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(81, activation="relu")(x)

model = Model(i, x)

opt = keras.optimizers.Adam(lr=0.1, clipnorm=1.)
opt2 = keras.optimizers.SGD(lr=0.01, nesterov=True)
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

## compile and fit model: 
model.compile(optimizer = "adam",
	loss="mse",
	metrics=["accuracy"])

print(model.summary())

r = model.fit(n_qmega, n_smega, epochs=50, callbacks=[callback], batch_size=500, validation_split=0.01)


n_predictions = model.predict(n_qmega) # ideally, this should be either a subset, or a different set of quizzes

# reshape to original 9 X 9 np array:
n_qmega = n_qmega.reshape(SIZE, 9, 9)
n_predictions = n_predictions.reshape(SIZE,9,9)

## reverse the normalization: 
n_qmega = n_qmega * 9
n_predictions = n_predictions * 9
n_smega = n_smega * 9
n_smega = n_smega.reshape(SIZE,9,9)

##Check random puzzles:
rand1 = random.randint(0,500)
rand2 = random.randint(1000, 2000)

print("given: ")
print(np.round(n_qmega[rand1], decimals=0))
print("prediction: ")
print(np.round(n_predictions[rand1], decimals=0))
print("solution: ")
print(np.round(n_smega[rand1], decimals=0))
print("2: ")
print("given: ")
print(np.round(n_qmega[rand2], decimals=0))
print("prediction: ")
print(np.round(n_predictions[rand2], decimals=0))
print("solution: ")
print(np.round(n_smega[rand2], decimals=0))


# Check Range of Values in Predictions:
preds = np.round(n_predictions, decimals=0)
uniqueValues, occurCount = np.unique(preds, return_counts=True)

print("Unique Values in Prediction: ", uniqueValues)
print("Occurence Count in Prediction: ", occurCount)
print()


## Plot Range of Values in Prediction
histogramP = plt.bar(uniqueValues, occurCount, )
plt.xlabel('Distribution of Digits in Sudoku Solutions')
plt.ylabel('Frequency')
plt.show()

## Plot model loss over time
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Plot model accuracy over time
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()


## SUDOKU SOLUTION CHECKING LOGIC:

#master functions to check a sudoku:
def isSolved(sudoku):
	return (check_rows(sudoku) and check_cols(sudoku) and check_blocks(sudoku))

def check_rows(arr):
	for l in arr:
		my_list = []
		for i in l:
			my_list.append(i)
		if checkIfDuplicates_1(my_list) or sum(my_list) != 45:
			return False
	return True

def check_cols(arr):
	trans = arr.transpose()
	return check_rows(trans)

def func(arr, h, h2, e, e2):
	my_list = [] 
	for i in range(h, h2):
		for j in range(e, e2):
			my_list.append(arr[i][j])
	if checkIfDuplicates_1(my_list) or sum(my_list) != 45:
		return False
	return True

def check_blocks(arr):
	if(func(arr, 0,3,0,3) and
		func(arr, 0,3,3,6) and
		func(arr, 0,3,6,9) and
		func(arr, 3,6,0,3) and
		func(arr, 3,6,3,6) and
		func(arr, 3,6,6,9) and
		func(arr, 6,9,0,3) and
		func(arr, 6,9,3,6) and
		func(arr, 6,9,6,9)):
		return True
	else:
		return False

def checkIfDuplicates_1(listOfElems):
    if len(listOfElems) == len(set(listOfElems)):
        return False
    else:
        return True


## TRUE ACCURACY CHECK
## Check random set of predictions and whether they are correctly solved sudokus: 

checkLength = 1000
booleanList = []
for i in range(0,checkLength):
	booleanList.append(isSolved(preds[random.randint(0,SIZE)]))

trues = booleanList.count(True)
total_accuracy = trues / checkLength * 100

print("#####################################################################")
print()
print("ACCURACY: ")
print("About ", total_accuracy, " percent of predicted solutions are correct")
print(trues, " of ", checkLength, " predicted sudokus were solved correctly")
print()
print("#####################################################################")












