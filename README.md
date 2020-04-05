# SudokuSolverKeras

An attempt to teach python to solve sudoku puzzles without knowing the rules of sudoku

I'm using Tensorflow Keras to set up the neural network (testing with Sequential model)

The sudoku puzzles and solutions are downloaded from: 
https://www.kaggle.com/bryanpark/sudoku/data#

But can also be generated using the code from this website: 
https://www.ocf.berkeley.edu/~arel/sudoku/main.html

The 1 million sudokus from Kaggle turned out to not be super useful because they contain mainly zeroes. For the algorithm to learn anything, there is not enough data available to make a useful inference as to how to arrive at a solution. 

I tweaked the sudokuMake algorithm to create sudokus in a range of difficulties, with some boards missing only a few numbers, while others are mainly empty. To create a million more sudokus like this, I had to use multiprocessing, and an AWS EC2 instance with 16 CPU cores, and it still took 3 hours to generate the puzzles...

