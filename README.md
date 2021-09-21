* Course: Elements of AI(SP21-BL-CSCI-B551-37653)
* CS B551 - Assignment 2: Games
* Name: Bharath Kumar Maturi, Jonathan Satish Tirupuranthakam, Sravya Garaga
* UserName: bkmaturi-josatiru-sgaraga
* GIT repo: bkmaturi-josatiru-sgaraga-a2

----------------------------------------------------------------------------
Part 1: Pikachu ->
--------------------

## Search abstraction:
*Initial state*: The given board containing white and black pichus and pikachus.
*Terminal State*: If all the pichus or pikachus of any of the player are zero, that is the terminal state. It is the position of the board when the game gets over.
*Successor states*: any board which is obtained after a valid move of pichu or pikachu
*Evaluation function*: 
For w turn:
(2 * no of white pikachus + no of white pichus) – (2 * no of black pikachus + no of black pichus)
For b turn:
(2 * no of black pikachus + no of black pichus) – (2 * no of white pikachus + no of white pichus)

## Design decisions and working:
*Different functions have been written to calculate the successors for the given player. According to our functions a pichu will be able to move one step forward, left and right if the square is empty and it can move two steps forward if there is an opponent in between and the next square is empty. A pikachu moves forward, backward, left and right any number of squares and jumps over the opponent and then moves a square, if the squares between the Pikachu and the opponent are empty and the one next to the opponent is also empty. all these functions that define each move of the player have been called in a function successors().
*A successor function to return all the possible successors for the player ‘w’ or ‘b’.The list_successors_w has the list of successors of w and list_successors_b has the list of successors of b. the list of successors are appended in such a way that each successor is checked if it is not the same as the board passed as a parameter.
*We used minimax algorithm with alpha beta pruning and iterative deepening search tree to form the leaf nodes and then back track the values from the depth upwards using the evaluation function applied to the leaf nodes till the root node and then the algorithm gives the best move considering the alpha and beta values of the max and min players respectively. We assign the default alpha beta values to the root, we then carry these values alpha and beta to the child node on the left. And now from the utility value of the terminal state, we will update the values of alpha. Thus alpha beta pruning is done and all the successors with the payoffs are pushed into the stack. The best move or the successor with max value is popped out. Which is the ultimate best move for the player and new board.
*The max value and min value of the successors are being calculated by using the functions:
def max_v(successor, alpha, beta, depth, player, depthlimit,N):
def min_v(successor, alpha, beta, depth, player, depthlimit,N):
these functions are used to calculate the alpha and beta values for min successors and max successors
*there is a terminal state or end state which returns true if the number of w’s or b’s is 0.If the depth reaches the limit then it starts calculating the payoffs using the evaluation function.
*An evaluation function also known as heuristic evaluation function is used to estimate the value or goodness of a position of the player in a particular game. In the evaluation function, the pikachus are given a value of 2 and pichus are given 1. We calculated the payoff to be ((2 * count_W) + (count_w) - (2 * count_B) - (count_b)) for w’s turn. and the payoff to be ((2 * count_B) + (count_b) - (2 * count_W) - (count_w)) for b’s turn.

### Assumptions :
*We took the default values of alpha and beta to be 100000000 and -100000000.
*We assumed the weights of pikachus to be 2 and weights of pichus to be 1 and calculated the weighted features for b and w

### Difficulties:
*We faced difficulty in setting the limits for the row and columns as we have converted the given string to a 2d array. In order to iterate through the rows and columns for the edge cases we had to pass different boards if the output was correctly updating or not.
*We faced hurdle to figure out how to set the time limit and how to run the minimax algorithm such that it stops at the particular leaf node according to the time limit given and how to back track the alpha and beta values to the root node and apply the alpha beta pruning.

********-------------------------------------********

----------------------------------------------------------------------------
Part 2: The Game of Sebastian ->
--------------------

## Probelm Formulation

* The problem here is to write the algorithm for getting maximum score after rolling 5 dice and re-rolling specific dice/s "2 more times" to increase the score.

* One game will have 13 turns. In problem statement, 13 categories has been given and we need to assign a category only once can't assigned again in 13 turns 

Here, we used Expect minimax algorithm to get the best score. 

* the key idea we have used is to select all combinations of dices to re-roll and for each combination we have calculated the expected value. 
We have considered the combination which has maximum expected value and continued the process. 

We have given two classes(Sebastian, SebastianState). 
Our task is to implement Sebastian AutoPlayer class which will return the dice numbers that needs to be re-rolled. 

## Our Algorithm Description
* We have created two functions (cat_score, expected_score) 
* cat_score will take the dices output and returns the best category we can assign for that output.
* expected_score will take the combination of dice to be re-rolled and will returns the combination which has maximum expected value. 
* We have used there two functions and we returned the dices to be re-rolled in first re-roll (function: first_roll(self, dice, scorecard)) 
and second re-roll (function: second_roll(self, dice, scorecard)). 
* function "third_roll(self, dice, scorecard)" will return the best category we can assign after second re-roll to main function (Sebastian.py). 
* After playing 100 games of 13 turns in each gaem, it will calculate the Minimum, Maximum and Mean scores of 100 games. 

## Assumptions, Difficulties.

* We have assumed that if we are unable to assign category for a combination, we assigned score as "0(zero)" for that combination. 
* We have faced dificulties while assigning the best category for each time in 13 turns of one game. 
* We have overcome that issue by declaring a dictionary in constructor of SebastianAutoPlayer class which will store the categories those are already assigned. 
So by doing this, our algorithm will assign the next best category if best category is already assigned in previous turns.

********-------------------------------------********

----------------------------------------------------------------------------
Part3: Document Classification ->
--------------------

## Problem Formulation
* The idea is that to use the Naive Bayes Classification for telling whether it belongs to which category.

* As per the problem statement we need to calculate the P(spam|word in a sentence) or P(not spam|word in ) on the testing data using training data:

P(spam|words in a sentence)=(P(words in a sentence|spam)*P(spam))/P(spam)

P(not spam|words in a sentence)=(P(words in a sentence|not spam)*P(not spam))/P(not spam)

* For calculating the likelihood probabilities, we used the Multinomial Naive Bayes algorithm.

The Multinomial Naive Bayes algorithm implements the naive Bayes algorithm for multinomially distributed data, and is one of the two classic Naive Bayes variant used for text classification.

We took reference from the following website: http://scikit-learn.org/stable/modules/naive_bayes.html

According to Multinomial Naive Bayes algorithm,
likelihood(word_i/spam or not spam) = (N_yi + alpha)/(N_y + alpha*n)

where
N_yi = Number of times that particular word repeated
n= number of unique words
N_y = sum of N_yi
alpha is used to smoothing the priors

* So after calculating the above Bayes formula we will send to the output which ever is greater after normalization.

* We apply normalization for making the sum of probabilities to 1, thought it's not necessary.


## Our Algorithm Description

* Basically we just implemented the method which is mentioned above using the formula given.

* We have used dictionaries for storing the number of times a word gets repeated by considering each word as a key.

* We have calculated the prior probability of spam or not spam and we didn't calculate the independent probability of words as it is common denominator during comparison and won't have significance in decision.

* We then calculated the likelihood function and multiplied it with the prior value for spam and not spam.

* We compared both P(spam|words) and P(not spam|words) and assigned the category which has the greater probability.

* Then the program returns the accuracy considering the list we returned.

## Assumptions
* The smoothing priors  accounts for features not present in the learning samples and prevents zero probabilities in further computations. 
* Setting  is called Laplace smoothing, while  is called Lidstone smoothing.
* We assumed the alpha value to be 0.3 as we are getting max accuracy for that particular value upon trying different values.
