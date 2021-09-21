Document Classification ->
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
