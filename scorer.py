# This program calculates the accuracy of sentiment.py by comparing the generated answers to the key. It then outputs the accuracy and confusion matrix .
# 
# To use: type 'python3 scorer.py [name of document containing answers] sentiment-test-key.txt' into the terminal. 


import sys
from sklearn.metrics import confusion_matrix
import pandas as pd
import re

def main(): 

    # opens and reads the tagged test data
    test = open(sys.argv[1])
    test_string = test.read()

    # tokenizes the test data
    test_array = test_string.split('\n')
    test_array = list(filter(None, test_array))

    # opens and reads the key
    key = open(sys.argv[2])
    key_string = key.read()

    # tokenizes the key
    key_array = key_string.split('\n')
    key_array = list(filter(None, key_array))

    correct = 0
    total = 0

    predicted_array = []
    actual_array = []

    predicted_sentiment = ''
    actual_sentiment = ''

    sentiments = ['negative', 'positive']

    # loops through each element of the test array and compares the predicted sense with the actual sense from the key
    for element in test_array: 
        
        # captures the id and the predicted sense
        match = re.match(r'<answer instance="(.*)" sentiment="(.*)"/>', element)
        id = match.group(1)
        predicted_sentiment = match.group(2)

        # loops through the key and looks for the matching sense
        for element2 in key_array: 
            if (id in element2): 
                actual_sentiment = re.match(r'.*sentiment="(.*)".*', element2).group(1) # captures the actual sense
                break
        
        # compares the two senses
        if (actual_sentiment == predicted_sentiment):
            correct += 1
        total += 1
        
        # appends each of the senses to an array (either predicted_sense or actual_sense) to create the confusion matrix
        predicted_array.append(predicted_sentiment)
        actual_array.append(actual_sentiment) 

    
    # calculates the accuracy
    accuracy = correct/total
    print(accuracy)

    # creates the confusion matrix
    matrix = confusion_matrix(actual_array, predicted_array, labels = sentiments)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    matrix = pd.DataFrame(matrix, index=sentiments, columns=sentiments)

    print(str(matrix))

if __name__ == '__main__':
    main()
