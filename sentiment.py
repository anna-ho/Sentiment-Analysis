# This program will determine the sentiment of a tweet by identifying features from training data, calculating the loglikelihood of each feature, ranking them, and then
# using a decision list. There are also three additional features used to help predict the sentiment of a tweet, like using bigrams instead of unigrams, expanding acronyms, 
# translating emoticons into their approximate word representation. 
#
# To use: type 'python3 sentiment.py sentiment-train.txt sentiment-test.txt [name of file containing model] > [name of file containing documents]' into the terminal.
# NLTK will need to be downloaded if it isn't already. Directions for that can be found here: https://www.nltk.org/data.html
# 
# Model: a decision list created using features from the tweets that had a log-likelihood greater than 0. In addition to the decision list, three feature types
# were added: a method that expanded acronyms for both the training and test data, a method that converted emoticons into text for both the training and test data, and
# the use of bigrams instead of unigrams when extracting features from the training data for the bag of words.
# 
# Results: 
# Accuracy of most frequent sentiment baseline: 0.6896551724137931
# Accuracy w/ only bag of words: 0.6422413793103449
# Accuracy w/ bag of words and acronym expansion feature: 0.6767241379310345
# Accuracy w/ bag of words and emoticon feature: 0.646551724137931
# Accuracy w/ bag of words made up of bigrams feature: 0.6508620689655172
# Accuracy w/ all features: 0.6551724137931034
# 
# Confusion matrix: 
#           negative  positive
# negative        34        38
# positive        42       118


import math
from operator import truediv
import sys
import re
from nltk.corpus import stopwords

# feature type that looks for acronyms and replaces them with their spelled out equivalent
def acronym_expansion(tweet): 

    # dictionary containing acronyms/shorthand spellings and their replacement
    acronyms = {
        "lol" : "laugh out loud",
        "omg" : "oh my god",
        "jk" : "just kidding",
        "btw" : "by the way",
        "tbh" : "to be honest",
        "ngl" : "not going to lie",
        "bc" : "because",
        "w/e" : "whatever",
        "w/": "with",
        "y" : "why",
        "u" : "you",
        "ur" : "your",
        "r" : "are",
        "yolo" : "you only live once",
        "ty" : "thank you",
        "yw" : "you're welcome",
        "pls" : "please",
        "ppl" : "you will",
        "txt" : "text",
        "fyi" : "for your information",
        "ymmv" : "your mileage my vary",
        "pov" : "point of view",
        "rn" : "right now",
        "rip" : "rest in piece",
        "idk" : "i don't know",
        "aka" : "also known as",
        "rofl" : "rolling on the floor laughing",
        "imo" : "in my opinion",
        "ikr" : "i know right",
        "tmi" : "too much information",
        "obv" : "obviously"
    }
    
    # splits tweets into words
    words = tweet.split()

    updated_tweet = ""

    # loops through each word in the tweet
    for word in words: 
        
        # loops through the acronyms
        for acronym in acronyms:
            
            # if the word matches an acronym, replace it
            if (word == acronym):
                word = acronyms[acronym]
                break
                               
        # strings the words back into a phrase
        if (word != words[-1]):
            updated_tweet += word + " "
        else: 
            updated_tweet += word

    return updated_tweet    

# feature type that looks for emoticons in the data and converts them into a word equivalent
def emoticons_to_text(tweet): 

    emoticons = {
        ":)" : "smile", 
        ":(" : "frown", 
        ":D" : "smile", 
        ";)" : "wink", 
        ":/": "neutral"
    }
    # splits tweets into words
    words = tweet.split()

    updated_tweet = ""

    # loops through each word in the tweet
    for word in words: 
        
        # loops through the acronyms
        for emoticon in emoticons:
            
            # if the word matches an acronym, replace it
            if (word == emoticon):
                word = emoticons[emoticon]
                break
                               
        # strings the words back into a phrase
        if (word != words[-1]):
            updated_tweet += word + " "
        else: 
            updated_tweet += word
    
    return updated_tweet    


def main(): 

    # opens and reads the training data
    train = open(sys.argv[1])
    train_string = train.read().lower()

    # closes the training data
    train.close()

    # gets the list of stop words
    stop_words = set(stopwords.words('english'))
    
    # tokenizes the string based on the end of each instance and removes elements aren't instances
    train_array = train_string.split('</instance>')
    for element in train_array: 
        if 'instance' not in element: 
            train_array.remove(element)

    features = {}
    
    instances_of_negative = 0
    instances_of_positive = 0

    bigram = []

    # loops through each tweet, gets the sentiment, and creates the feature vector and features dictionary
    for instance in train_array: 
                
        # captures the sentiment
        sentiment = re.search(r'.*sentiment="(.*)".*', instance).group(1)
        
        # increments instances of negative sentiment
        if (sentiment == 'negative'): 
            instances_of_negative += 1 
        # increments instances of positive sentiment
        if (sentiment == 'positive'): 
            instances_of_positive += 1 

        # captures the context
        context = re.search(r'.*\n<context>\n(.*)\n<\/context>\n.*', instance).group(1)
        
        # removes links
        context = re.sub(r'(https?:\/\/.[^\s]*)', '', context) 

        # converts emoticons into their text equivalent
        context = emoticons_to_text(context)

        # removes special characters except for hashtags and apostrophes (treats words like i'm, we're, didn't as one word)
        context = re.sub(r'[^A-Za-z0-9\s#\']+', ' ', context) 

        # expands any acronyms
        context = acronym_expansion(context)

        # tokenizes the instance
        context_array = context.split() 
        
        vector = {}

        # feature type that uses bigrams to create the bag of words instead of unigrams
        for word in context_array: 

            bigram.append(word)

            # if the bigram is at the appropriate length, join it into a string and increment its frequency in the vector dictionary
            if (len(bigram) == 2):
            
                phrase = ' '.join(bigram)
                
                if (phrase in vector): 
                    vector[phrase] += 1
                else:
                    vector[phrase] = 1

                bigram.pop(0)
        
        # adds the features to the 2d dictionary features
        for feature in vector: 
            if (feature in features):  
                if (sentiment in features[feature]):  
                    features[feature][sentiment] += 1
                else: 
                    features[feature][sentiment] = 1
            else: 
                features[feature] = {}
                features[feature][sentiment] = 1
    
    # determines the most frequent sentiment
    mfs = ''
    if (instances_of_negative > instances_of_positive): 
        mfs = 'negative'
    else:
        mfs = 'positive'

    features_log_likelihood = {} # dictionary that stores the log likelihood associated with each feature
    features_sentiment = {} # dictionary that stores the sense associated with each feature

    # calculates log likelihood and finds which sentiment appears most frequently with each feature
    for feature in features: 

        sentiment = ""
        
        # checks if the sentiment is present in the dictionary for that feature and sets the frequency to 0.001 if it isn't
        if ('negative' in features[feature]): 
            freq_feature_negative = features[feature]['negative']
        else: 
            freq_feature_negative = 0.001 

        if ('positive' in features[feature]): 
            freq_feature_positive = features[feature]['positive']
        else: 
            freq_feature_positive = 0.001 

        # calculates log_likelihood
        log_likelihood = abs(math.log(freq_feature_negative/freq_feature_positive))

        features_log_likelihood[feature] = log_likelihood

        # the sentiment will be the sentiment with highest frequency 
        # if they appear an equal number of times, the sentiment will be the most frequent sentiment
        if (freq_feature_negative > freq_feature_positive): 
            sentiment = 'negative'
        elif (freq_feature_positive >= freq_feature_negative):
            sentiment = 'positive'

        features_sentiment[feature] = sentiment
    
    # sorts features in descending order based on how discriminatory they are
    features_log_likelihood = dict(sorted(features_log_likelihood.items(), key = lambda x: x[1], reverse = True))

    # removes features that have a log-likelihood of 0
    features_log_likelihood = {key:value for key, value in features_log_likelihood.items() if value > 0}

    # formats the model to be printed
    model = ""
    for feature in features_log_likelihood: 
            
            model += "Feature: " + feature + "\n"
            model += "Log-likelihood: " + str(features_log_likelihood[feature]) + "\n"
            model += "Sentiment: " + features_sentiment[feature] + "\n\n"

    # prints the model
    file_name = sys.argv[3]
    f = open(file_name, "w")
    print(model, file = f) 
    f.close()

    # opens and reads the test data
    test = open(sys.argv[2])
    test_string = test.read().lower()

    # closes the test data
    test.close()
    
    # tokenizes the test string and removes elements that do not include a tweet
    test_array = test_string.split('</instance>')
    for element in test_array: 
        if 'instance' not in element: 
            test_array.remove(element)

    tweet_sentiments = {}

    for instance in test_array: 

        # captures the instance id
        id = re.search(r'.*instance id="(.*)".*', instance).group(1)

        # captures the tweet
        tweet = re.search(r'.*\n<context>\n(.*)\n<\/context>\n.*', instance).group(1)

        # expands acronyms
        tweet = acronym_expansion(tweet)

        # converts emoticons into text
        tweet = emoticons_to_text(tweet)

        # default sentiment is the sentiment that appeared most frequently
        sentiment = mfs

        # loops through feature dictionary and looks for the feature in the tweet
        for feature in features_log_likelihood:

            # if the feature exists, set the sentiment to the sentiment associated with that feature
            # once a feature is found, stop searching for other features
            if feature in tweet: 
                sentiment = features_sentiment[feature]
                break

        tweet_sentiments[id] = sentiment
    
    # formats the answers to be printed
    for element in tweet_sentiments: 
        print("<answer instance=\"" + element + "\" sentiment=\"" + tweet_sentiments[element] + "\"/>")


if __name__ == '__main__':
    main()
