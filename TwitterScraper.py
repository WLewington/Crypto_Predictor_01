# import tweepy library for twitter api access and textblob libary for sentiment analysis
import csv
import tweepy
import numpy as np
from textblob import TextBlob
import MLSentEngine as MLSent
import string
import datetime as dt
import time
localTime = time.asctime(time.localtime(time.time()))
print(" -------------- ENTRY AT " + str(localTime) + " ---------------- ")


# ------ Twitter Scraper("StringA", .. ,"StringC", "LX" or "ML" sentAnalysis types)
def ScrapeTwitter(SubjectA, SubjectB, SubjectC, SentMethod):
    #region - - - - - Twitter API Credentials - - - - - 
    consumer_key = "HX7LWWyl8dSCE9E8KJFZtCJWf"
    consumer_secret = "6LI7LM3xOX0QRenF3jZ13Kl66yf7wkWCkdZBvqi5J7hcaMmI5D"
    access_token = "964240989085929475-y73iQUuBlBn9Olv53nbuakOblUtgmZP"
    access_token_secret = "Q7vrB6ODzPUBThWJwKVeaj0HBE9OF56ajg6UAAfgQFfnC"

    # - - - Accesssing twitter via Tweepy
    auth  = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    twitter_api = tweepy.API(auth)
    #endregion
    # - - - File name & Directory
    fileName = open('Data/Tweet{}Sent_{}&{}&{}.csv'.format(SentMethod,
                                                             SubjectA,
                                                             SubjectB,
                                                             SubjectC),"a")
    i = 0
    while True:
        i += 1
        # - - - Searching twitter by Subjects 
        tweets = twitter_api.search(q=[SubjectA, SubjectB, SubjectC], count=4)
        # - - - Getting Polarity From Either LX or ML method
        print(tweets)
        print(type(tweets))
        if SentMethod == "LX":
            polarity = LXSentAnalysis(tweets)
        elif SentMethod == "ML":
            polarity = MLSentAnalysis(tweets)
        else:
            print("- - - Sentiment Method not found - - - ")
            break
        # - - - Averaging tweets sentiment.
        sentiment = np.mean(polarity)
        # - - - save sentiment data to csv file
        fileName.write(str(sentiment))
        # - - - Creating time stamp from datetime.NOW
        fileName.write(","+dt.datetime.now().strftime("%y-%m-%d %H:%M"))
        fileName.write("\n")
        fileName.flush()
        print(" - - - - Itteration: " + str(i) + " - - - - ")
        time.sleep(60)  
        

# LexiconSentAnalysis("text to analyse", File)
def LXSentAnalysis(tweets):
    # empty list for each tweets Polarity
    tweet_polarity = []
    # - - - Loops through list of tweets
    for tweet in tweets:
        # - - - Splits the tweet into words - - - 
        words = tweet.text.split()
        # - - Removes Punctuation
        wordsList = str.maketrans('','', string.punctuation.replace(".","")) 
        stripped = [w.translate(wordsList) for w in words]
        # - Reassembles words into string of text 
        strippedTweets = " ".join(stripped)
        # - writes the Average sentiment
        # file.write(strippedTweets+'\n')
        # - - - Lexicon Sentiment Analysis
        analysis = TextBlob(strippedTweets)
        tweet_polarity.append(analysis.sentiment.polarity)
        print(strippedTweets)
        print(tweet_polarity)
    return tweet_polarity

def MLSentAnalysis(tweets):
    # empty list for each tweets Polarity
    tweet_polarity = []
    # - - - Loops through list of tweets
    for tweet in tweets:
        # - - - Splits the tweet into words - - - 
        words = tweet.split()
        # - - Removes Punctuation
        wordsList = str.maketrans('','', string.punctuation.replace(".","")) 
        stripped = [w.translate(wordsList) for w in words]
        # - Reassembles words into string of text 
        strippedTweets = " ".join(stripped)
        # - - - Machine Learning Sentiment Analysis
        analysis = MLSent.sentiment(strippedTweets)
        tweet_polarity.append(analysis)
    return(tweet_polarity)

# ScrapeTwitter('Ethereum',"Crypto","price", "ML")
