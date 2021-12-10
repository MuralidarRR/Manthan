import cv2
import pytesseract
import shutil
import nltk
nltk.download('stopwords')
import os
import random

try:
  from PIL import Image

except ImportError:

  import Image
from nltk.util import pr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import csv
from operator import itemgetter
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
import streamlit as st
import warnings


data = pd.read_csv('labeled_data.csv')


data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
#print(data.head())

data = data[["tweet", "labels"]]
#print(data.head())

import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)
#print(data.head())

x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x,y) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)



def hate_speech_detection():
    
    import streamlit as st

    st.sidebar.title("Hate Content Detection")
    
    select=st.sidebar.selectbox('Run Analysis on',['', 'Text','Image'],key=1)
    
    st.sidebar.title("Hate Content Analysis in Youtube")

    select1=st.sidebar.selectbox('Select Type' , ['Comments' , 'Video'],key=2)



    if select =='':
      st.write("")

    elif select == "Text":

      
      st.title("Hate Speech Detection in Text")
      user = st.text_area("Enter any Tweet: ")
      Text = st.button("Predict Text")

      if Text:
        
        if len(user) < 1:
          st.write("  ")
        else:
          sample = user
          data = cv.transform([sample]).toarray()
          a = clf.predict(data)
          st.write(a)
      
      st.write(Text)

    else:
      st.title("Hate Speech Detection in Image")
      
      file = st.file_uploader("Upload file" , type =["csv", "png" , "jpg"])
      show_file = st.empty()

      if not file:
        show_file.info("Please upload a file")
        return
      
      content = file.getvalue()

      if(file):
        extractedInformation = pytesseract.image_to_string(Image.open(file))
        sample = extractedInformation
        data = cv.transform([sample]).toarray()
        a = clf.predict(data)
        st.write(a)
      
      else:
        df = pd.read_csv(file)
        st.dataframe(df.head(2))
      
      file.close()

   
    if select1 == 'Comments':


        st.title("Hate Speech Analysis in Comments")

        api_key="AIzaSyA4I_AmRa5PqmMDJ5U5gGJsq8Wntf5FbaM"

        #api_key = "AIzaSyA-1me5_f4JGSZ45tmy6lHgGbwpYC8AMTo" # Replace this dummy api key with your own.

        from apiclient.discovery import build
        youtube = build('youtube', 'v3', developerKey=api_key , cache_discovery = False)


        ID=st.text_area("Enter video ID")
        analyze = st.button("Analyze")

        if(analyze):



          box = [['Name', 'Comment', 'Time', 'Likes', 'Reply Count']]



            
          data = youtube.commentThreads().list(part='snippet', videoId=ID, maxResults='100', textFormat="plainText").execute()

          for i in data["items"]:

                  name = i["snippet"]['topLevelComment']["snippet"]["authorDisplayName"]
                  comment = i["snippet"]['topLevelComment']["snippet"]["textDisplay"]
                  published_at = i["snippet"]['topLevelComment']["snippet"]['publishedAt']
                  likes = i["snippet"]['topLevelComment']["snippet"]['likeCount']
                  replies = i["snippet"]['totalReplyCount']

                  box.append([name, comment, published_at, likes, replies])

                  totalReplyCount = i["snippet"]['totalReplyCount']

                  if totalReplyCount > 0:

                      parent = i["snippet"]['topLevelComment']["id"]

                      data2 = youtube.comments().list(part='snippet', maxResults='100', parentId=parent, textFormat="plainText").execute()

                      for i in data2["items"]:
                          name = i["snippet"]["authorDisplayName"]
                          comment = i["snippet"]["textDisplay"]
                          published_at = i["snippet"]['publishedAt']
                          likes = i["snippet"]['likeCount']
                          replies = ""

                          box.append([name, comment, published_at, likes, replies])

                  while ("nextPageToken" in data):

                    data = youtube.commentThreads().list(part='snippet', videoId=ID, pageToken=data["nextPageToken"],
                                                      maxResults='100', textFormat="plainText").execute()

                  for i in data["items"]:
                      name = i["snippet"]['topLevelComment']["snippet"]["authorDisplayName"]
                      comment = i["snippet"]['topLevelComment']["snippet"]["textDisplay"]
                      published_at = i["snippet"]['topLevelComment']["snippet"]['publishedAt']
                      likes = i["snippet"]['topLevelComment']["snippet"]['likeCount']
                      replies = i["snippet"]['totalReplyCount']

                      box.append([name, comment, published_at, likes, replies])

                      totalReplyCount = i["snippet"]['totalReplyCount']

                      if totalReplyCount > 0:

                          parent = i["snippet"]['topLevelComment']["id"]

                          data2 = youtube.comments().list(part='snippet', maxResults='100', parentId=parent, textFormat="plainText").execute()

                          for i in data2["items"]:
                              name = i["snippet"]["authorDisplayName"]
                              comment = i["snippet"]["textDisplay"]
                              published_at = i["snippet"]['publishedAt']
                              likes = i["snippet"]['likeCount']
                              replies = ''

                              box.append([name, comment, published_at, likes, replies])

              
            

          dq = pd.DataFrame({ 'Comment': [i[1] for i in box]})
          #dq

          def cleanTxt(text):
            text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
            text = re.sub('#', '', text) # Removing '#' hash tag
            text = re.sub('RT[\s]+', '', text) # Removing RT
            text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
          
            return text


          # Clean the tweets
          dq['Comment'] = dq['Comment'].apply(cleanTxt)

          # Show the cleaned tweets
          #st.write(dq)



          #Create a function to get the subjectivity
          def getSubjectivity(text):
            return TextBlob(text).sentiment.subjectivity

          # Create a function to get the polarity
          def getPolarity(text):
            return  TextBlob(text).sentiment.polarity


          # Create two new columns 'Subjectivity' & 'Polarity'
          dq['Subjectivity'] = dq['Comment'].apply(getSubjectivity)
          dq['Polarity'] = dq['Comment'].apply(getPolarity)

          # Show the new dataframe with columns 'Subjectivity' & 'Polarity'
          #dq

          # word cloud visualization
          allWords = ' '.join([twts for twts in dq['Comment']])
          wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)


          plt.imshow(wordCloud, interpolation="bilinear")
          plt.axis('off')
          plt.show()

          st.subheader("Most Commonly used words in the comment section")
          st.pyplot()

          # Create a function to compute negative (-1), neutral (0) and positive (+1) analysis
          
          def getAnalysis(score):
            if score < 0:
              return 'Negative'
            elif score == 0:
              return 'Neutral'
            else:
              return 'Positive'
              
          dq['Analysis'] = dq['Polarity'].apply(getAnalysis)
          # Show the dataframe
          st.write(dq)




          # Plotting and visualizing the counts
          plt.title('Sentiment Analysis')
          plt.xlabel('Sentiment')
          plt.ylabel('Counts')
          dq['Analysis'].value_counts().plot(kind = 'bar')
          plt.show()

          st.pyplot()
            
            
hate_speech_detection()




