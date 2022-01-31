import json
import nltk
import scispacy   
import spacy
import en_core_sci_sm 
import en_core_web_sm
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

nlp = en_core_web_sm.load()
from spacy import displacy 
import pandas as pd
from nltk.corpus import stopwords

# Installing the spacy library
#pip install -U spacy
#pip install scispacy
#pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz
nltk.download('stopwords')
stop_words=set(stopwords.words("english"))
paragraphs = list()
 
# Opening JSON file
f = open('CORD-19.json')
 
# returns JSON object as
# a dictionary
data = json.load(f)
 
# Iterating through the json
# list
for i in data['abstract']:
    paragraphs.append(i)
 
# Closing file
f.close()
print ('********************* Question1********************') 
print("                               ")
print ("This is the abstract")
print("                               ")
for i in range(len(paragraphs)):
    print (paragraphs[i]['text'])
    print("                               ")
print("                               ")
def remove_stop_wordss(paragraphs):
    filtered = []
    words = [word for word in paragraphs.split() if word.lower() not in stop_words]
    new_text = " ".join(words)
    return new_text
new_paragraphs = list()
for i in range(len(paragraphs)):
    new_paragraphs.append(remove_stop_wordss(paragraphs[i]['text']))
print ('                                                   ')
print ('********************* Question2********************') 
print("                               ") 
print ('The Abstract after removing the stop words') 
print("                               ")
print(new_paragraphs[0])
print("                               ")
print(new_paragraphs[1])
print("                               ")
print(new_paragraphs[2])
print("                               ")
print(new_paragraphs[3])
print("                               ")
print(new_paragraphs[4])
print("                               ")

nltk.download('maxent_ne_chunker')
nltk.download('words')
def preprocess(sent):
    tokens = nltk.word_tokenize(sent)
    cp = nltk.ne_chunk(nltk.pos_tag(tokens), binary=True)
    named_entities=[]
    for t in cp.subtrees():
        if t.label() == "NE":
            named_entities.append((t))
    return (named_entities)
Entity_1 = preprocess(paragraphs[0]['text'])
Entity_2 = preprocess(paragraphs[1]['text'])
Entity_3 = preprocess(paragraphs[2]['text'])
Entity_4 = preprocess(paragraphs[3]['text'])
Entity_5 = preprocess(paragraphs[4]['text'])
print ('                                                   ')
print ('********************* Question3********************') 
print ('                                                   ')
#print ('The Named Entities with Allennlp and spacy')
print ('The Named Entity using NLTK of the Abstracts are listed below')
print(Entity_1)
print("                               ")
print(Entity_2)
print("                               ")
print(Entity_3)
print("                               ")
print(Entity_4)
print("                               ")
print(Entity_5)
print("                               ")

#import spacy
from spacy import displacy

NER = spacy.load("en_core_web_sm")
text= NER(paragraphs[0]['text'])
text1= NER(paragraphs[1]['text'])
text2= NER(paragraphs[2]['text'])
text3= NER(paragraphs[3]['text'])
text4= NER(paragraphs[4]['text'])
print ('                                                   ')
print ('********************* Question4********************') 
print ('                                                   ')
print ('The Named Entities with Allennlp and spacy') 
for word in text.ents:
   
    print("                               ")
    print(word.text,word.label_)
for word in text1.ents:
    print(word.text,word.label_)
    print("                               ")
for word in text2.ents:
    print(word.text,word.label_)
    print("                               ")
for word in text3.ents:
    print(word.text,word.label_)
    print("                               ")
for word in text4.ents:
    print(word.text,word.label_)
    print("                               ")
from allennlp.predictors.predictor import Predictor
import allennlp_models.classification
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/basic_stanford_sentiment_treebank-2020.06.09.tar.gz")
#sentence= "This is a really good story."
score = predictor.predict(paragraphs[0]['text'])
score_1 = predictor.predict(paragraphs[1]['text'])
score_2 = predictor.predict(paragraphs[2]['text'])
score_3 = predictor.predict(paragraphs[3]['text'])
score_4 = predictor.predict(paragraphs[4]['text'])
print ('********************* Question5********************') 
print ('                                                   ')
print ('                                                   ')
#print ('The Named Entity using NLTK of the Abstracts are listed below')
print ('The sentiments of the Abstracts are listed below')
print ('                                                   ')
print("Score: +ve ", score['probs'][0], " -ve ", score['probs'][1])
print("Score: +ve ", score_1['probs'][0], " -ve ", score_1['probs'][1])
print("Score: +ve ", score_2['probs'][0], " -ve ", score_2['probs'][1])
print("Score: +ve ", score_3['probs'][0], " -ve ", score_3['probs'][1])
print("Score: +ve ", score_4['probs'][0], " -ve ", score_4['probs'][1])