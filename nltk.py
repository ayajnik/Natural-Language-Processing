import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import nltk.corpus
import os
#data importing
#print (os.listdir(nltk.data.find("corpus")))
#accessing one of the datasets
#hamlet = nltk.corpus.gutenberg.words('shakespere-hamlet.txt')
#hamlet
#for i in hamlet[:500]:
#    print (i, sep=" ", end = " ")
#workingon this data now
AI = '''In computer science, artificial intelligence (AI), sometimes called machine intelligence, is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Colloquially, the term "artificial intelligence" is used to describe machines that mimic "cognitive" functions that humans associate with other human minds, such as "learning" and "problem solving".[1]

As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect.[2] A quip in Tesler's Theorem says "AI is whatever hasn't been done yet."[3] For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.[4] Modern machine capabilities generally classified as AI include successfully understanding human speech,[5] competing at the highest level in strategic game systems (such as chess and Go),[6] autonomously operating cars, intelligent routing in content delivery networks, and military simulations.

Artificial intelligence can be classified into three different types of systems: analytical, human-inspired, and humanized artificial intelligence.[7] Analytical AI has only characteristics consistent with cognitive intelligence; generating cognitive representation of the world and using learning based on past experience to inform future decisions. Human-inspired AI has elements from cognitive and emotional intelligence; understanding human emotions, in addition to cognitive elements, and considering them in their decision making. Humanized AI shows characteristics of all types of competencies (i.e., cognitive, emotional, and social intelligence), is able to be self-conscious and is self-aware in interactions with others.

Artificial intelligence was founded as an academic discipline in 1956, and in the years since has experienced several waves of optimism,[8][9] followed by disappointment and the loss of funding (known as an "AI winter"),[10][11] followed by new approaches, success and renewed funding.[9][12] For most of its history, AI research has been divided into subfields that often fail to communicate with each other.[13] These sub-fields are based on technical considerations, such as particular goals (e.g. "robotics" or "machine learning"),[14] the use of particular tools ("logic" or artificial neural networks), or deep philosophical differences.[15][16][17] Subfields have also been based on social factors (particular institutions or the work of particular researchers).[13]

The traditional problems (or goals) of AI research include reasoning, knowledge representation, planning, learning, natural language processing, perception and the ability to move and manipulate objects.[14] General intelligence is among the field's long-term goals.[18] Approaches include statistical methods, computational intelligence, and traditional symbolic AI. Many tools are used in AI, including versions of search and mathematical optimization, artificial neural networks, and methods based on statistics, probability and economics. The AI field draws upon computer science, information engineering, mathematics, psychology, linguistics, philosophy, and many other fields.

The field was founded on the claim that human intelligence "can be so precisely described that a machine can be made to simulate it".[19] This raises philosophical arguments about the nature of the mind and the ethics of creating artificial beings endowed with human-like intelligence which are issues that have been explored by myth, fiction and philosophy since antiquity.[20] Some people also consider AI to be a danger to humanity if it progresses unabated.[21] Others believe that AI, unlike previous technological revolutions, will create a risk of mass unemployment.[22]

In the twenty-first century, AI techniques have experienced a resurgence following concurrent advances in computer power, large amounts of data, and theoretical understanding; and AI techniques have become an essential part of the technology industry, helping to solve many challenging problems in computer science, software engineering and operations research.'''

print(type(AI))
#tokenizing the file
from nltk.tokenize import word_tokenize
AI_tokenize = word_tokenize(AI)
print(AI_tokenize)
print (len(AI_tokenize))
#finding the wordcount
from nltk.probability import FreqDist
fDist = FreqDist()
for i in AI_tokenize:
    fDist[i.lower()]+=1
print(fDist)
fDist_top10 = fDist.most_counts(10) #top 10words
print(fDist_top10)
#bigrams,trigrams,ngrams
from nltk.util import bigrams, trigrams,ngrams
quotes_bigrams = list(nltk.bigrams(AI))
print (quotes_bigrams)
quotes_trigrams = list(nltk.trigrams(AI))
print (quotes_trigrams)
quotes_ngrams = list(nltk.ngrams(AI, 4))
print (quotes_ngrams)
#stemming
from nltk.stem import PorterStemmer
prt=PorterStemmer()
prt.stem("having")
#lancasterstemmer
from nltk.stem import LancasterStemmer
lst=LancasterStemmer()
for i in AI:
    print (i+":"+lst.stem(i))
#lemmatization
from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
word_lem = WordnetLemmatizer()
for i in AI:
    print (i+":"+word_lem.lemmatize(i))
#parts of speech
#let us consider a string and we will identify which word is noun, adjective etc.
sent ="Ayush is natural when it comes to python."
#first tokenizing
sent_tokenize = word_tokenize(sent)
#now passing the parts of speech function to identift the words
for token in sent_tokenize:
    print(nltk.pos_tag([token]))
#name, place animal thing recognition
from nltk import ne_chunk
NE_sent = "The US President stays in the WHITE HOUSE."
NE_tokens = word_tokenize(NE_sent)
NE_tags = nltk.pos_tag(NE_tokens)
NE_ner = ne_chunk(NE_tags)
print(NE_ner)

