import sys
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import numpy as np
import re
import string
from stopwords import final_stopWords
import operator

topWords=500

# temp_file = open('stopwords.txt', 'r')
# final_stopWords = [line.rstrip('\n') for line in temp_file]
# print (final_stopWords)

_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                       r'|^In article|^Quoted from|^\||^>)')


def strip_newsgroup_header(text):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.
    """
    _before, _blankline, after = text.partition('\n\n')
    return after

def strip_newsgroup_quoting(text):
    """
    Given text in "news" format, strip lines beginning with the quote
    characters > or |, plus lines that often introduce a quoted section
    (for example, because they contain the string 'writes:'.)
    """
    good_lines = [line for line in text.split('\n')
                  if not _QUOTE_RE.search(line)]
    return '\n'.join(good_lines)

def strip_newsgroup_footer(text):
    """
    Given text in "news" format, attempt to remove a signature block.

    As a rough heuristic, we assume that signatures are set apart by either
    a blank line or a line made of hyphens, and that it is the last such line
    in the file (disregarding blank lines at the end).
    """
    lines = text.strip().split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip('-') == '':
            break

    if line_num > 0:
        return '\n'.join(lines[:line_num])
    else:
        return text

def removePunctuation(newsText):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~+=|'''
    for x in newsText.lower(): 
        if x in punctuations: 
            newsText = newsText.replace(x, ' ')
    for x in newsText:
        if x in '''0123456789''':
            newsText = newsText.replace(x, ' ')
    return newsText

def processText(newsText):
    newsText = str(newsText, 'latin1')
    newsText = newsText.lower()
    newsText = strip_newsgroup_header(newsText)
    newsText = strip_newsgroup_footer(newsText)
    newsText = strip_newsgroup_quoting(newsText)
    newsText = removePunctuation(newsText)
    return newsText

def getWordsFromFile(filename):
    with open(filename, 'rb') as f:
        newsText = f.read()
    newsText = processText(newsText)
    # print (newsText, '\n\n\n')
    word_tokens = word_tokenize(newsText)
    filtered_sentence = [w for w in word_tokens if not w in final_stopWords]
    # print (filtered_sentence)
    return filtered_sentence

def createVocabulary():
    path = 'data/' # set path
    for file_ in os.listdir(path):

        # This is the loop for the folders...
        
        f = open('vocab/vocab_'+file_+'.txt', 'a+')
        if os.path.isdir(os.path.join(path,file_)):
            print ('Doing', file_, 'now!\n')
            dictionary = dict()
            for textFile in tqdm(os.listdir(os.path.join(path,file_))):
                # print (textFile)
                wordsFromFile = getWordsFromFile(os.path.join(path,file_,textFile))
                for word in wordsFromFile:
                    if word in dictionary.keys():
                        counter = dictionary.get(word, '')
                        counter += 1
                        d1 = {word: counter}
                        dictionary.update(d1)
                    else:
                        dictionary[word] = 1
        sorted_dict = sorted(dictionary.items(), key=operator.itemgetter(1))
        sorted_dict.reverse()
        count = 0
        for x in sorted_dict:
            count += 1
            if count>topWords:
                break
            # print (x[0])
            f.write(x[0]+'\n')
        print ('Finished', file_, '\n\n\n')
        f.close()


# Append all the files

createVocabulary()
            

