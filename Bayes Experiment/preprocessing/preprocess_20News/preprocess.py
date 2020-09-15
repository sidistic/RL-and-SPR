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



_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                       r'|^In article|^Quoted from|^\||^>)')



def updateVocabulary (filename):
    listOfWords = []
    with open(filename, 'r') as f:
        for line in f:
            listOfWords.append(line.strip('\n'))
    f.close()
    stop_words = stopwords.words('english')

    new_ListOfWords = []

    for i in tqdm(range(len(listOfWords))):
        if listOfWords[i] not in stop_words:
            new_ListOfWords.append(listOfWords[i])
    
    with open(filename, 'w') as f1:
        for x in new_ListOfWords:
            temp = x + '\n'
            f1.write(temp)
    f1.close()    

def createDict():
    
    listOfWords = []
    with open('new_vocabulary.txt', 'r') as f:
        for line in f:
            listOfWords.append(line.strip('\n'))
    f.close()
    tempDict = dict([(key, 0) for key in listOfWords])
    # print (tempDict)//
    # print (type(tempDict))
    return tempDict

def resetDict(dictForReset):
    dictForReset_ = dictForReset.fromkeys(dictForReset, 0)
    # print (dictForR   eset_)
    return dictForReset_

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
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
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
    # print (filtered_sentence, '\n\n\n')
    return filtered_sentence

def updateCounter(dictOfWords, listOfWords):
    for x in listOfWords:
        if x in dictOfWords:
            counter = dictOfWords.get(x, '')
            # print (x, counter,)
            counter += 1
            # print (counter)
            d1 = {x: counter}
            dictOfWords.update(d1)
    return dictOfWords

def mainLoop():
    dictGeneral = createDict()
    arr = np.empty((0,len(dictGeneral)), int)
    # print ('Initial shape', arr.shape)
    f_classes = open("classes_newrandomperson.txt","a+")
    # this is the main loop which runs for all the files through the folders
    path = 'data/' # set path
    if os.path.isdir(path):
        for file_ in os.listdir(path): # loop through the folders in the data folder
            if os.path.isdir(os.path.join(path,file_)):
                print ('Doing', file_, 'now!\n\n')
                for textFile in tqdm(os.listdir(os.path.join(path,file_))): # looping through the files present in the folder
                    # print (textFile)
                    dictGeneral = resetDict(dictGeneral)
                    WordsFromFile = getWordsFromFile(os.path.join(path,file_,textFile))
                    updatedDict = updateCounter(dictGeneral, WordsFromFile)
                    arr = np.vstack([arr, getNumpyArrayFromDict(updatedDict)])
                    # print ('This is the one in between', arr.shape)
                    f_classes.write(file_+'\n')
    print ('\n\n\n\n', arr.size)
    np.save('x_data_newrandomperson.npy', arr)
            
                    
def getNumpyArrayFromDict(dictUpdated):
    temp = np.array(list(dictUpdated.values())).reshape(1,len(dictUpdated))
    # print ('This is from the function', temp.shape)
    return temp

mainLoop()



