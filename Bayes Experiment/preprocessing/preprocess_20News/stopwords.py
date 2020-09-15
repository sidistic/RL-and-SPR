final_stopWords = []


temp_file = open('stopwords.txt', 'r')
final_stopWords = [line.rstrip('\n') for line in temp_file]
# print (final)