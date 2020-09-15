import glob, os

full_vocabulary = []

os.chdir("vocab")
for file_ in glob.glob("*.txt"):
    full_path = os.path.join(file_)
    f = open(full_path, 'r')
    for line in f:
        full_vocabulary.append(line.rstrip('\n'))

full_vocabulary_final = list(set(full_vocabulary))
f1 = open('final_vocabulary.txt', 'a+')
for x in full_vocabulary_final:
    f1.write(x+'\n')
f1.close()


