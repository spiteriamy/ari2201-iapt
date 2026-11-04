import os
import re

# when loading the corpus we are interested
# in the word itself and the morphological
# root: first field and last field of each line

# get each sentence as a list of tokens and roots
# sentence: [{'token': token1, 'root': root1}, {'token': token2, 'root': root2}, ...]


# Load the blogs corpus
def load_blogs():
    path = os.path.join('Corpus', 'Blogs') # get path to data
    return get_sentences(path)

# Load the web corpus
def load_web():
    path = os.path.join('Corpus', 'Web') # get path to data
    return get_sentences(path)


# Get the sentences from the corpus
# return a list of all the sentences
def get_sentences(path):
    sentences = []
    current_sentence = []
    
    # go through all the files
    for file_path in os.listdir(path):
        with open(os.path.join(path, file_path), encoding="utf-8") as file:
            # get the lines
            lines = file.readlines()

            # process lines
            for line in lines:
                # clean the line
                clean_line = re.sub(r'<(?!(s id="\d+"|/s))[^>]+>', '', line) # Remove tags except <s> and </s>
                clean_line = clean_line.strip() # Remove extra whitespace
                clean_line = clean_line.replace('\u200b', '') # space character remove

                if not clean_line or clean_line == '<s>': # if the line is empty or started a new sentence
                    if current_sentence: # skip empty sentences
                        sentences.append(current_sentence) # add the sentence to list
                        current_sentence = [] # reset current sentence for next one
                    continue

                # get the fields from the line
                # there should be 4
                fields = clean_line.split()
                if len(fields) == 4:
                    token, pos, lemma, root = fields
                    current_sentence.append({'token': token, 'root': root})

    return sentences

