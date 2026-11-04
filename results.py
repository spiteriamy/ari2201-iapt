import pandas as pd
import json

# blogs

# load info
sentences_details_blogs = pd.read_csv('sentence_details_blogs.csv')
sentences_details_blogs['has_codeswitching'] = sentences_details_blogs['has_codeswitching'].astype(bool)

# calculate percentage of sentences that contain code-switching
counts = sentences_details_blogs['has_codeswitching'].value_counts(normalize=True) * 100
cs_percentage = counts.get(True, 0)

print(f"Percentage of sentences in Blogs that contain code-switching: {cs_percentage:.2f}%")

# calculate percentage of maltese words
with open('word_classifications_blogs.json', 'r') as f:
    words = json.load(f)

total = len(words)
mt_count = sum(1 for v in words.values() if v == 'mt')
mt_percentage = (mt_count / total) * 100

print(f"Percentage of Maltese words in Blogs: {mt_percentage:.2f}%")


# web

# load info
sentences_details_web = pd.read_csv('sentence_details_web.csv')
sentences_details_web['has_codeswitching'] = sentences_details_web['has_codeswitching'].astype(bool)

# calculate percentage of sentences that contain code-switching
counts = sentences_details_web['has_codeswitching'].value_counts(normalize=True) * 100
cs_percentage = counts.get(True, 0)

print(f"Percentage of sentences in Web that contain code-switching: {cs_percentage:.2f}%")

# calculate percentage of maltese words
with open('word_classifications_web.json', 'r') as f:
    words = json.load(f)

total = len(words)
mt_count = sum(1 for v in words.values() if v == 'mt')
mt_percentage = (mt_count / total) * 100

print(f"Percentage of Maltese words in Web: {mt_percentage:.2f}%")

