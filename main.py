from corpus import load_blogs, load_web
import fasttext
import langid
from langid.langid import LanguageIdentifier, model

# load fasttext model
ft_model = fasttext.load_model("lid.176.bin")

# langid model
li_model = LanguageIdentifier.from_modelstring(model, norm_probs=True)

# get the sentences from the corpus
# blogs_sentences = load_blogs()
web_sentences = load_web()

# identify the language used both at
# a sentence level and token level

# blogs corpus


# web corpus

for i, sentence in enumerate(web_sentences):
    sentence_str = ' '.join([w['token'] for w in sentence])
    print(f'\nSentence {i+1}: {sentence_str}')

    # classify language sentence level
    label_ft_sentence, conf_ft_sentence = ft_model.predict(sentence_str)
    label_ft_sentence = label_ft_sentence[0].replace('__label__', '')
    label_li_sentence, conf_li_sentence = li_model.classify(sentence_str)
    print(f'fasttext -> {label_ft_sentence} ({conf_ft_sentence[0]*100:.2f}%)')
    print(f'langid -> {label_li_sentence} ({conf_li_sentence*100:.2f}%)')

    for word in sentence: 
        token = word["token"]
        root = word["root"]

        # check if the root exists
        # if so, classify it as maltese
        if root != "null":
            label_root = 'mt'
        else:
            label_root = 'null'

        # check with fasttext library
        label_ft, conf_ft = ft_model.predict(token)
        label_ft = label_ft[0].replace('__label__', '')

        # check with langid library
        label_li, conf_li = li_model.classify(token)

        if label_root == 'mt' or label_ft == 'mt' or label_li == 'mt':
            classification = 'mt'
        else:
            classification = 'non_mt'

        print(f'{token:15} | root: {label_root:10} | fasttext: {label_ft:10} | langid: {label_li:10} | classification: {classification}')


# issues: the libraries are not very accurate at identifying at a token level
# a bit better on a sentence level


