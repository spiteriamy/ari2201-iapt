import fasttext
import langid
from langid.langid import LanguageIdentifier, model
import csv
import json

class CodeSwitchingDetector:
    def __init__(self):
        # load models
        self.ft_model = fasttext.load_model("lid.176.bin")
        self.li_model = LanguageIdentifier.from_modelstring(model, norm_probs=True)

        # language classification for individual words
        self.word_classification = dict()

    def classify_word(self, word: dict): # classify the language of a word
        token = word["token"]
        root = word["root"]

        classification = {'token': token, 
                          'root_available': root != 'null', # true or false
                          'fasttext': {}, # lang and confidence
                          'langid': {}, # lang and confidence
                          'final_classification': ''} # mt or non_mt
        
        # root based classification - check if the root exists, if so, classify as maltese
        if root != "null":
            classification['final_classification'] = 'mt'

        # fasttext classification
        ft_pred, ft_conf = self.ft_model.predict(token)
        ft_lang = ft_pred[0].replace('__label__', '')
        classification['fasttext'] = {'language': ft_lang, 
                                      'confidence': float(ft_conf[0])}

        # langid classification
        li_lang, li_conf = self.li_model.classify(token)
        classification['langid'] = {'language': li_lang, 
                                    'confidence': float(li_conf)}

        # final classification
        if classification['final_classification'] == '':
            if (classification['fasttext']['language'] == 'mt' or 
                classification['langid']['language'] == 'mt'):
                classification['final_classification'] = 'mt'
            else: 
                classification['final_classification'] = 'non_mt'

        # to save the word classification
        self.word_classification[token] = classification['final_classification']

        return classification

    def detect_codeswitching_points(self, sentence: list[dict]): 
        # detect the points in a sentence were code-switching occurs
        switching_points = [] # a list of indices where code switching happens
        prev_lang = None
        has_cs = False # if there is any code-switching

        for i, word in enumerate(sentence):
            current_classification = self.classify_word(word)
            current_lang = current_classification['final_classification']

            if prev_lang and prev_lang != current_lang: # there is a change between mt or non_mt
                has_cs = True # change in language - so there codeswitching
                switching_points.append({'position': i, 
                                         'from': prev_lang, 
                                         'to': current_lang,
                                         'token': word['token']})
                
            prev_lang = current_lang

        return has_cs, switching_points

    def analyse_sentence(self, sentence: list[dict]):
        sentence_text = ' '.join([w['token'] for w in sentence])
        
        # sentence level language detection
        ft_sent_pred, ft_sent_conf = self.ft_model.predict(sentence_text)
        ft_sent_lang = ft_sent_pred[0].replace('__label__', '')
        
        li_sent_lang, li_sent_conf = self.li_model.classify(sentence_text)

        # token level language detection
        token_classifications = []
        for word in sentence:
            classification = self.classify_word(word)
            final_classification = classification['final_classification']
            token_classifications.append(final_classification)

        # code switching detection
        has_cs, switching_points = self.detect_codeswitching_points(sentence)

        # final sentence information
        sentence_info = {'sentence_text': sentence_text, 
                         'sentence_level': {'fasttext': {'lang': ft_sent_lang, 'conf': float(ft_sent_conf[0])}, 
                                            'langid': {'lang': li_sent_lang, 'conf': float(li_sent_conf)}}, 
                         'token_classifications': token_classifications, 
                         'has_codeswitching': has_cs,
                         'switching_points': switching_points}
        
        return sentence_info
    
    def analyse_corpus(self, sentences: list, max_sentences: int = None):
        # analyse all the sentences in the corpus
        if max_sentences:
            sentences = sentences[:max_sentences]

        analyses = []
        total_sentences = len(sentences)

        print(f"Analysing {total_sentences} sentences...")

        for i, sentence in enumerate(sentences):
            analysis = self.analyse_sentence(sentence)
            analyses.append(analysis)

        return analyses

    def save_results(self, analyses: list[dict], name: str): # save the corpus analysis to a file
        # save to csv
        csv_filename = f'sentence_details_{name}.csv'
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # header
            writer.writerow([
                'sentence_id', 'sentence_text', 
                'fasttext_sentence_lang', 'fasttext_sentence_conf',
                'langid_sentence_lang', 'langid_sentence_conf', 
                'token_classifications',
                'has_codeswitching', 'switching_points',   
            ])

            # data rows
            for i, analysis in enumerate(analyses):
                sent_level = analysis['sentence_level']
                
                writer.writerow([
                    i + 1,
                    analysis['sentence_text'],
                    sent_level['fasttext']['lang'],
                    sent_level['fasttext']['conf'],
                    sent_level['langid']['lang'],
                    sent_level['langid']['conf'],
                    json.dumps(analysis['token_classifications']),
                    analysis['has_codeswitching'],
                    json.dumps(analysis['switching_points'])
                ])

        print(f"Sentence information saved to: {csv_filename}")

        # save to json
        json_filename = f'sentence_details_{name}.json'
        with open(json_filename, 'w') as f:
            json.dump(analyses, f)

        print(f"Sentence information saved to: {json_filename}")

        words_filename = f'word_classifications_{name}.json'
        with open(words_filename, 'w') as f:
            json.dump(self.word_classification, f)

        print(f'Word information saved to: {words_filename}')

  