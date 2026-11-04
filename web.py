from code_switching_detector import CodeSwitchingDetector
from corpus import load_web

# load detector
print("Loading detector...")
detector = CodeSwitchingDetector()

# load sentences
print("\nLoading corpus...")
web_sentences = load_web()

# analyse sentences for code-switching
print("\nStarting analysis...")
analyse_web = detector.analyse_corpus(web_sentences, max_sentences=5000)

# save results to file
print("\nSaving results...")
detector.save_results(analyse_web, 'web')
