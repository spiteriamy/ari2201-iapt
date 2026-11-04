from code_switching_detector import CodeSwitchingDetector
from corpus import load_blogs

# load detector
print("Loading detector...")
detector = CodeSwitchingDetector()

# load sentences
print("\nLoading corpus...")
blogs_sentences = load_blogs()

# analyse sentences for code-switching
print("\nStarting analysis...")
analyse_blogs = detector.analyse_corpus(blogs_sentences, max_sentences=5000)

# save results to file
print("\nSaving results...")
detector.save_results(analyse_blogs, 'blogs')
