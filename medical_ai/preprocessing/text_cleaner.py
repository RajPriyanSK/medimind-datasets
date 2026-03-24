import re
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

import nltk
# Check if punkt/stopwords are available, otherwise download silently
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextCleaner:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spacy en_core_web_sm...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Cleans and tokenizes text for ML/NLP usage."""
        if not text:
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove special characters except dots, slashes and colons (needed for numbers/dates)
        text = re.sub(r'[^a-z0-9\.\:\/\s%]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def tokenize_and_remove_stopwords(self, text):
        clean = self.clean_text(text)
        tokens = word_tokenize(clean)
        
        filtered_tokens = [w for w in tokens if w not in self.stop_words and w not in string.punctuation]
        return filtered_tokens

    def extract_medical_keywords(self, text):
        """Uses spaCy to extract potential entities."""
        doc = self.nlp(text)
        keywords = []
        for ent in doc.ents:
            keywords.append((ent.text, ent.label_))
        
        return keywords

if __name__ == "__main__":
    cleaner = TextCleaner()
    sample = "Patient's Blood Glucose is 120 mg/dL, and BP is 130/85 mmHg."
    print("Cleaned:", cleaner.clean_text(sample))
    print("Tokens:", cleaner.tokenize_and_remove_stopwords(sample))
