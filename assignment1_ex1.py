import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

text = "Kazakh-British Technical University (KBTU) is a public university located in Almaty, Kazakhstan. It was founded in 2001. KBTU’s research is focused in the main sectors of the Kazakhstani economy – oil and gas, information technologies, banking and finance, management and telecommunications."

nltk_tokens = word_tokenize(text, language='english', preserve_line=True)

lemmatizer = WordNetLemmatizer()
nltk_lemmas = [lemmatizer.lemmatize(word) for word in nltk_tokens]

nltk_stopwords = set(stopwords.words('english'))
nltk_filtered = [word for word in nltk_lemmas if word.lower() not in nltk_stopwords]

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

spacy_tokens = [token.text for token in doc]

spacy_lemmas = [token.lemma_ for token in doc]

spacy_filtered = [token.lemma_ for token in doc if not token.is_stop]

print("--- NLTK ---")
print("Tokens:", nltk_tokens)
print("Lemmatized:", nltk_lemmas)
print("Stopword Removed:", nltk_filtered)

print("\n--- spaCy ---")
print("Tokens:", spacy_tokens)
print("Lemmatized:", spacy_lemmas)
print("Stopword Removed:", spacy_filtered)