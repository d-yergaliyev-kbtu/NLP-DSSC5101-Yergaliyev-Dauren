from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis')

# Test with some sentences
sentences = [
    "I love this product!",
    "I hate waiting in long lines.",
    "The movie was okay, not great but not bad."
]

# Analyze sentiment
results = sentiment_analyzer(sentences)

# Display the results
for sentence, result in zip(sentences, results):
    print(f"Sentence: {sentence}")
    print(f"Sentiment: {result['label']} (Confidence: {result['score']:.2f})\n")
