import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

#initilize the nltk
sia=SentimentIntensityAnalyzer()

text="i didn't liked the coffee"
t1=sia.polarity_scores(text)
print(t1)

"""
**Using Textblob**
TextBlob is a simple library for processing textual data, which provides an easy-to-use API for common natural language processing (NLP) tasks including sentiment analysis."""

from textblob import TextBlob

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Example usage
new_statements = [
    "I am very happy with the service.",
    "This is the worst experience ever.",
    "It's okay, not great but not bad either."
]

for statement in new_statements:
    polarity, subjectivity = get_sentiment(statement)
    print(f"Statement: {statement} | Polarity: {polarity} | Subjectivity: {subjectivity}")

OUTPUT:
[nltk_data] Downloading package vader_lexicon to /root/nltk_data...
{'neg': 0.437, 'neu': 0.563, 'pos': 0.0, 'compound': -0.3252}
Statement: I am very happy with the service. | Polarity: 1.0 | Subjectivity: 1.0
Statement: This is the worst experience ever. | Polarity: -1.0 | Subjectivity: 1.0
Statement: It's okay, not great but not bad either. | Polarity: 0.14999999999999997 | Subjectivity: 0.6388888888888888
