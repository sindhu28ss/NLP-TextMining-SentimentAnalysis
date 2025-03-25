# NLP-TextMining-SentimentAnalysis

The analysis of customer reviews is crucial for businesses aiming to understand customer sentiments, identify trends, and make informed strategic decisions. This project leverages advanced text mining techniques to analyze a dataset of reviews from McDonald’s, a leading global fast-food chain. By systematically categorizing customer feedback into sentiments, this study illuminates patterns that assist in enhancing marketing strategies and customer service practices, helping McDonald's adapt to market demands and improve customer satisfaction.

## Data Description
The dataset contains 30,000 reviews from McDonald’s locations across the U.S., with 14 columns covering customer feedback, ratings, store locations

## Research Questions
- How do socio-economic factors, sentiment analysis, and geographic distribution influence customer reviews and ratings for McDonald's locations?
- Some other questions that our research answers:
  -	What is the general perception of McDonald’s? Is it positive or negative?
  -	What factors influence a customer to leave a review? (Ex: Food, service etc)
  -	Which part of the country receives the most reviews? 

## Business Impact
- **Consumer Insights:** Uncovers customer sentiments and preferences to help tailor offerings and enhance satisfaction.
- **Data-Driven Decisions:** Links feedback with socio-economic factors and locations to guide strategic business actions.
- **Competitive Advantage:** Empowers businesses to respond to feedback effectively and stay ahead in a dynamic market.


## Methodology

**Data Preprocessing:**
- Collected socio-economic data (e.g., population, income) from usa.com to provide context for each review.
- Cleaned review text: standardized to lowercase, removed non-alphabetic characters, tokenized using `nltk.word_tokenize`, and removed stop words.
<p align="left">
  <img src="https://github.com/sindhu28ss/NLP-TextMining-SentimentAnalysis/blob/main/images/wordcloud.png" width="500">
   <img src="https://github.com/sindhu28ss/NLP-TextMining-SentimentAnalysis/blob/main/images/common%20words.png" width="180">
</p>

**Sentiment Analysis:**
- Used NLTK’s `SentimentIntensityAnalyzer` to classify reviews into positive, negative, or neutral sentiments for deeper insight into customer feedback.
<p align="left">
  <img src="https://github.com/sindhu28ss/NLP-TextMining-SentimentAnalysis/blob/main/images/pos%20words.png" width="500">
  <img src="https://github.com/sindhu28ss/NLP-TextMining-SentimentAnalysis/blob/main/images/neg%20words.png" width="500">
</p>

<p align="center">
  <img src="https://github.com/sindhu28ss/NLP-TextMining-SentimentAnalysis/blob/main/images/sentdist.png" width="500">
</p>

The analysis showed that over half of the reviews carried a positive sentiment, even when paired with lower star ratings. This highlights a key insight: customer sentiment doesn't always align with numerical ratings. Sentiment analysis uncovers this nuance, offering deeper context for understanding and acting on customer feedback.

- Performed correlation analysis to explore relationships between socio-economic factors (e.g., population density, income) and customer ratings.

## Results
The analysis yielded insightful results that shed light on various aspects of customer sentiments, socio-economic correlations, and geographical trends within the dataset. We utilized various visualization techniques to present our findings. The following visual representations encapsulate our key insights:

