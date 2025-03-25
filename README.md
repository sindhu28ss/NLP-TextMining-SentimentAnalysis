# NLP-TextMining-SentimentAnalysis

The analysis of customer reviews is crucial for businesses aiming to understand customer sentiments, identify trends, and make informed strategic decisions. This project leverages advanced text mining techniques to analyze a dataset of reviews from McDonald’s, a leading global fast-food chain. By systematically categorizing customer feedback into sentiments, this study illuminates patterns that assist in enhancing marketing strategies and customer service practices, helping McDonald's adapt to market demands and improve customer satisfaction.

<p align="center">
  <img src="https://github.com/sindhu28ss/NLP-TextMining-SentimentAnalysis/blob/main/images/SentimentAnalysis_Review.png" width="500">
</p>

## Data Description
The [dataset](https://github.com/sindhu28ss/NLP-TextMining-SentimentAnalysis/blob/main/McDonald_s_Reviews_data.csv) contains 30,000 reviews from McDonald’s locations across the U.S., with 14 columns covering customer feedback, ratings, and store locations.

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
<p align="center">
  <img src="https://github.com/sindhu28ss/NLP-TextMining-SentimentAnalysis/blob/main/images/pos%20words.png" width="400">
  <img src="https://github.com/sindhu28ss/NLP-TextMining-SentimentAnalysis/blob/main/images/neg%20words.png" width="400">
</p>

<p align="center">
  <img src="https://github.com/sindhu28ss/NLP-TextMining-SentimentAnalysis/blob/main/images/sentdist.png" width="400">
</p>

The analysis showed that over half of the reviews carried a positive sentiment, even when paired with lower star ratings. This highlights a key insight: customer sentiment doesn't always align with numerical ratings. Sentiment analysis uncovers this nuance, offering deeper context for understanding and acting on customer feedback.

**Statistical & Visual Analysis:**
To deepen the analysis, I explored how socio-economic factors like population and median household income relate to customer ratings. This helped uncover patterns in customer sentiment influenced by location and demographics, providing valuable context for strategic insights.

**Income vs Star Ratings:**
The box plot below highlights a clear trend — areas with higher median household incomes tend to give higher star ratings. This insight helps understand how economic factors influence customer satisfaction and can inform targeted business strategies.

<p align="left">
  <img src="https://github.com/sindhu28ss/NLP-TextMining-SentimentAnalysis/blob/main/images/Ratingvsincome.png" width="350">
  <img src="https://github.com/sindhu28ss/NLP-TextMining-SentimentAnalysis/blob/main/images/map.png" width="400">
</p>

**Geospatial View of Ratings & Income:**
This map visualizes store locations across the U.S., with point size representing review count and color intensity reflecting median household income. It provides a geographic lens into where reviews are concentrated and how income levels vary across regions.


## Recommendations

Based on the insights drawn from sentiment analysis, socio-economic correlations, and geographic patterns, the following recommendations can be made to enhance customer satisfaction and support strategic decision-making:

- **Targeted Service Improvements:** Locations with a high percentage of 1-star reviews, especially in lower-income areas, may benefit from operational improvements such as faster service, better order accuracy, or enhanced staff training.

- **Localized Marketing Strategies:** Geographic and socio-economic variations in sentiment suggest that marketing campaigns should be tailored to specific regions. Stores in suburban or moderately populated areas, where feedback volume is higher, present opportunities for hyper-local engagement.

- **Customer Engagement Programs:** Encourage satisfied customers to leave reviews, particularly in high-income areas where positive sentiment aligns with higher ratings. This can help balance overall public perception and boost reputation.

- **Resource Allocation:** Insights from population vs. rating trends indicate that stores in less populated areas still receive significant feedback. These locations should not be overlooked in strategic planning or resource distribution.

- **Feedback-Driven Innovation:** Common themes found in both positive and negative reviews (e.g., food quality, service speed) should be used to guide continuous product and service improvements, ensuring the brand evolves with customer expectations.

