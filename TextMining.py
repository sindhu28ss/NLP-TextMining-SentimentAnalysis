'''Text Mining Project - version 3'''

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import seaborn as sns
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import numpy as np


McData = pd.read_csv('/Users/sindhujaarivukkarasu/Documents/BAN 675 Text Mining & SMA/Project/McDonald_s_Reviews_data.csv', encoding='ISO-8859-1')
McData.head()

# Loop through each column in the DataFrame
for column in McData.columns:
    # Count the number of unique values in the current column
    distinct_values = len(McData[column].unique())
    print(f"{column}: {distinct_values} distinct values")
 
# Add a new column named "star" at index 10 by extracting the first part of each value in the "rating" column
McData.insert(10, "star", McData["rating"].str.split(" ").str[0])
# Convert the values in the "star" column to integer
McData['star'] = McData['star'].astype(int)
McData

# Remove specified columns from the DataFrame McData
McData.drop(['store_name','category','reviewer_id','review_time', 'rating'], axis = 1, inplace = True)
McData.columns

# Print the count of missing values in each column of the DataFrame McData
print(McData.isnull().sum())
# Remove rows containing any missing values from the DataFrame McData
McData.dropna(inplace=True)

print(McData.isnull().sum())

# Create a copy of the DataFrame McData and assign it to the variable df
df = McData.copy()

## Clean Review Function
def clean_review(review):
    """
    Clean the input review by:
    1. Converting it to lowercase
    2. Removing non-alphabetic characters and whitespaces
    3. Removing English stopwords
    4. Stemming (optional, currently commented out)
    
    Parameters:
    - review (str): Input review to be cleaned
    
    Returns:
    - str: Cleaned review
    """
    review = review.lower()# Convert the review to lowercase
    review = re.sub(r'[^a-zA-Z\s]', '', review) #remove every non-aplhabetic word and white spaces
    review = re.sub(r'\s+', ' ', review).strip() # Replace multiple spaces with a single space and remove leading/trailing spaces
    
    #remove stop words
    stop_words = set(stopwords.words('english'))# Get English stopwords
    review_tokens = nltk.word_tokenize(review)# Tokenize the review into words
    review = ' '.join([word for word in review_tokens if word not in stop_words])# Filter out stop words
    
    #stemming
    #stemmer = PorterStemmer()
    #stemmed_words = [stemmer.stem(word) for word in nltk.word_tokenize(review)]
    #review = ' '.join(stemmed_words)

    return review

# Create a new column named 'clean_reviews' in the DataFrame df to store the cleaned version of the 'review' column
df['clean_reviews'] = df['review'].apply(clean_review)

print(df[['clean_reviews']])

#*************************************************************************************************************************************************************************

'''Pie chart: Distribution of Star Ratings'''

# Count occurrences of each rating in the DataFrame and store the counts
sentiment_counts = df.groupby("star").size()
# Define colors for each rating category
colors = ['salmon', 'yellow', 'seagreen', 'skyblue', 'plum']  
# Create a new figure and axis for plotting
fig, ax = plt.subplots()

wedges, texts, autotexts = ax.pie(
    x=sentiment_counts, 
    labels=sentiment_counts.index,
    autopct=lambda p: f'{p:.2f}%\n({int(p*sum(sentiment_counts)/100)})', 
    colors=colors
)

ax.legend(sentiment_counts.index, title="Star", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
ax.axis('equal')  

# Set title at the top
ax.set_title('Rating Distribution', fontsize=14, fontweight='bold', color='#333333', pad=20)
plt.show()

#*************************************************************************************************************************************************************************

'''Word Cloud: Visualizing Frequent Words in Reviews'''

# Create a copy of the DataFrame df and assign it to df2
df2 = df.copy()

# Combine all clean reviews into a single string, split it into words, and count the frequency of each word
all_words = ' '.join(df2['clean_reviews']).split()
word_freq = Counter(all_words)
# Generate a word cloud from the word frequency data
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Frequent Words in Reviews', fontsize=16, fontweight='bold', color='navy', pad=20)
plt.show()

#****************************************************************************************************************************************************

'''Common Words: Analyzing the Most Frequent Words in Reviews'''

# Create a new column 'word_list' in the DataFrame df2 by tokenizing each review text
df2['word_list'] = df2['clean_reviews'].apply(lambda x: word_tokenize(str(x)))
# Count the occurrences of each word in the 'word_list' column and store them in the 'top_words' variable
top_words = Counter([item for sublist in df2['word_list'] for item in sublist])
print("Top 20 most common words:")
for word, count in top_words.most_common(20):
    print(f"{word}: {count}")
    
#******************************************************************************************************************************************************

'''Sentiment analysis (pos/neg/neutral)'''

# Initialization of Sentiment Analyzer
sentim_analyzer = SentimentIntensityAnalyzer()
# Applying sentiment analysis to each review and calculating sentiment score
df2['sentiment_score'] = df2['clean_reviews'].apply(lambda text: sentim_analyzer.polarity_scores(text)['compound'])
# Categorizing sentiment scores into positive, negative, or neutral
df2['sentiment'] = df2['sentiment_score'].apply(lambda score: 'positive' if score >= 0.05 else ('negative' if score <= -0.05 else 'neutral'))
# Printing the first 10 rows of the DataFrame with cleaned reviews, sentiment scores, and sentiment labels
print(df2[['clean_reviews', 'sentiment_score', 'sentiment']].head(10))

## Separate DataFrames for Positive, Negative, and Neutral Sentiments
Positive_sent = df2[df2['sentiment'] == 'positive']
Negative_sent = df2[df2['sentiment'] == 'negative']
Neutral_sent = df2[df2['sentiment'] == 'neutral']

#*************************************************************************************************************************************************************************

'''Analyzing Most Common Words by Sentiment'''

# Positive Sentiment Analysis
positive_words = Counter([item for sublist in df2[df2['sentiment'] == 'positive']['word_list'] for item in sublist])
print("Most common positvie words:")
for word, count in positive_words.most_common(20):
    print(f"{word}: {count}")

# Negative Sentiment Analysis
negative_words = Counter([item for sublist in df2[df2['sentiment'] == 'negative']['word_list'] for item in sublist])
print("Most common negative words:")
for word, count in negative_words.most_common(20):
    print(f"{word}: {count}")

# Neutral Sentiment Analysis
neutral_words = Counter([item for sublist in df2[df2['sentiment'] == 'neutral']['word_list'] for item in sublist])
print("Most common neutral words:")
for word, count in neutral_words.most_common(20):
    print(f"{word}: {count}")

'''Plotting Sentiment Distribution on Pie Chart'''

# Calculating counts of each sentiment category
sentiment_counts = df2.groupby("sentiment").size()
colors = ['salmon', 'silver', 'yellowgreen']  
fig, ax = plt.subplots()

wedges, texts, autotexts = ax.pie(
    x=sentiment_counts, 
    labels=sentiment_counts.index,
    autopct=lambda p: f'{p:.2f}%\n({int(p*sum(sentiment_counts)/100)})', 
    colors=colors)

ax.legend(sentiment_counts.index, title="Sentiment", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
ax.axis('equal')  
ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold', color='navy', pad=20)
plt.show()
#************************************************************************************************************************************************************************

'''Lexical Dispersion Plot'''

# Function to generate lexical dispersion plot
def lexical_dispersion_plot(word_list, words, title):
    # Initialize a list to store the positions of words
    positions = []
    # Iterate over each review in the word_list
    for i, review_words in enumerate(word_list):
        # Iterate over each word in the review
        for word in review_words:
            # If the word is in the list of words of interest
            if word in words:
                # Append the tuple (word, position) to the positions list
                positions.append((word, i))
    
    # Extract unique words and their positions
    unique_words, indices = zip(*positions)
    # Plot the dispersion of words
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.xlabel('Review Index')
    plt.ylabel('Words')
    # Scatter plot for each word
    for word in set(unique_words):
        word_indices = [i for w, i in positions if w == word]
        plt.scatter(word_indices, [word] * len(word_indices), label=word)
    # Move legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

# Define the selected words for each sentiment category
positive_words = ['good', 'excellent', 'great', 'clean', 'fast', 'staff', 'friendly']
negative_words = ['worst', 'bad', 'rude', 'wrong', 'minutes', 'manager']
neutral_words = ['neutral', 'slow', 'long', 'wait']

# Generate lexical dispersion plots for each sentiment category

lexical_dispersion_plot(Positive_sent['word_list'], positive_words, title='Lexical Dispersion Plot for Positive Words in Positive Reviews')

lexical_dispersion_plot(Negative_sent['word_list'], negative_words, title='Lexical Dispersion Plot for Negative Words in Negative Reviews')

lexical_dispersion_plot(Neutral_sent['word_list'], neutral_words, title='Lexical Dispersion Plot for Neutral Words in Neutral Reviews')

#*************************************************************************************************************************************************************************

'''Rating Distribution Across Top 10 Locations - version 2'''

# Convert 'star' column to string type
df2['star'] = df2['star'].astype(str)
# Extract 'City' and 'State' from 'store_address' column
df2[['City', 'State']] = df2['store_address'].apply(lambda x: pd.Series(x.split(', ')[-3:-1]))
plt.figure(figsize=(15, 6))# Create a new figure with a specified size
sns.set_palette("magma") # Set color palette for the plot
sns.countplot(x='State', hue='star', data=df2, order=df2['State'].value_counts().iloc[:10].index)# Plot count of star ratings by state, with hue based on star rating
plt.title('Distribution of Star Ratings by Location (Top 10)')# Add title to the plot
plt.xlabel('Location')# Label x-axis
plt.ylabel('Count of ratings')# Label y-axis
plt.xticks(rotation=0)# Rotate x-axis labels to improve readability
plt.legend(title='Star Rating')# Add legend with title for star ratings
plt.show()


'''Percentage Distribution of Ratings by Location (Top 10) - version 3'''

# Convert 'star' column to string type
df2['star'] = df2['star'].astype(str)
# Extract 'City' and 'State' from 'store_address' column
df2[['City', 'State']] = df2['store_address'].apply(lambda x: pd.Series(x.split(', ')[-3:-1]))
# Get the top 10 locations with the highest number of ratings
top_10_locations = df2['State'].value_counts().iloc[:10].index
# Filter the DataFrame to include only the top 10 locations
df_top_10 = df2[df2['State'].isin(top_10_locations)]
# Calculate total ratings for each location
total_ratings_per_location = df_top_10.groupby('State')['star'].count()
# Calculate percentage of each star rating within each location
df_new = df_top_10.groupby(['State', 'star']).size().unstack(fill_value=0).apply(lambda x: x / x.sum() * 100, axis=1)
# Create a new figure with a specified size
plt.figure(figsize=(15, 6))
# Plot percentage of star ratings by state, with hue based on star rating
df_new.plot(kind='bar', stacked=True, color=sns.color_palette('tab10', n_colors=5))
plt.title('Percentage Distribution of Ratings by Location (Top 10)')
plt.xlabel('Location')
plt.ylabel('Percentage of Ratings')
plt.xticks(rotation=45)
plt.legend(title='Star Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#*************************************************************************************************************************************************************************

'''Correlation Analysis for socio-economic factors - version 2'''

correlation_matrix = df2[['median household income', 'population', 'population density', 'star']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix', fontsize=16)
plt.show()

'''Correlation Analysis for socio-economic factors - version 3'''

correlation_matrix = df2[['median household income', 'population', 'population density', 'star']].corr()
# Replace the ones in the correlation matrix with zeros
np.fill_diagonal(correlation_matrix.values, 0)
# Plot the heatmap with annotations
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# Set the title of the plot
plt.title('Correlation Matrix', fontsize=16)
# Display the plot
plt.show()

#******************************************************************************************************************************************************************

''''Average Median Household Income by different Locations'''

avg_incomes = df2.groupby('State')['median household income'].mean().reset_index()

#Plotting
plt.figure(figsize=(10, 6))
plt.bar(avg_incomes['State'], avg_incomes['median household income'], color='skyblue')
plt.xlabel('Location')
plt.ylabel('Median Household Income')
plt.title('Average Median Household Income by different Locations')
plt.xticks(rotation=90) 
plt.tight_layout()
plt.show()

#******************************************************************************************************************************************************************

'''Average population by different Locations'''

avg_population = df2.groupby('State')['population'].mean().reset_index()

#Plotting
plt.figure(figsize=(10, 6))
plt.bar(avg_incomes['State'], avg_population['population'], color='skyblue')
plt.xlabel('Location')
plt.ylabel('Population')
plt.title('Average population by different Locations')
plt.xticks(rotation=90) 
plt.tight_layout()
plt.show()

#*******************************************************************************************************************************************************************

'''Average Median Household Income and Population by Locations'''

fig, ax1 = plt.subplots(figsize=(12, 8))
bar_width = 0.35
x = np.arange(len(avg_incomes['State']))
ax1.bar(x - bar_width/2, avg_incomes['median household income'], width=bar_width, color='purple', label='Median Income')

ax2 = ax1.twinx()
ax2.bar(x + bar_width/2, avg_population['population'], width=bar_width, color='pink', label='Population')

ax1.set_xlabel('Location')
ax1.set_ylabel('Median Household Income', color='skyblue')
ax2.set_ylabel('Population', color='lightgreen')
ax1.set_title('Average Median Household Income and Population by Locations')

ax1.set_xticks(x)
ax1.set_xticklabels(avg_incomes['State'], rotation=90)

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

#******************************************************************************************************************************************************************

'''Rating vs. Median Household Income'''

df2= df2.sort_values(by='star')
sns.set_style('whitegrid')
sns.set_palette('muted')

plt.figure(figsize=(10, 6))
sns.boxplot(x='star', y='median household income', data=df2)
plt.title('Rating vs. Median Household Income')
plt.xlabel('Rating')
plt.ylabel('Median Household Income')
plt.yticks([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000], 
           ['10000', '20000', '30000', '40000', '50000', '60000', '70000', '80000', '90000', '100000', '1100000', '1200000'])
plt.grid(True)
plt.show()

#******************************************************************************************************************************************************************

'''Geographic Distribution of McDonald\'s Stores showing Median Household Income and Ratings'''

fig = px.scatter_geo(df2, 
                     lat='latitude ', 
                     lon='longitude', 
                     color='median household income',
                     size='rating_count',
                     projection='albers usa',
                     title='Geographic Distribution of McDonald\'s Stores showing Median Household Income and Ratings',
                     labels={'median household income': 'Median Household Income ($)'},
                     opacity=0.7
                    )

fig.update_geos(projection_type="albers usa")
fig.update_layout(coloraxis_colorbar=dict(title='Median Household Income ($)'))
fig.write_html('/Users/sindhujaarivukkarasu/Documents/BAN 675 Text Mining & SMA/Project/geospatial_plot1.html')

#*******************************************************************************************************************************************************************

'''Population vs. Rating Count'''

# Set the style of the plot
sns.set(style="ticks")

# Create scatter plot for population vs. rating
sns.lmplot(x='population', y='rating_count', data=df2, scatter_kws={"s": 50}, palette='Blues', height=6, aspect=2)

# Set titles and labels
plt.title('Population vs. Rating Count')
plt.xlabel('Population')
plt.ylabel('Rating Count')

# Show the plot
plt.show()

#********************************************************************************************************************************************************************

'''Population vs Rating Count Across Locations'''

# Group population state-wise and calculate mean rating count for each state
state_population_rating = df2.groupby('State').agg({'population': 'mean', 'rating_count': 'mean'}).reset_index()

# Set the size of the plot
plt.figure(figsize=(12, 8))

# Create scatter plot with state on x-axis, population on y-axis, and rating count as bubble size
sns.scatterplot(data=state_population_rating, x='State', y='population', size='rating_count', color='navy', sizes=(50, 500), alpha=0.7)

# Set titles and labels
plt.title('Population vs Rating Count Across Locations')
plt.xlabel('Location')
plt.ylabel('Population')

# Show the plot
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

#**********************************************************************************************************************************************************************

'''Population Density vs. Rating Count Across Locations'''

# Group by state and calculate mean population density
state_wise_density = df2.groupby('State').agg({'population density': 'mean', 'rating_count': 'mean'}).reset_index()

# Set the style of the plot
sns.set(style="ticks")

# Create scatter plot with bubble size based on rating count
plt.figure(figsize=(12, 8))
sns.scatterplot(data=state_wise_density, x='State', y='population density', size='rating_count', color='navy', sizes=(50, 500), alpha=0.7)
plt.title('Population Density vs. Rating Count Across Locations')
plt.xlabel('Location')
plt.ylabel('Population Density')
plt.legend(title='Rating Count')
# Show the plot
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

#**************************************************************************************************************************************************************************

'''Population Growth Percentage vs. Rating Count Across Locations'''

# Group by state and calculate mean population growth percentage
state_wise_growth = df2.groupby('State').agg({'population growth(%)': 'mean', 'rating_count': 'mean'}).reset_index()

# Set the style of the plot
sns.set(style="ticks")

# Create scatter plot with bubble size based on rating count
plt.figure(figsize=(12, 8))
sns.scatterplot(data=state_wise_growth, x='State', y='population growth(%)', size='rating_count', color='navy', sizes=(50, 500), alpha=0.7)
plt.title('Population Growth Percentage vs. Rating Count Across Locations')
plt.xlabel('Location')
plt.ylabel('Population Growth Percentage (%)')

# Set y-axis limits based on the range of population growth percentage values
plt.ylim(-50, 160)

plt.legend(title='Rating Count')
# Show the plot
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

#*************************************************************************************************************************************************************************

