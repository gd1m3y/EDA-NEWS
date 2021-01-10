#  Sentiment Analysis on Author data by using Python, Vader sentiment and Specrom Api
>Sentiment analysis (or opinion mining) is a natural language processing technique used to determine whether data is positive, negative or neutral. Sentiment analysis is often performed on textual data to help businesses monitor brand and product sentiment in customer feedback, and understand customer needs.Sentiment analysis can help us get some insights on what actually is the content about, So in this post we will try to what kind of content each individual author provides on a monthly basis.

![img](https://monkeylearn.com/static/3ca10d6ce5dc6922836f278aef38f765/50bf7/what-is-sentiment-analysis6%402x.png)
 [source](https://monkeylearn.com/sentiment-analysis/)
## Getting our Data 
> Data is heart and soul of our data analysis , It is the factor which determines how accurate or relevant our results will be. So to get our data we will be utilizing the Specrom API. Specrom News API fetches recent news articles(<24) from more than 20000 sources in very less amount of time. This api is very easy to use and just requires algorithmia python library

```
import Algorithmia
client = Algorithmia.client('your_key')
algo = client.algo('specrom/LatestNewsAPI/0.1.6')
algo.set_options(timeout=300)
final_data = []


input = {
"domains": "theguardian.com",
"topic": "politics",
"q": "",
"qInTitle": "",
"content": "true",
"page": '1',
"author_only": "true"
}
results = algo.pipe(input).result
```
>The api requires a Input dictionary which contains the conditions for the result query such as a domain value which specifies the domain / source we want to fetch our data from, topic is the category in which we want our news such as sports,politics etc.content specifies whether we want the whole content of our news,the api returns 100 queries per page so to access other queries we can manipulate page value

```
{
  "Article": [
    {
      "author": "[\"Lauren Fox and Ted Barrett, CNN\"]",
      "description": "Divisions within the Republican conference spilled out once again Tuesday as GOP senators dismissed key pieces of their own leadership's stimulus proposal not even a day after its release.",
      "publishedAt": "2020-07-28",
      "source_name": "CNN",
      "source_url": "cnn.com",
      "title": "Republicans revolt against GOP's initial stimulus plan  - CNNPolitics",
      "url": "https://www.cnn.com/2020/07/28/politics/republican-reaction-gop-stimulus-plan/index.html?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+rss%2Fcnn_allpolitics+%28RSS%3A+CNN+-+Politics%29",
      "urlToImage": "https://cdn.cnn.com/cnnnext/dam/assets/200518181852-senator-ben-sasse-1-super-tease.jpg"
    },
  ],
  "status": "ok",
  "totalResults": 113
}
```
>So i ran the above query using different parameters and got 267710 posts from past 1 month of  different sources .

## Exploratory Data Analysis

>Since the Author names were in a string list i used json loads to get them into proper format,
now lets see how our data looks like.

![img](https://i.ibb.co/vZzGWk3/Screenshot-82.png)

>At first glance we can see that we have 267710 entries in our data frame with no NULL values.But when we futher inspect our author names .

![img](https://i.ibb.co/yYV9ncS/Screenshot-84.png)

>we can see that the top value is [""] which is problamatic for us since it means there is no author for that data.So we run a query and perform some cleaning of data ie removing the data with either invalid names or null names or representin the name of the organization . All the steps for doing this are included in the notebook.

> So to know a bit more the Topics which our posts we plotted a bar chart to get a better understanding.Also the frequency by which each author posts for the past month is plotted.
![img](https://i.ibb.co/1GhF2Y5/download.png)
![img](https://i.ibb.co/MPhrJr5/download-1.png)

## Sentiment Analysis
> Now onto our major objective which is the sentiment analysis of our authors data,So to perform the sentiment analysis we will use the Vader library since it can provide almost near to perform in a very minimal amount of time.
```
!pip install vaderSentiment
```
> After installing we can perform sentiment analysis by import vader library and initializing its object.
```
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
senti = SentimentIntensityAnalyzer()
senti.polarity_scores('\n Two cities have around a 40 per cent chance of seeing snow on the big day\n ')

Output: {'neg': 0.0, 'neu': 0.882, 'pos': 0.118, 'compound': 0.25}
```
> The polarity Scores method returns a dictionary consisting of the probality of each sentiment, where compound is the mixture of positive and negative sentiments.

>Now we have to choose a author from our list to perform sentiment analysis.For our use case we will take the author with the highest no of articles.Our final goal is to plot a time-series plot of average sentiment of the articles posted by the author in a month.So to extract the dates and average sentiments we created a custom function .
```
#final function 
def custom_senti_analysis(final_frame,top,senti):
    #Extracting all the posts by the particular author
    author_frame = final_frame[final_frame.author==top]
    
    # list of dates
    dates = list(author_frame.groupby('date').count().index)
    date_wise_dict = {}
    #sorting the articles posted on the same dates together and appending them to a dictionary
    for i in dates:
        temp_frame = author_frame[author_frame.date == i]
    #     print(temp_frame.head())
        for j in range(len(temp_frame)):
            if i in date_wise_dict:
                date_wise_dict[i].append(senti.polarity_scores(temp_frame.full_text.iloc[j]))
            else:
                date_wise_dict[i] = []
    
    # calculating the average sentiment of each date
    final_dict = {}
    
    for i in dates:
        counters = collections.Counter()
        for d in date_wise_dict[i]:
            counters.update(d)
        counters = dict(counters)
        total = len(date_wise_dict[i])
        counters = {k: int((v / total)*100) for k, v in counters.items()}
        final_dict[i] = counters
    
    #plotting the time series
    frame = pd.DataFrame(final_dict.values())
    frame.index = dates
    ax = frame.plot(figsize=(12,8),title='Average sentiment per day by ' + top)
    ax.set_xlabel('Dates',fontsize=15,rotation='horizontal')
    plt.show()
```
> Before I explain what each line does let me show the resulting time series plot.
![img](https://i.ibb.co/MGS3tnh/download-2.png)
We can see that on average our author has mostly posted neutral content.But the thing to notice is that there are many spikes on compound sentiment which means that there are some articles which is a combination of mostly positive and negative sentiments.Sometimes delivering a negative speech but sugarcoating it can be termed as compound,So this insight helps us alot in understanding the context behind the content.

now to understand each block of code 
```
date_wise_dict = {}
for i in dates:
    temp_frame = author_frame[author_frame.date == i]
#     print(temp_frame.head())
    for j in range(len(temp_frame)):
        if i in date_wise_dict:
            date_wise_dict[i].append(senti.polarity_scores(temp_frame.full_text.iloc[j]))
        else:
            date_wise_dict[i] = []
```
> Since our goal was to plot a time series graph we need a properly formated data where each sentiment is paired properly with its corresponding date. So we stored this data in a dictionary with dates being the respective keys for the sentiments.

```

final_dict = {}
for i in dates:
    counters = collections.Counter()
    for d in date_wise_dict[i]:
        counters.update(d)
    counters = dict(counters)
    total = len(date_wise_dict[i])
    counters = {k: int((v / total)*100) for k, v in counters.items()}
    final_dict[i] = counters
    
```
> now to plot the time series we need average sentiment data , this block of code completes that purpose by calculating the average sentiment for each date and storing them in a final dictionary.

#### Other resutls
> Here are the analysis on some other authors.
![img](https://i.ibb.co/zQPpqvk/download-3.png)
The missing values in the plot just indicates the author didnt posted anything on that particular day.
![img](https://i.ibb.co/3mnT5dT/download-4.png)


That's it for today i hope you understood my article about author sentiment Analysis using python,Vadersentiment and specrom API.Also as a ending note i would like to give a shout out to specrom news api as it is very cheap and easy to use as compared to other costly apis for budding data scientists and students like me .
