**ABOUT DATASET**

his is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment .

Content
It contains the following 6 fields:

target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)

ids: The id of the tweet ( 2087)

date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)

flag: The query (lyx). If there is no query, then this value is NO_QUERY.

user: the user that tweeted (robotickilldozr)

text: the text of the tweet (Lyx is cool)

https://www.kaggle.com/datasets/kazanova/sentiment140**

![Uploading twitterdataset.jpg…]()



**GOAL**

Classify tweets as positive, negative, or neutral sentiments, allowing for a comprehensive analysis of public opinion. This classification helps in understanding emotional tones, identifying trends in user interactions, and providing valuable insights into consumer sentiments and perceptions regarding various topics


**ALGORİTHM**

Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture designed to model sequences and time series data. LSTM was introduced by Sepp Hochreiter and Jürgen Schmidhuber in their 1997 paper titled "Long Short-Term Memory." It was developed to overcome the limitations of traditional RNNs, particularly the problem of vanishing and exploding gradients, which hinder the training of networks on long sequences.

**Key Features of LSTM:**
Memory Cells: LSTM networks use memory cells to maintain information over long periods. This enables them to capture long-term dependencies in the data.

Gates: LSTMs have three main gates that control the flow of information:

Forget Gate: Decides what information from the cell state should be discarded.
Input Gate: Determines what new information should be stored in the cell state.
Output Gate: Controls what information is sent to the next layer based on the current cell state.
Cell State: The cell state is the internal memory of the LSTM, allowing it to retain information across time steps. It carries relevant information from previous inputs to the current input.

**Applications of LSTM:**
LSTMs are widely used in various applications, including:

Natural Language Processing (NLP): Sentiment analysis, language modeling, and machine translation.
Time Series Forecasting: Stock price prediction, weather forecasting, and resource consumption prediction.
Speech Recognition: Transcribing spoken language into text.
Video Analysis: Recognizing patterns in sequences of frames.
Conclusion
LSTM is a powerful tool in the field of deep learning, particularly for tasks involving sequential data. Its ability to learn from long-term dependencies makes it a preferred choice for many applications, leading to advancements in areas like NLP, time series analysis, and more. The architecture continues to be a foundation for many state-of-the-art models in artificial intelligence.

**RESULT**

In the sentiment analysis project, the LSTM model achieved an impressive accuracy of 0.85 on the test dataset. This result indicates that the model correctly classified 85% of the tweets into their respective sentiment categories—positive, negative, or neutral. This level of accuracy demonstrates the model's effectiveness in capturing the nuanced patterns within the tweet text, suggesting that it has learned to understand the context and sentiment conveyed in the language used.

The choice of using LSTM (Long Short-Term Memory) networks in this project is particularly advantageous due to their ability to handle sequential data and retain long-term dependencies. Unlike traditional neural networks, LSTMs are designed to overcome the vanishing gradient problem, making them more suitable for processing and learning from sequences, such as sentences in text data. This characteristic is crucial for sentiment analysis, as the sentiment of a tweet often depends on the context established by the preceding words.

In summary, the model's accuracy of 0.85 reflects its strong performance in classifying sentiments, which can be attributed to the capabilities of the LSTM architecture in managing sequential dependencies and capturing the complexity of language. This makes LSTM a fitting choice for sentiment analysis tasks

![en spm](https://github.com/user-attachments/assets/2f591c4a-ef7c-4634-99c2-0798ecd412d2)


