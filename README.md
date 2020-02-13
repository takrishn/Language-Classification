# Language Identification for Nonfictional and Fictional Languages 

CS 175, Winter 2020
Team Members: Victor Le, Theja Krishna, Natalie Ma

1. **Project Summary**
Problem: There are approximately 6500 spoken languages in the world right now with many fictional ones being created every so often (i.e Tolkien Elvish, Simlish, and Dothraki to name a few). Identifying what language a given text or audio is in can be a challenge for any human, but we intend to off-load the task to the machine. This project will use machine learning to classify a given text input and output the name of the language being used. If we have time, we would like to include a speech-to-text portion to give our project more of a real-world usage. This way, we could watch or listen to a movie or podcast and be able to tell which language is being spoken without having to rely on subtitles. Example: “Cries in Spanish”
Google Translate also utilizes RNN but to a much higher degree, taking it a step further than us by both detecting which language is being used, and translating it to a language of choice. We cannot compete with Google’s model, but we can take a small step towards their algorithm through language classification.

2. **Technical Approach**
We plan on utilizing an RNN machine learning model with datasets that include commonly used words in a language, including ngrams, to help classify which language is being tested. Our goal is to be able to classify at least five real languages and for a fun challenge at least one fictional language. 

3. **Data Sets**
We plan on using Google’s ngrams for English. For the other languages we plan on aggregating the corpus for our other languages ourselves. This can be done through scraping through Yelp reviews in Spanish and other languages (example: https://www.yelp.it/milano) or ebooks (https://www.gutenberg.org/catalog/). We are looking to compare the performance of using different data sets for translating, and plan on using Google’s ngrams for English as the control. For fictional languages, there are existing datasets for some fictional languages found on the web on sites like reddit and https://eldamo.org/ (a database of Elvish words from the Tokien’s Lord of the Ring franchise). 

For this classifying problem, we only need to classify which language the given text is a part of. 

4. **Experiments and Evaluation**
Since we plan on using both foreign Yelp reviews and eBooks in different languages, we want to test to see which vocabulary set will produce more accurate results (as Yelp reviews would likely contain more commonly used words, while eBooks would hold more unique and advanced words.) We’d like to incorporate a speech-to-text component as well, so we also want to train our machine using audio clips from various sources, such as (for example) audio from a news report and audio from a game show to see which performs better. Once we see what types of samples work best, we can use it and other similar sources to improve our machine.

5. **Software**
We will be using Python along with a few additional Python libraries such as numpy, NLTK, and Pytorch. Version control will be handled through Git, and our program will be written on Jupyter Notebook for a clean visualization of our data at each step of our process. Jupyter Notebook will also make it easier to document our code as we write it.

We won’t be using any additional API calls, but instead we will be processing the data ourselves and building the model with Pytorch. Regarding our extra speech-to-text feature, we’ll probably use Mozilla’s open source DeepSpeech library to convert spoken language to text, and then run the generated text through our existing model.
