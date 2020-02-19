import requests
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

# python -m nltk.downloader stopwords
# python -m nltk.downloader punkt 

def makeSoup(url: str) -> BeautifulSoup:
    """makes soup"""

    response = requests.get(url)
    return BeautifulSoup(response.content, 'html.parser')

def scrapeElvish() -> dict:
    """scrapes Elvish site; returns {Elvish:English} dict"""

    soup = makeSoup('https://eldamo.org/content/word-indexes/words-p.html')
    elvish_dict = dict()

    for entry in soup.find_all('dt'):
        elvish = re.search('.html">(.*)</a>', str(entry))
        english = re.search('“(.*)”', str(entry))
        
        if elvish: elvish = elvish.group(1)
        if english: english = english.group(1)

        elvish_dict[str(elvish)] = str(english)
    
    return elvish_dict

def removeStopWords(lang_dict: dict) -> dict:
    """takes in dict {Language:English} and returns dict {Language:English}
    with stop words removed based on corresponding english definition"""

    filtered_dict = dict()
    stop_words = set(stopwords.words('english')) 
    language, english = list(lang_dict.keys()), list(lang_dict.values())

    for i, word in enumerate(english):
        if word not in stop_words:
            filtered_dict[language[i]] = english[i]
        else:
            print('REMOVING STOP WORD {}: {}'.format(language[i], english[i]))
    
    return filtered_dict

def main():
    elvish_dict = scrapeElvish()
    print('Number of Elvish words:', len(elvish_dict), '\n')
    filtered_elvish_dict = removeStopWords(elvish_dict)
    print('\nNumber of Elvish words after removing stop words:', len(filtered_elvish_dict))


if __name__ == '__main__':
    main()
