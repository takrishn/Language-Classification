import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import re

def makeSoup(url: str) -> BeautifulSoup:
    response = requests.get(url)
    return BeautifulSoup(response.content, 'html.parser')

def scrapeElvish() -> dict:
    """scrapes Elvish site; returns { Elvish : English } dict"""

    soup = makeSoup('https://eldamo.org/content/word-indexes/words-p.html')
    elvish_dict = dict()

    for entry in soup.find_all('dt'):
        elvish = re.search('.html">(.*)</a>', str(entry))
        english = re.search('“(.*)”', str(entry))
        
        if elvish: elvish = elvish.group(1)
        if english: english = english.group(1)

        elvish_dict[elvish] = english
    
    return elvish_dict

def main():
    elvish_dict = scrapeElvish()
    print(elvish_dict)


if __name__ == '__main__':
    main()
