import re
import urllib
import requests
from bs4 import BeautifulSoup

from elvishScraper import writeToCSV

def getParser(url: str) -> BeautifulSoup:
    """ returns a BeautifulSoup parser of the URL"""
    page = requests.get(url)
    return BeautifulSoup(page.text, 'html.parser')
    
    
def extractSimlish(parser: BeautifulSoup) -> list:
    """ extracts all Simlish words and phrases from the parser """
    corpus = parser.find_all('li')
    result = []

    # ignore lines parsed from html not relevant to Simlish
    count = 0
    
    for line in corpus:
        text = str(line)
        word_check = re.search('<b>\D+<\/b>', text)
        if(word_check):
            if count > 7:
                text = line.get_text()

                # discard the english translation of the word
                end_of_phrase = text.find(':')
                simlish = text[1:end_of_phrase] 

                # find instances of where 2 Simlish terms are grouped in one line
                find_or = simlish.find(', or')
                find_or2 = simlish.find(' or ')

                # find instances where extra unneeded commentary is added
                find_exclam = simlish.find("!")
                find_paren = simlish.find("(")

                if find_or != -1:
                    result.append(simlish[:find_or])
                    result.append(simlish[find_or + 5:])
                elif find_or2 != -1:
                    result.append(simlish[:find_or2])
                    result.append(simlish[find_or2 + 4:])
                elif find_exclam != -1:
                    result.append(simlish[:find_exclam])
                elif find_paren != -1:
                    result.append(simlish[:find_paren - 1])
                else:
                    result.append(simlish)
            count += 1
    #print(result)
    return result

def main():
    url = "https://sims.fandom.com/wiki/Simlish"
    parser = getParser(url)
    words = extractSimlish(parser)
    writeToCSV('Simlish', words, 'language_dataset.csv')

if __name__ == '__main__':
    main()