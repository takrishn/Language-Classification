import requests
import re
import csv
import os
import unicodecsv as csv
from bs4 import BeautifulSoup
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

# python -m nltk.downloader stopwords
# python -m nltk.downloader punkt 

def makeSoup(url: str) -> BeautifulSoup:
    """makes soup --Theja"""

    response = requests.get(url)
    return BeautifulSoup(response.content, 'html.parser')

def extractElvish() -> dict:
    """scrapes Elvish site; returns {Elvish:English} dict --Theja"""

    soup = makeSoup('https://eldamo.org/content/word-indexes/words-p.html')
    elvish_dict = dict()

    for entry in soup.find_all('dt'):
        elvish = re.search('.html">(.*)</a>', str(entry))
        english = re.search('“(.*)”', str(entry))
        
        if elvish: elvish = elvish.group(1)
        if english: english = english.group(1)

        elvish_dict[str(elvish).strip('!')] = str(english)
    
    return elvish_dict

def extractWords(lang_name: str, lang_sites: dict) -> list:
    """scrape url to return list of words from language --Victor"""

    url = lang_sites[lang_name]
    nonEnglish = False
    alternate = False
    if (url != lang_sites["English"]):
        nonEnglish = True
        alternate = True
    soup = makeSoup(url)
    corpus = soup.find_all('td')
    result = []

    for text in corpus:
        text = str(text)
        word_check = re.match(r"<td>\D+<\/td>", text)
        if(word_check):
            if(nonEnglish):
                if(alternate):
                    result.append(text[4:-5])
                    alternate = False
                else:
                    alternate = True
            else:
                result.append(text[4:-5])

    return result

def extractSimlish() -> list:
    """extracts all Simlish words and phrases from site; returns
    list of Simlish words --Natalie"""

    parser = makeSoup('https://sims.fandom.com/wiki/Simlish')
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

    return result

def removeFictionalStopWords(lang_dict: dict) -> dict:
    """takes in dict {Language:English} and returns dict {Language:English}
    with stop words removed based on corresponding english definition --Theja"""

    filtered_dict = dict()
    stop_words = set(stopwords.words('english')) 
    language, english = list(lang_dict.keys()), list(lang_dict.values())

    for i, word in enumerate(english):
        if word.lower() not in stop_words:
            filtered_dict[language[i]] = english[i]
    
    return filtered_dict

def removeStopWords(lang_name: str, word_list: list) -> list:
    """takes in language name and list of words from language and removes
    stop words (only languages supported by nltk stop word library) --Theja"""

    filtered_words = []
    stop_words = set(stopwords.words(lang_name.lower()))
    
    for i, word in enumerate(word_list):
        if word.lower() not in stop_words:
            filtered_words.append(word)
    
    return filtered_words

def writeToCSV(lang_name: str, word_list:list, csv_file_path:str) -> None:
    """given language name and list of words from that language, create
    csv with 'Word' and 'Language' as headers; else add word/language
    entries to existing csv --Theja"""

    mode = 'wb'
    if os.path.exists(csv_file_path): mode = 'ab'

    with open(csv_file_path, mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=',', encoding='utf-8')
        if mode == 'wb': writer.writerow(['Word', 'Language'])

        for word in word_list:
            writer.writerow([word.strip(','), lang_name])

def main():
    dataset_path = 'language_dataset_no_stopwords.csv'

    lang_sites = {'English': 'https://1000mostcommonwords.com/1000-most-common-english-words/',
                'Spanish':'https://1000mostcommonwords.com/1000-most-common-spanish-words/',
                'German':'https://1000mostcommonwords.com/1000-most-common-german-words/',
                'Italian':'https://www.1000mostcommonwords.com/words/1000-most-common-italian-words',
                'French':'https://1000mostcommonwords.com/1000-most-common-french-words/'}

    for lang_name in lang_sites.keys():
        lang_words = extractWords(lang_name, lang_sites)
        filtered_lang_words = lang_words # removeStopWords(lang_name, lang_words)
        writeToCSV(lang_name, filtered_lang_words, dataset_path)
        print('{}: {}'.format(lang_name, len(filtered_lang_words)))

    simlish_words = extractSimlish()
    writeToCSV('Simlish', simlish_words, dataset_path)
    print('Simlish:', len(simlish_words))

    filtered_elvish_dict = extractElvish() # removeFictionalStopWords(extractElvish())
    writeToCSV('Tolkien Elvish', list(filtered_elvish_dict.keys()), dataset_path)
    print('Tolkien Elvish:', len(filtered_elvish_dict))


if __name__ == '__main__':
    main()