import re
import sklearn
import requests
from bs4 import BeautifulSoup #run command to install: conda install -c anaconda beautifulsoup4

lang_sites = {"eng": "https://1000mostcommonwords.com/1000-most-common-english-words/",
                "spn":"https://1000mostcommonwords.com/1000-most-common-spanish-words/",
                "ger":"https://1000mostcommonwords.com/1000-most-common-german-words/",
                "itl":"https://www.1000mostcommonwords.com/words/1000-most-common-italian-words",
                "fre":"https://1000mostcommonwords.com/1000-most-common-french-words/"}

def extract_words(url):
    nonEnglish = False
    alternate = False
    if (url != lang_sites["eng"]):
        nonEnglish = True
        alternate = True
    page = requests.get(url)
    #page.add_header('Accept-Encoding', 'utf-8')
    soup = BeautifulSoup(page.text, 'html.parser')
    corpus = soup.find_all('td')#.decode('utf-8')
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

def writeToFile(word_list, fileName):
    f = open(fileName + ".txt", "w")
    for word in word_list:
        f.write(word + "\n")
    f.close()

def export_lang_to_text():
    global lang_sites
    eng_words = extract_words(lang_sites["eng"])
    spn_words = extract_words(lang_sites["spn"])
    ger_words = extract_words(lang_sites["ger"])
    itl_words = extract_words(lang_sites["itl"])
    fre_words = extract_words(lang_sites["fre"])

    writeToFile(eng_words, "1000english_words.txt")
    writeToFile(spn_words, "1000spanish_words.txt")
    writeToFile(ger_words, "1000german_words.txt")
    writeToFile(itl_words, "1000italian_words.txt")
    writeToFile(fre_words, "1000french_words.txt")

def main():
    print("Running...")
    export_lang_to_text()
    print("Complete!")

main()