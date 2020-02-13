import re
import sklearn
import urllib
from sklearn.feature_extraction.text import * 
from lxml.html import * #run command to install: conda install -c anaconda lxml 

lang_sites = {"eng": "https://1000mostcommonwords.com/1000-most-common-english-words/",
                    "spn":"https://1000mostcommonwords.com/1000-most-common-spanish-words/",
                    "ger":"https://1000mostcommonwords.com/1000-most-common-german-words/",
                    "itl":"https://www.1000mostcommonwords.com/words/1000-most-common-italian-words",
                    "fre":"https://1000mostcommonwords.com/1000-most-common-french-words/"}

def grab_html(url):
    infile = open(url, 'r', encoding="utf8")
    result = ""
    for line in infile:
        result += line
    return result

def main():
    eng_html = grab_html(lang_sites["eng"])    
    #spn_html = grab_html(lang_sites["spn"])    
    #ger_html = grab_html(lang_sites["ger"])    
    #itl_html = grab_html(lang_sites["itl"])    
    #fre_html = grab_html(lang_sites["fre"])

    print(eng_html)


