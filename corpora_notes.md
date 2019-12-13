# Diachronic corpora
For each language, **two** time-specific corpora are provided for evaluation. 
Participants are required to predict the lexical semantic change of the target words between these two corpora.

### Format of SemEval corpora
- Sentences have been randomly shuffled within each corpus. 
- Each line in the distributed corpora contains one sentence. 
- All tokens are **lemmatized**. Lemmas are not lower-cased.
- **Punctuation** has been removed. 
- **Empty** (and one-word) **sentences** have been removed.
- Further preprocessing will be applied to the corpora, e.g. deleting **low-frequency words**.

## English
t1: 1810-1860 \
t2: 1960-2010

- **COHA** (Davies, 2002) 
  - period: 1810-2010
  - size: 400M tokens
  - https://www.english-corpora.org/coha/
- **Hansard** 
  - period: 1803-2005
  - size: 1.6B tokens
  - lemmatised
  - https://www.english-corpora.org/hansard/
- **COHA** (Davies, 2000) 
  - period: 1990-2017
  - size: 570M tokens
  - https://www.english-corpora.org/coca/
- more genre-specific corpora: https://www.english-corpora.org


## German
t1: 1810-1860 \
t2: 1946-1990

- **DTA** (Deutsches Textarchiv, 2017)
  - period: 1600-1900
  - size: 370M tokens
  - http://www.deutschestextarchiv.de/
- **BZ** (Berliner Zeitung, 2018)
  - period: 1945-1990
  - size: 168K newspaper pages
  - http://zefys.staatsbibliothek-berlin.de/index.php?id=155
- **ND** (Neues Deutschland, 2018)
  - period: 1946-1990
  - size: 168K newspaper pages
  - http://zefys.staatsbibliothek-berlin.de/index.php?id=156


## Swedish
t1: 1800-1830 \
t2: 1900-1925

- **KubHist** 
  - period: 1645–1926
  - size: 1.1B tokens (soon 6.6B)
- **The Swedish Sub-corpus of the Newspaper and Periodical Corpus**
  - period: 1770-1950
  - size: 3.5B tokens
  - https://tinyurl.com/swedish-sub-corpus
    
    
## Latin
t1: 200BC-0 \
t2: 0-2000

- **LatinISE** (McGillivray & Kilgarriff, 2013)
  - period: 2nd century BC - 21st century AD
  - size: 13M tokens
  - https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2506


# Annotated test sets

## English
- Gulordava & Baroni (2011)
  - 100 words (not lemmatized, e.g. both *woman* and *women* are included)
  - https://www.aclweb.org/anthology/W11-2508/

## German 
- DuRel (Schlechtweg et al., 2018)
  - 22 words
  - https://www.aclweb.org/anthology/N18-2027/
  
