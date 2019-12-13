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

## German
t1: 1810-1860 \
t2: 1946-1990

- **DTA** (Deutsches Textarchiv, 2017)
  - period: 1600-1900
  - size: 370M tokens
- **BZ + ND** (Berliner Zeitung, 2018; Neues Deutschland, 2018)
  - period: 1945-1990
  - size: 340K newspaper pages


## Swedish
t1: 1800-1830 \
t2: 1900-1925

- **KubHist** 
    - period: 1645–1926
    - size: 1.1B tokens (soon 6.6B)
    
## Latin
t1: 200BC-0 \
t2: 0-2000

- **LatinISE** (McGillivray & Kilgarriff, 2013)
  - period: 2nd century BC - 21st century AD
  - size: 13M tokens


----
<h4>References</h4>
<ul>
<li>Yvonne Adesam, Dana Dannells, and Nina Tahmasebi. <a href="http://ceur-ws.org/Vol-2364/1_paper.pdf">Exploring the Quality of the Digital Historical Newspaper Archive KubHist</a>. In Proceedings of the Digital Humanities in the Nordic Countries 4th Conference, DHN 2019, Copenhagen, Denmark, March 7-9, 2019.</li>
<li>Berliner Zeitung. Diachronic newspaper corpus published by Staatsbibliothek zu Berlin [<a href="http://zefys.staatsbibliothek-berlin.de/index.php?id=155">online</a>]. 2018.</li>
<li>Lars Borin, Markus Forsberg and Johan Roxendal. 2012. <a href="https://www.aclweb.org/anthology/papers/L/L12/L12-1098/">Korp &ndash; the corpus infrastructure of Spr&aring;kbanken</a>. Proceedings of LREC 2012. Istanbul: ELRA, pages 474&ndash;478.</li>
<li>Mark Davies. 2002. <a href="https://www.english-corpora.org/coha/">The Corpus of Historical American English (COHA)</a>: 400 million words, 1810-2009. Brigham Young University.</li>
<li>Deutsches Textarchiv. Grundlage f&uuml;r ein Referenzkorpus der neuhochdeutschen Sprache. Herausgegeben von der Berlin-Brandenburgischen Akademie der Wissenschaften [<a href="http://www.deutschestextarchiv.de/">online</a>]. 2017.</li>
<li>Barbara McGillivray and Adam Kilgarriff. 2013. <a href="https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2506">Tools for historical corpus research, and a corpus of Latin</a>. In Paul Bennett, Martin Durrell, Silke Scheible, Silke and Richard J. Whitt (eds.), New Methods in Historical Corpus Linguistics, T&uuml;bingen. Narr.</li>
<li>Neues Deutschland. Diachronic newspaper corpus published by Staatsbibliothek zu Berlin [<a href="http://zefys.staatsbibliothek-berlin.de/index.php?id=156">online</a>]. 2018.</li>
</ul>
