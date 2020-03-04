-------------------------------------
Siehe unten für die deutsche Version.
-------------------------------------

Diachronic Usage Relatedness (DURel) - Test Set and Annotation Data


This data collection supplementing the paper referenced below contains:


   - a semantic change test set with 22 German lexemes divided into two classes: (i) lexemes for which the authors found innovative or (ii) reductive meaning change occuring in Deutsches Textarchiv (DTA) in the 19th century. (Note that for some lexemes the change is already observable slightly before 1800 and some lexemes occur more than once in the test set (see paper).) It comes as a tab-separated csv file where each line has the form

     lemma	POS	type	description	earlier	later	delta_later	compare	frequency_1750-1800/1850-1900	source

    The columns 'earlier' and 'later' contain the mean of all judgments for the respective word. The columns 'delta_later' and 'compare' contain the predictions of the annotation-based measures of semantic change developed in the paper;

   - the full annotation table as annotators received it and a results table with rows in the same order. The result table comes in the form of a tab-separated csv file where each line has the form

     lemma	date1	date2	group	annotator1	annotator2	annotator3	annotator4	annotator5	mean	comments1	comments2	comments3	comments4	comments5

     The columns 'date1' and 'date2' contain the date of the first and second use in the row. 'mean' contains the mean of all judgments for the use pair in this row without 0-judgments;

   - the annotation guidelines in English and German;
   - data visualization plots. 

Find more information in 

Dominik Schlechtweg, Sabine Schulte im Walde, Stefanie Eckmann. 2018. Diachronic Usage Relatedness (DURel): A Framework for the Annotation of Lexical Semantic Change. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL HLT). New Orleans, Louisiana USA 2018.

The resources are freely available for education, research and other non-commercial purposes. More information can be requested via email to the authors.

-------
Deutsch
-------

Diachroner Wortverwendungsbezug (DURel) - Test Set und Annotationsdaten


Diese Datensammlung ergänzt den unten zitierten Artikel und enthält folgende Dateien:


   - ein Test set für semantischen Wandel mit 22 deutschen Lexemen, die in zwei Klassen fallen: (i) Lexeme, für die die Autoren innovativen oder (ii) reduktiven Bedeutungswandel im Deutschen Textarchiv (DTA) für das 19. Jahrhundert festgestellt haben. (Für einige Lexeme ist der Wandel schon etwas vor 1800 zu beobachten und manche Lexeme kommen mehr als einmal im Test set vor (siehe Artikel).) Hierbei handelt es sich um eine tab-separierte CSV-Datei, in der jede Zeile folgende Form hat:

     Lexem	Wortart	Klasse	Beschreibung	earlier	later	delta_later	compare	Frequenz_1750-1800/1850-1900	Quelle

    Die Spalten 'earlier' und 'later' enthalten den Mittelwert der Bewertungen für das jeweilige Wort. Die Spalten 'delta_later' und 'compare' enthalten die Vorhersagen der annotationsbasierten Maße für semantischen Wandel, die im Artikel entwickelt werden;

   -  Die Annotationstabelle, wie sie die Annotatoren erhalten haben, und eine Ergebnistabelle mit Zeilen in derselben Reihenfolge. Die Ergebnistabelle ist eine tab-separierte CSV-Datei, in der jede Zeile folgende Form hat:

     Lexem	Datum1	Datum2	Gruppe	Annotator1	Annotator2	Annotator3	Annotator4	Annotator5	Mittelwert	Kommentar1	Kommentar2	Kommentar3	Kommentar4	Kommentar5

     Die Spalten 'Datum1' und 'Datum2' enthalten das Datum der ersten bzw. der zweiten Wortverwendung in der Zeile. 'Mittelwert' enthält den Mittelwert aller Bewertungen für das Verwendungspaar dieser Zeile ohne 0-Bewertungen;

   - die Annotationsrichtlinien auf Deutsch und Englisch;
   - Visualisierungsplots der Daten.  
Mehr Informationen finden Sie in

Dominik Schlechtweg, Sabine Schulte im Walde, Stefanie Eckmann. 2018. Diachronic Usage Relatedness (DURel): A Framework for the Annotation of Lexical Semantic Change. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL HLT). New Orleans, Louisiana USA 2018.

Die Ressourcen sind frei verfügbar für Lehre, Forschung sowie andere nicht-kommerzielle Zwecke. Für weitere Informationen schreiben Sie bitte eine E-Mail an die Autoren.