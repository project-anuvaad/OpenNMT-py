# nmt-anuvada
We intent to collect parallel dataset for HINDI - ENGLISH language corpus. The primary usage, it to investigate translation accuracy of the mentioned corpus. 

#Detail about Corpora(original_data)
IITB Hindi-English parallel corpus(approx size 1.5M) contains the data from the following domain:
GNOME			         1
KDE			            145706
Quran			        242933
Chats			        430013
Movie Dialogs		    434711
General			        438933
Hi-Eng Word-Linkage	    712818
Admin Dictionary	    887993
Admin Examples		    954457
Admin Definitions	    1001292
TED Talks		        1047815
Indic Multi-Parallel	1090398
Judicial I		        1100747
Judicial II		        1105754
Govt Websites		    1109481
Wikipedia		        1232841
Book Translations	    1265704
Govt Website II		    1492827
	     		        1561840
# Currently we are using following corpus:
 1. From IITB corpus currently we are keeping the data from these groups -chats, Movie Dialogs, general,Hi-Eng Word-Linkage,Admin Dictionary,
	Admin Examples,Admin Definitions, ted talks, Indic Multi-Parallel, JudicialI and II, Govt Websites I and II, Book Translations, Wikipedia, Book translation,
 2. Aditional data- 
 	law commison(41k), 
	letters(5o sentences),
	Supreme Court Judgement Preface Specifc(1.1k) 

 Final Corpus before pre-processing: 1174323
# steps:
1.fileSpliter - to get splitted domain wise files from IITB corpus
2.fileMerger - to merge all src and tgt data
3.fileCleaner - remove duplicates and get final src and tgt files
4.format_handler - for handling special character, numbers, dates etc.
