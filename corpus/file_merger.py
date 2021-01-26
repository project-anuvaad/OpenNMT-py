''' merge files (all src and all tgt ) -vertical join (step 2 after IITB split)

From IITB corpus currently we are keeping the data from these groups -chats, Movie Dialogs, general,Hi-Eng Word-Linkage,Admin Dictionary,
Admin Examples,Admin Definitions, ted talks, Indic Multi-Parallel, JudicialI, Govt Websites, Book Translations, 
aditional data- law commison, letters 
'''

def file_merger(file_names,merged_file_location):
    with open("{0}".format(merged_file_location), 'w') as outfile:
        for fname in file_names:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
             