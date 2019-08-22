import helper_functions.format_handler as format_handler

## below is for dropping duplicate from text file
def drop_duplicate(inFile,outFile):
    lines_seen = set() # holds lines already seen
    outfile = open("{0}".format(outFile), "w")
    for line in open("{0}".format(inFile), "r"):
        if line not in lines_seen: # not a duplicate
           outfile.write(line)
           lines_seen.add(line)
    outfile.close()

## separation into master corpus src and tgt for training. After this tokenization needs to be done(indic nlp, moses),then feed into OpenNMT
def separate_corpus(col_num,inFile,outFile):
    outfile = open("{0}".format(outFile), "w")
    delimiter = "\t"
    for line in open("{0}".format(inFile), "r"):
        # col_data.append(f.readline().split(delimiter)[col_num])
        outfile.write(str(line.split(delimiter)[col_num].replace('\n','')))
        outfile.write("\n")    
    outfile.close()

" remove duplicate and whitespaces from parallel corpus built after merging step "
def tab_separated_parllel_corpus(mono_corpus1,mono_corpus2,out_file):
    with open("{0}".format(mono_corpus1)) as xh:
      with open("{0}".format(mono_corpus2)) as yh:
        with open("{0}".format(out_file),"w") as zh:
          #Read first file
          xlines = xh.readlines()
          #Read second file
          ylines = yh.readlines()
      
          #Write to third file
          for i in range(len(xlines)):
            line = ylines[i].strip() + '\t' + xlines[i]
            zh.write(line)

