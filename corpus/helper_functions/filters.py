"filter anciliary functions"

def retain_duplicate(inFile,outFile):
    lines_seen = set() # holds lines already seen
    outfile = open("{0}".format(outFile), "w")
    for line in open("{0}".format(inFile), "r"):
        if line not in lines_seen: # not a duplicate
           print("c")
           lines_seen.add(line)
        else:
           outfile.write(line)
           lines_seen.add(line) 
    outfile.close()

def line_containing_link(in_file,out_file,out_file2):
    outfile = open("{0}".format(out_file), "w")
    outfile2 = open("{0}".format(out_file2), "w")
    for line in open("{0}".format(in_file), "r"):
      if  any(v in line for v in ['http','https','www','HTTP','HTTPS','WWW']):

          outfile.write(line)
      else:
          outfile2.write(line)            
    outfile.close()

