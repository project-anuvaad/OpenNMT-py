import sentencepiece as spm
import sys, getopt 


def train_spm(input_file,prefix,vocab_size,model_type):  
    try:
        spm.SentencePieceTrainer.Train('--input={} --model_prefix={} --vocab_size={} --model_type={}'.format(input_file,prefix,vocab_size,model_type))
        return
    except:
        print("something went wrong!")
        print("Unexpected error:", sys.exc_info()[0])
        return

def encode_as_pieces(load_model,src_file,tgt_file):
    # makes segmenter instance and loads the model file (m.model)
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(load_model)
        with open(src_file) as xh:    
            with open(tgt_file,"w") as outfile:
                xlines= xh.readlines()
        
                for i in range(len(xlines)):
                    encLine = sp.encode_as_pieces(xlines[i])
                    outfile.write(str(encLine))
                    outfile.write("\n")
    except:
        print("something went wrong!")
        print("Unexpected error:", sys.exc_info()[0])
        return
    

def decode_as_pieces(load_model,src_file,tgt_file):
    try:
        print("decoding")
        spH = spm.SentencePieceProcessor()
        spH.load(load_model)
        with open(src_file) as xh:    
            with open(tgt_file,"w") as outfile:
                xlines= xh.readlines()
        
                for i in range(len(xlines)):
                    if not xlines[i].startswith("["):
                        print("here1")
                        xlines[i] = "["+xlines[i].rstrip()
                   
                    if not xlines[i].rstrip().endswith("]"):
                        print("here2")
                        print(xlines[i])
                        xlines[i] = xlines[i].rstrip()+"]" 
                        print(xlines[i])         
                    encLine = spH.DecodePieces(eval(xlines[i]))
                    outfile.write(str(encLine))
                    outfile.write("\n")
    except:
        print("something went wrong!")
        print("Unexpected error:", sys.exc_info()[0])
        return
    
if sys.argv[1] == "train":
    train_spm(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
elif sys.argv[1] == "encode":
    encode_as_pieces(sys.argv[2],sys.argv[3],sys.argv[4])
elif sys.argv[1] == "decode":
    decode_as_pieces(sys.argv[2],sys.argv[3],sys.argv[4])
else:
    print("invalid request",sys.argv)           
