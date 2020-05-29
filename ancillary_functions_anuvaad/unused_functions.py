"lookup table"
def lookup_table(model_id,token):
    if model_id in [1]:
        with open("lookup_dictionary_eng_hin.txt",encoding ='utf-16') as xh:
            xlines = xh.readlines()
            for i in range(len(xlines)):
                if xlines[i].split('|||')[0] == token.strip():
                    token = xlines[i].split('|||')[1].strip()
                else:
                    token = ""    
    else:
        token = ""                
    
    return token                
'''
    @app.route("/download-src", methods=['GET'])
    def get_file():
        """Download a file."""
        out = {}
        type = request.args.get('type')
        print(type)
        if  not type:
            out['status'] = statusCode["TYPE_MISSING"]
            return jsonify(out)
        if type not in ['Gen','LC','GoI','TB']:
            out['status'] = statusCode["INVALID_TYPE"]
            return jsonify(out)  

        try:
            logger.info("downloading the src %s.txt file" % type)
            return send_file(os.path.join(API_FILE_DIRECTORY,'source_files/', '%s.txt' % type), as_attachment=True)
        except:
            out['status'] = statusCode["SYSTEM_ERR"]
            logger.info("Unexpected error: %s"% sys.exc_info()[0])
            return jsonify(out) 

    @app.route("/upload-tgt", methods=["POST"])
    def post_file():
        """Upload a file."""
        print(request.files)
        out = {}
        if 'file' not in request.files:
            out['status'] = statusCode["FILE_MISSING"]
            return jsonify(out)
        print(request.form)    
        if 'type' not in request.form:
            out['status'] = statusCode["TYPE_MISSING"]
            return jsonify(out)  
        if request.form['type'] not in ['Gen','LC','GoI','TB']:
            out['status'] = statusCode["INVALID_TYPE"]
            return jsonify(out)  

        try:
            file = request.files['file']
            tgt_file_loc = os.path.join(API_FILE_DIRECTORY,'target_files/', '%s.txt' % request.form['type'])
            tgt_ref_file_loc = os.path.join(API_FILE_DIRECTORY,'target_ref_files/', '%s.txt' % request.form['type'])
            file.save(tgt_file_loc)

            if os.path.exists("bleu_out.txt"):
               os.remove("bleu_out.txt")
            
            os.system("perl ./tools/multi-bleu-detok.perl ./%s < ./%s > bleu-detok.txt" %(tgt_ref_file_loc,tgt_file_loc))
            os.system("python ./tools/calculatebleu.py ./%s ./%s" %(tgt_file_loc,tgt_ref_file_loc))            
            os.remove(tgt_file_loc)
            logger.info("Bleu calculated and file removed")
            with open("bleu-detok.txt") as zh:
                out['status'] = statusCode["SUCCESS"]
                out['response_body'] = {'bleu_for_uploaded_file':float(', '.join(zh.readlines())),
                                        'openNMT_custom':bleu_results.OpenNMT_Custom, 'google_api': bleu_results.GOOGLE_API 
                                        }
        except:
            out['status'] = statusCode["SYSTEM_ERR"]
            logger.info("Unexpected error: %s"% sys.exc_info()[0])
        
        return jsonify(out)
'''