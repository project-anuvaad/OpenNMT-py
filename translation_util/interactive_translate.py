import ctranslate2
import anuvada
import tools.sp_enc_dec as sp
from config import sentencepiece_model_loc as sp_model
import sys


MODEL_BASE_PATH = "available_models/interactive_models/test/"


def model_conversion():
    try:
        converter = ctranslate2.converters.OpenNMTPyConverter("available_models/model_en-hi_exp-5.6_2019-12-09-model_step_150000.pt")
        print("x")

        output_dir = converter.convert(
                     "available_models/interactive_models/test",         # Path to the output directory.
                     "TransformerBase",   # A model specification instance from ctranslate2.specs.
                     vmap=None,               # Path to a vocabulary mapping file.
                     quantization=None,       # Weights quantization: "int8" or "int16".
                     force=False)
        print("done")               
    except Exception as e:
        print(e)


def interactive_translation(inputs):
    try:
        translator = ctranslate2.Translator(MODEL_BASE_PATH)

        i_tok = anuvada.moses_tokenizer(inputs['src'])
        i_enc = str(sp.encode_line(sp_model.english_hindi["ENG_EXP_5.6"],i_tok))

        tp_tok = anuvada.indic_tokenizer(inputs['target_prefix'])
        tp_enc = str(sp.encode_line(sp_model.english_hindi["HIN_EXP_5.6"],tp_tok))
        
        i_final = format_converter(i_enc)
        tp_final = format_converter(tp_enc)
        tp_final[-1] = tp_final[-1].replace(']',",")
        print("succes till here")

        out = translator.translate_batch([i_final],beam_size = 5, target_prefix = [tp_final])    
        tok = out[0][0]['tokens']  
        y_1 = " ".join(tok)
        translation = sp.decode_line(sp_model.english_hindi["HIN_EXP_5.6"],y_1)
        translation = anuvada.indic_detokenizer(translation)
        return translation
    except Exception as e:
        print(e)
        print(sys.exc_info()[0])

def format_converter(input):
    inp_1 = input.split(', ')
    inp_2 = [inpt+',' if inpt != inp_1[-1] else inpt for inpt in inp_1 ]
    return inp_2



inp = "['▁It', '▁was', '▁pleaded', '▁that', '▁plaintiff', '▁was', '▁never', '▁installed', '▁Shankaracharya', '▁of', '▁Jyotirmath', '▁/', '▁Jyotishpeeth', '▁and', '▁never', '▁exercised', '▁duties', '▁as', '▁such', '▁.']"


# inputs =  {
#         "src": "It was pleaded that plaintiff was never installed Shankaracharya of Jyotirmath/Jyotishpeeth and never exercised duties as such.",
#         "target_prefix": ["['▁यह',", "'▁दलील',", "'▁दी',", "'▁गई',", "'▁कि',", "'▁अभियोगी',"]
#     }
# interactive_translation(inputs)               