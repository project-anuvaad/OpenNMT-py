'''
Various regex patterns used to support translation
'''

patterns = {
    "p1" : { "regex":r'(\d+,)\s(\d+)', "description":"remove space between number separated by ," },
    "p2" : { "regex":r'(\d+.)\s(\d+)', "description":"remove space between number separated by ." },
    "p3" : { "regex":r'\d+', "description":"indentify numbers in a string" },
    "p4" : { "regex":r'(NnUuMm.,)\s(NnUuMm+)', "description":"remove space between number separated by ," },
    "p5" : { "regex":r'(NnUuMm..)\s(NnUuMm+)', "description":"remove space between number separated by ." }
}

hindi_numbers = ['०','१','२','३','४','५','६','७','८','९','१०','११','१२','१३','१४','१५','१६','१७','१८','१९','२०','२१','२२','२३','२४','२५','२६','२७','२८','२९','३०']
