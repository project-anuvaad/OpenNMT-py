'''
Various model types to train depending on language pair
These are used against the model_type variable in the Jinkins training pipelines
'''

language_pair = {
    "en-hi" : { "en-to-hi":"english-hindi", "hi-to-en":"hindi-english" },
    "en-ta" : { "en-to-ta":"english-tamil", "ta-to-en":"tamil-english" },
    "en-gu" : { "en-to-gu":"english-gujarati", "gu-to-en":"gujarati-english" },
    "en-bn" : { "en-to-bn":"english-bengali", "bn-to-en":"bengali-english" },
    "en-mr" : { "en-to-mr":"english-marathi", "mr-to-en":"marathi-english" },
    "en-kn" : { "en-to-kn":"english-kannada", "kn-to-en":"kannada-english" },
    "en-te" : { "en-to-te":"english-telugu", "te-to-en":"telugu-english" },
    "en-ml" : { "en-to-ml":"english-malayalam", "ml-to-en":"malayalam-english" },
    "en-pu" : { "en-to-pu":"english-punjabi", "pu-to-en":"punjabi-english" },
    "en-ur" : { "en-to-ur":"english-urdu", "ur-to-en":"urdu-english" }
}