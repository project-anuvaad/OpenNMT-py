import translation_util.interactive_translate as interactive_translation



inputs =  {
        "src": "It was pleaded that plaintiff was never installed Shankaracharya of Jyotirmath/Jyotishpeeth and never exercised duties as such.",
        "target_prefix": "यह दलील दी गई कि वादी को कभी भी ज्योतिर्मठ/ज्योतिष्पीठ के शंकराचार्य के रूप में स्थापित नहीं किया गया था और उन्होंने"
    }
x = interactive_translation.interactive_translation(inputs)

print(x)

# interactive_translation.model_conversion()