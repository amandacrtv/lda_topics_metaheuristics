import traceback
from langdetect import detect, DetectorFactory
import math
import regex
from emoji import UNICODE_EMOJI

DetectorFactory.seed = 0

def detect_lang(text):
    if isinstance(text, str):
        try:
            return detect(text)
        except: 
            return 'invalid'
    else:
        return 'invalid'

def classify_lang(df, row, filename):
    # df = pd.read_csv(filename, usecols=cols)
    df['lang'] = [detect_lang(text) for text in df[row]]
    return df
    # df.to_csv('lang_' + filename, index=False) 

def filter_by_row(df, row, value):
    df = df.drop(df[df[row] != value].index)
    return df

def is_empty(text):
    return (text is None 
        or (isinstance(text, float) and math.isnan(text)) 
        or (isinstance(text, str) and len(text) == 0))

def whole_words_count(text): 
    return len(text.split())

def total_chars_count(text):
    return len(text)

def numbers_count(text):
    return sum(map(str.isdigit, text))

def letter_count(text):
    return sum(map(str.isalpha, text))

def spaces_count(text):
    return sum(map(str.isspace, text))

def has_url(text):
    return ('http' in text and '://' in text) or ('www.' in text)

def emoji_count(text):
    data = regex.findall(r'\X', text)
    return sum(map(lambda x: any(char in UNICODE_EMOJI for char in x), data))

def classify_text_content_csv(df, row):
    is_empty_l = []
    whole_words_count_l = []
    total_chars_count_l = []
    numbers_count_l = []
    letter_count_l = []
    special_chars_l = []
    has_url_l = []
    emoji_count_l = []

    for text in df[row]:
        try:
            if is_empty(text):
                is_empty_l.append(True)
                whole_words_count_l.append('')
                total_chars_count_l.append('')
                numbers_count_l.append('')
                letter_count_l.append('')
                special_chars_l.append('')
                has_url_l.append('')
                emoji_count_l.append('')
            else:
                total_chars = total_chars_count(text)
                numbers = numbers_count(text)
                letters = letter_count(text)
                is_empty_l.append(False)
                whole_words_count_l.append(whole_words_count(text))
                total_chars_count_l.append(total_chars)
                numbers_count_l.append(numbers)
                letter_count_l.append(letters)
                special_chars_l.append(total_chars - numbers - spaces_count(text) - letters)
                has_url_l.append(has_url(text))
                emoji_count_l.append(emoji_count(text))
        except Exception as e:
            print('Error while classifing text {}'.format(text))
            traceback.print_exc()
            quit()

    df['is_empty'] = is_empty_l
    df['whole_words'] = whole_words_count_l
    df['total_chars'] = total_chars_count_l
    df['numbers'] = numbers_count_l
    df['letters'] = letter_count_l
    df['special_chars'] = special_chars_l
    df['has_url'] = has_url_l
    df['emoji_count'] = emoji_count_l
    return df