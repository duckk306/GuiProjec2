# from underthesea import word_tokenize, pos_tag, sent_tokenize
from pyvi.ViTokenizer import tokenize
from pyvi import ViTokenizer
from underthesea import word_tokenize
import re
from importlib import resources as impresources
from . import files

def remove_emojis(text):
    emojicon_file = impresources.files(files) / 'emojicon.txt'
    with open(emojicon_file, 'r', encoding='utf-8') as f:
        emojicons = [w.strip() for w in f.readlines() if w.strip()]
        
    for emo in emojicons:
        text = text.replace(emo, ' ')
    return text

def normalize_teencode(text):
    teencode_file = impresources.files(files) / 'teencode.txt'
    
    teencode_map = {}
    with open(teencode_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                teencode_map[parts[0]] = " ".join(parts[1:])
            
    for key, val in teencode_map.items():
        text = re.sub(rf'\b{re.escape(key)}\b', val, text)
    return text

def remove_special_chars(text):
    text = re.sub(r'[^\w\s]', ' ', text)  # loại ký tự đặc biệt
    text = re.sub(r'\s+', ' ', text).strip()  # loại khoảng trắng thừa
    return text

def remove_stopwords(text):
    stop_word_file = impresources.files(files) / 'vietnamese-stopwords.txt'
    
    with open(stop_word_file, 'r', encoding='utf-8') as f:
        stopwords = set([w.strip() for w in f.readlines() if w.strip()])

    tokens = word_tokenize(text, format="text").split()
    tokens = [t for t in tokens if t not in stopwords]
    return ' '.join(tokens)

def clean_text(text):
    text = str(text).lower()
    text = remove_emojis(text)
    text = normalize_teencode(text)
    text = remove_special_chars(text)
    text = remove_stopwords(text)
    return text

def clean_text_fromdf(data):
    data = data[['id', 'Tiêu đề','Mô tả chi tiết']]
    data['Content'] = data['Mô tả chi tiết'].apply(lambda x: ' '.join(x.split()[:200]))
    data['clean_text'] = data['Content'].apply(clean_text)
    return data
#--------------------------------
