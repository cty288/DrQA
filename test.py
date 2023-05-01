import gensim
import numpy as np

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


word_limit = 2000000

with open('word_vectors_limited.txt', 'w', encoding='utf-8') as f:
  
    available_words = min(len(model.key_to_index), word_limit)
  
    f.write(f"{available_words} {model.vector_size}\n")
    
    for key in model.index_to_key[:word_limit]:
        word_vector = ' '.join([str(num) for num in model[key]])
        f.write(f"{key} {word_vector}\n")
    

with open('excluded_words.txt', 'w', encoding='utf-8') as f:
    available_words = len(model.key_to_index)
    
    if word_limit < available_words:
        for key in model.index_to_key[word_limit:]:
            f.write(f"{key}\n")