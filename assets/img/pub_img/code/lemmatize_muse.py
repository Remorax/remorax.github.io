import sys
from trankit import Pipeline

dict_file = sys.argv[1]
if ".lem." in dict_file or "romance_multi" in dict_file:
    sys.exit(f"Exiting because {dict_file} is a lemmatized file")
src_lang = dict_file.split("/")[-1].split("-")[0]
tgt_lang = dict_file.split("/")[-1].split("-")[1].split(".")[0]

langs={'en': 'english', 'it': 'italian', 'es':'spanish', 'fr':'french', 'ro':'romanian', 'pt':'portuguese', 'fa': 'persian', 'hi': 'hindi'}
src_words, tgt_words = [], []
for line in open(dict_file):
    if not line.strip():
        continue
    try:
        src_word, tgt_word = line.split(" ")
    except:
        src_word, tgt_word = line.split("\t")
    src_words.append(src_word.strip())
    tgt_words.append(tgt_word.strip())


p = Pipeline(langs[src_lang])
p.add(langs[tgt_lang])

p.set_active(langs[src_lang])

def lemmatize(x, lang):
    output = p(x, is_sent=True)
    if 'tokens' not in output or not output['tokens']:
        print (f"Cant detect tokens in {x} from lang {lang}. Output: {output}")
        return x
    if 'lemma' not in output['tokens'][0]:
        return output['tokens'][0]['expanded'][0]['lemma']
    return output['tokens'][0]['lemma']

src_lemmas={word: lemmatize(word, src_lang) for word in set(src_words)}
print ("Src Lemmas not found for: ", [word for word in src_lemmas if not src_lemmas[word]])
src_lemmas={word: src_lemmas[word] if src_lemmas[word] else word for word in src_lemmas}

p.set_active(langs[tgt_lang])

tgt_lemmas={word: lemmatize(word, tgt_lang) for word in set(tgt_words)}
print ("Tgt Lemmas not found for: ", [word for word in tgt_lemmas if not tgt_lemmas[word]])
tgt_lemmas={word: tgt_lemmas[word] if tgt_lemmas[word] else word for word in tgt_lemmas}

processed_list = []
for src_word, tgt_word in zip(src_words, tgt_words):
    processed_list.append("\t".join([src_word, src_lemmas[src_word], tgt_word, tgt_lemmas[tgt_word]]))
folder = "/".join(dict_file.split("/")[:-1])
open(f"{folder}/{src_lang}-{tgt_lang}.lem.txt","w+").write("\n".join(processed_list) + "\n")
