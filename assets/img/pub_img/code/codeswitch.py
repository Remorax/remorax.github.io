import argparse, pickle, re
from glob import glob
import numpy as np
from collections import Counter
from Levenshtein import distance as levenshtein_distance

exists_trans, notexists_trans = 0, 0
correct_inflections, incorrect_inflections, correct_lemmas, total_substitutions = 0, 0, 0, 0
inflection_error, missed_error, total_calls, avg_choices, avg_tot = 0, 0, 0, 0, 0
muse_samples, missed_entries, logs_verbose = [], [], []
replaced_Counter = dict()

def clean_emojis(string):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F" # emoticons
        u"\U0001F300-\U0001F5FF" # symbols & pictographs
        u"\U0001F680-\U0001F6FF" # transport & map symbols
        u"\U0001F1E0-\U0001F1FF" # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", 
        flags=re.UNICODE
    )
    
    return emoji_pattern.sub(r'', string)
def intersect_muse_translations(word, trans):
    global inflection_error, missed_error, total_calls
    total_calls += 1
    word, trans = word.lower(), trans.lower()
    if (word,trans) in muse_translations:
        if args.randomize_inflections:
            return True
        if len(muse_translations[(word,trans)]) != 1:
            inflection_error +=1
        return len(muse_translations[(word,trans)]) == 1
    # print (word.lower(),trans.lower(), "not found")
    missed_error += 1
    missed_entries.append((word, trans))
    return False

def codeswitch(sent_obj):
    global exists_trans, notexists_trans, correct_inflections, incorrect_inflections, correct_lemmas, total_substitutions, muse_samples, avg_choices, avg_tot, replaced_Counter

    sent, res, ref_sent = sent_obj # res is a list of tuples (sent_idx, pos, lemma, text, babelID)
    if args.prefer_roots:
        # Sort choices by levenshtein distance between lemma and text
        choices = [(tup[0], tup[1], tup[2], tup[3], tup[4], levenshtein_distance(tup[2], tup[3])) for tup in res]
        choices = [tup[:-1] for tup in sorted(choices, key=lambda l:l[-1])]
    else:
        choices = np.random.permutation(res)
    total = len([elem for elem in choices if elem[1]!="PUNCT"])
    output = [tup[-2] for tup in res]
    choices = [elem for elem in choices if elem[1]!="PUNCT" and elem[-1][:2] == "bn" and elem[-1] in results]
    if args.exclude_propn:
        choices = [elem for elem in choices if elem[1]!="PROPN"]
    avg_choices += len(choices)
    avg_tot += total
    if not choices:
        return False
    replaced = set()
    muse_temp = dict()
    log_sentence = list(output)
    translations, synonyms, similar_tos, hypernyms = [], [], [], []
    logs = {}
    
    # Main loop for choosing candidate translations from a sentence
    for choice in choices:
        sent_idx, pos, lemma, text, babelID = choice
        log_sentence[int(sent_idx)] = f"{text} [{babelID}]"
        trans = None
        if not results[babelID]["translations"]:
            # Use synonyms, similar words, or hypernyms if translations are not available
            if results[babelID]["synonyms"] and args.synonyms_backoff:
                trans = np.random.choice(results[babelID]["synonyms"])
                synonyms.append((sent_idx, trans))
            elif (results[babelID]["similar_tos"] or results[babelID]["also_sees"]) and args.similars_backoff:
                similars = results[babelID]["similar_tos"] + results[babelID]["also_sees"]
                trans = np.random.choice(list(similars))
                similar_tos.append((sent_idx, trans))
            elif results[babelID]["hypernyms"] and args.hypernyms_backoff:
                hyps = results[babelID]["hypernyms"]
                trans = np.random.choice(list(hyps))
                hypernyms.append((sent_idx, trans))
        else:
            exists_trans += 1
            if args.use_muse and (text.lower() != lemma.lower()):
                # Intersection with MUSE lexicons for morphological prediction
                muse_cands = [(cand, muse_translations[(text.lower(), cand.lower())]) for cand in list(results[babelID]["translations"]) if intersect_muse_translations(text, cand)]
                if not muse_cands:
                    incorrect_inflections += 1
                    trans = np.random.choice(list(results[babelID]["translations"]))
                else:
                    correct_inflections += 1
                    index = np.random.choice(list(range(len(muse_cands))))
                    cand, transes = muse_cands[index]
                    trans = np.random.choice(transes)
                    if args.include_pos:
                        if pos == args.include_pos:
                            muse_temp[sent_idx] = (text, lemma, cand, trans)
                    else:
                        muse_temp[sent_idx] = (text, lemma, cand, trans)
            else:
                correct_lemmas += 1
                trans = np.random.choice(list(results[babelID]["translations"]))
            translations.append((sent_idx, trans))
        logs[sent_idx] = (text, lemma, pos, babelID, trans) if trans else (text, lemma, pos, babelID, "")
        if not trans:
            notexists_trans += 1
            continue

    replacements = translations
    if args.synonyms_backoff:
        replacements += synonyms
    elif args.similars_backoff:
        replacements += similar_tos
    elif args.hypernyms_backoff:
        replacements += hypernyms
    muse_replaced = 0
    muse_output = list(output)
    
    # Choosing which words to replace
    for rep in replacements:
        if len(replaced)/total >= float(args.replacement_ratio):
            break
        sent_idx, trans = rep
        total_substitutions += 1
        if sent_idx not in replaced:
            output[int(sent_idx)] = trans # Code-switching
            replaced.add(int(sent_idx))
            
        # For logging purposes
        if sent_idx in muse_temp:
            muse_replaced += 1
            (text, lemma, cand, trans) = muse_temp[sent_idx]
            muse_output[int(sent_idx)] = f"{trans} ({text}->{lemma}->{cand}->{trans})"
            log_sentence[int(sent_idx)] += f"({text}->{lemma}->{cand}->{trans})"
        else:
            (text, lemma, pos, babelID, trans) = logs[sent_idx]
            if (text, pos) not in replaced_Counter:
                replaced_Counter[(text, pos)] = Counter()
            replaced_Counter[(text, pos)][trans] += 1
            log_sentence[int(sent_idx)] += f"({text}->{lemma}->{trans})"
    if muse_replaced:
        muse_samples.append((muse_replaced, " ".join(muse_output)))
    if args.verbose:
        logs_verbose.append(" ".join(log_sentence))
    return " ".join(output)


parser = argparse.ArgumentParser()
parser.add_argument('--disambiguated_corpus', help="Provide WSD results", type=str)
parser.add_argument('--bn_translations', help="Provide translations extracted from BabelNet", type=str)
parser.add_argument('--src_lang', help="Provide source language ISO code", type=str)
parser.add_argument('--save_cs_path', help="Path for saving code-switched corpus (input to encoder)", type=str)
parser.add_argument('--save_tgt_path', help="Path for saving reference corpus (target for decoder)", type=str)
parser.add_argument('--muse_dir', help="Path for MUSE dictionaries for accurate in-context prediction (used with --use_muse)", type=str, nargs="?", default='')
parser.add_argument('--seed', help="Random seed", type=str, nargs="?", default='0')

parser.add_argument('--replacement_ratio', help="Provide replacement ratio", type=str, nargs="?", default='0.1')
parser.add_argument('--prefer_roots', help="Whether or not to prefer root words for replacement", action="store_true", default=False)
parser.add_argument('--include_pos', help="Which POS tag to code-switch in", type=str, nargs="?", default='')
parser.add_argument('--verbose', help="Logs output of code-switching process", action="store_true", default=False)
parser.add_argument('--exclude_propn', help="Whether or not to exclude proper nouns from substitution", action="store_true", default=False)
parser.add_argument('--use_muse', help="Whether or not to intersect with MUSE translations for accurate in-context prediction", action="store_true", default=False)
parser.add_argument('--randomize_inflections', help="Whether or not to randomize inflections when intersecting with MUSE dictionaries", action="store_true", default=False)
parser.add_argument('--synonyms_backoff', help="Whether or not to backoff to synonyms", action="store_true", default=False)
parser.add_argument('--similars_backoff', help="Whether or not to backoff to similar_to words", action="store_true", default=False)
parser.add_argument('--hypernyms_backoff', help="Whether or not to backoff to hypernyms", action="store_true", default=False)

args = parser.parse_args()

input_corpus = pickle.load(open(args.disambiguated_corpus, "rb")) # WSD results
results = pickle.load(open(args.bn_translations, "rb")) # BabelNet translations

print (f"Hypernyms backoff: {args.hypernyms_backoff} Synoynms backoff: {args.synonyms_backoff} Similars backoff: {args.similars_backoff}")

# Preprocessing MUSE dictionaries for morphological prediction
if args.muse_dir:
    muse_translations = {}
    for dict_file in glob(f"{args.muse_dir}/{args.src_lang.lower()}-*.lem.txt"):
        for line in open(dict_file):
            src_word, src_lemma, tgt_word, tgt_lemma = line.lower().strip().split("\t")
            if (src_word, tgt_lemma) in muse_translations:
                muse_translations[(src_word, tgt_lemma)].append(tgt_word)
            else:
                muse_translations[(src_word, tgt_lemma)] = [tgt_word]

np.random.seed(int(args.seed)) # Setting random seed for reproducibility

print ("Code-switching sentences...")
final_output, final_input = [], []
for sent_obj in input_corpus:
    noised = codeswitch(sent_obj)
    if not noised:
        continue
    final_output.append(clean_emojis(noised))
    final_input.append(clean_emojis(sent_obj[-1]))

open(args.save_cs_path, "w+").write("\n".join(final_output))
open(args.save_tgt_path, "w+").write("\n".join(final_input))

open(f"muse_samples-{args.src_lang}_{args.include_pos}.txt", "w+").write("\n".join([sample[1] for sample in muse_samples[:1000]])+"\n")
if args.verbose:
    print (logs_verbose[:100])
    pickle.dump(logs_verbose, open(f"logs-{args.src_lang}.pkl", "wb"))
pickle.dump(missed_entries, open(f"missed-{args.src_lang}.pkl", 'wb'))
pickle.dump(replaced_Counter, open(f"replaced-{args.src_lang}_it.pkl", 'wb'))

print (f"============================ STATS for language {args.src_lang} in {args.disambiguated_corpus} ==========================================")
print (f"Match percentage in BabelNet: {float(exists_trans/(exists_trans+notexists_trans))}")
print (f"Avg. Disambiguation percentage per sentence: {float(avg_choices/avg_tot)}")
if args.muse_dir:
    print (f"Substitution Accuracies (Correct lemma/Correct inflection/Incorrect inflection): {float(correct_lemmas/total_substitutions),float(correct_inflections/total_substitutions),float(incorrect_inflections/total_substitutions)}")
    print (f"Cause of Errors (Inflection Error/Missed Error): {float(inflection_error/total_calls),float(missed_error/total_calls)}")
print ("All done. Sample sentences: \n", np.random.choice(final_output, 10))
