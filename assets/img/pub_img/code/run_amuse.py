import glob, time, tqdm, os
import requests, pickle, sys
from pathlib import Path
from urllib.parse import urlencode

# Retrieve a single page and report the URL and contents
results = []
def handle_query(l):
    data = [{"text":line[0], "lang":lang} for line in l]
    refs = [line[1] for line in l]
    x = requests.post(url, json=data)
    try:
        res = [(l[j],[(elem['index'], elem['pos'], elem['lemma'], elem['text'], elem['bnSynsetId']) for elem in i['tokens']], tgt) for (tgt,(j,i)) in zip(refs,enumerate(x.json()))]
    except:
        print (x.json())
    return res

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

# Uncomment this with path to data!!
# mono_path=<path_to_mono>

src_file=sys.argv[1]
ref_file=sys.argv[2]
ref_fp=open(mono_path + "/" + ref_file)
port=sys.argv[3]
lang = src_file.split(".")[-1][:2].upper()

f= f"{mono_path}/{src_file}"
url = f"http://0.0.0.0:{port}/api/model"

lines = []
for line in open(f):
    ref_line=next(ref_fp).strip()
    line = line.strip()
    lines.append((line,ref_line))
window_size=100000
start_offset = int(sys.argv[4]) if len(sys.argv) > 4 else 0
end_offset = int(sys.argv[5]) if len(sys.argv) > 4 else len(lines)
t = time.time()

thread_count = 5
mono_path=os.path.dirname(f)
# We can use a with statement to ensure threads are cleaned up promptly
os.makedirs(mono_path + f"/checkpoints-{lang}/", exist_ok=True)

for i in tqdm.tqdm(range(0, len(lines), window_size)):
    part_id = int(i/window_size)
    if i < start_offset:
        print (f"Offset Skipping part {part_id}...")
        continue
    if i >= end_offset:
        print (f"Offset Skipping part {part_id}...")
        continue
    
    if Path(mono_path + f"/checkpoints-{lang}/part{part_id}.pkl").exists():
        print (f"Skipping part {part_id}...")
        continue
    tmp_lines = lines[i:i+window_size]
    part_results = handle_query(tmp_lines)
    pickle.dump(part_results, open(mono_path + f"/checkpoints-{lang}/part{part_id}.pkl", "wb"))

results = []
for checkpoint in glob.glob(mono_path + f"/checkpoints-{lang}/*.pkl"):
    results.extend(pickle.load(open(checkpoint, "rb")))
pickle.dump(results, open(mono_path + f"/results-{lang}.pkl", "wb"))
print ("All done. ", t-time.time())
