from collections import Counter, defaultdict
import re
import csv
import os


def normalize_token(tok: str) -> str:
    tok = tok.strip()
    tok = tok.strip('.,;:\"\'"()[]')
    tok = tok.replace('*', '')
    return tok


def is_cyrillic_letter(ch: str) -> bool:
    return 'а' <= ch.lower() <= 'я' or ch.lower() == 'ё'


with open("nomenclature.csv", "r", encoding="utf-8") as file:
    lines = [l.strip() for l in file.read().splitlines() if l.strip()]

token_counts = Counter()
start_counts = Counter()
end_counts = Counter()
lines_tokens = []

for line in lines:
    
    toks = [normalize_token(t) for t in line.split() if normalize_token(t)]
    lines_tokens.append(toks)
    if toks:
        start_counts[toks[0]] += 1
        end_counts[toks[-1]] += 1
    for t in toks:
        token_counts[t] += 1


print("Top tokens:")
for tok, cnt in token_counts.most_common(40):
    print(repr(tok), cnt)


cap_word_re = re.compile(r"^[А-ЯЁA-Z][\w\-\.()\/]*$")

def gen_ngrams(tokens, max_n=3):
    n = len(tokens)
    for L in range(1, max_n + 1):
        for i in range(0, n - L + 1):
            yield i, i + L, tokens[i:i+L]


measurement_tokens = {
    "шт", "шт.", "шт/", "шт/уп", "уп", "уп.", "кг", "г", "г.", "гр", "гр.",
    "мл", "мл.", "л", "л.", "м", "м.", "см", "см.", "п/эт", "п/п", "упак",
}
stop_words = {"и", "с", "по", "на", "для", "от", "в", "поиск", "ср-во"}


ngram_lines = defaultdict(set)  
ngram_preceders = defaultdict(set)
ngram_followers = defaultdict(set)

for idx, toks in enumerate(lines_tokens):
    normed = toks
    n = len(normed)
    for i, j, gram in gen_ngrams(normed, max_n=3):
        
        if any('/' in w for w in gram):
            continue
        if any(w.lower() in measurement_tokens for w in gram):
            continue
        
        if not all(cap_word_re.match(w) and any(is_cyrillic_letter(ch) for ch in w if ch.isalpha()) for w in gram):
            continue

        ngram = tuple(gram)
        ngram_lines[ngram].add(idx)
        
        if i - 1 >= 0:
            ngram_preceders[ngram].add(normed[i-1])
        
        if j < n:
            ngram_followers[ngram].add(normed[j])


cand_info = {}
for gram, line_set in ngram_lines.items():
    count = len(line_set)
    if count < 2:
        continue
    if all(len(w) < 2 for w in gram):
        continue
    gram_s = " ".join(gram)

    end_hits = 0
    start_hits = 0
    for li in line_set:
        toks = lines_tokens[li]
        L = len(gram)
        if L <= len(toks) and toks[-L:] == list(gram):
            end_hits += 1
        if toks[:L] == list(gram):
            start_hits += 1

    end_ratio = end_hits / count
    start_ratio = start_hits / count
    preceders = ngram_preceders.get(gram, set())
    followers = ngram_followers.get(gram, set())
    preceder_div = len(preceders)
    follower_div = len(followers)

    cand_info[gram_s] = {
        "count": count,
        "end_ratio": end_ratio,
        "start_ratio": start_ratio,
        "preceder_div": preceder_div,
        "follower_div": follower_div,
        "len": len(gram),
    }


selected = []
for s, info in cand_info.items():
    c = info["count"]
    er = info["end_ratio"]
    sr = info["start_ratio"]
    pd = info["preceder_div"]
    L = info["len"]

    
    if max(er, sr) < 0.5:
        continue

    accept = False
    if L > 1:
        
        if er >= 0.15 or pd >= 3 or c >= 5:
            accept = True
    else:
        
        w = s
        if w.lower() in stop_words:
            accept = False
        elif max(er, sr) >= 0.5 and pd >= 2:
            accept = True
        elif c >= 50 and max(er, sr) >= 0.1:
            accept = True

    if accept:
        selected.append((s, info))


selected.sort(key=lambda x: (-x[1]["count"], len(x[0])))
accepted = []
accepted_set = set()
for s, info in selected:
    parts = s.split()
    skip = False
    
    for k in (1, 2):
        if len(parts) > k:
            suffix = " ".join(parts[-k:])
            if suffix in accepted_set:
                skip = True
                break
    if skip:
        continue
    accepted.append((s, info))
    accepted_set.add(s)


MIN_COUNT = 200
accepted = [(s, info) for s, info in accepted if info.get("count", 0) >= MIN_COUNT]
final_candidates = [s for s, _ in accepted]

print("\nFiltered supplier candidates (final):")
for s, info in accepted[:200]:
    print(s, info["count"], f"end_ratio={info['end_ratio']:.2f}", f"pred={info['preceder_div']}")


with open("suppliers.txt", "w", encoding="utf-8") as out:
    for s in final_candidates:
        out.write(s + "\n")

print(f"\nWrote {len(final_candidates)} candidates to suppliers.txt")


def load_russian_words(path="russian_words.txt"):
    if not os.path.exists(path):
        return None
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                s.add(w)
    return s

rus_words = load_russian_words()

def phrase_is_russian(phrase: str) -> bool:
    if rus_words is None:
        return False
    for w in phrase.split():
        w2 = re.sub(r"[^а-яёА-ЯЁa-zA-Z-]", "", w).lower()
        if not w2:
            return False
        if w2 not in rus_words:
            return False
    return True

csv_path = "suppliers_with_metrics.csv"
with open(csv_path, "w", encoding="utf-8", newline="") as csvf:
    writer = csv.DictWriter(csvf, fieldnames=["supplier","count","end_ratio","start_ratio"])
    writer.writeheader()
    for s, info in accepted:
        writer.writerow({
            "supplier": s,
            "count": info["count"],
            "end_ratio": f"{info['end_ratio']:.3f}",
            "start_ratio": f"{info['start_ratio']:.3f}",
        })

print(f"Wrote metrics CSV: {csv_path}")
