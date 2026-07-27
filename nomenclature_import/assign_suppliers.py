import re

SUPPLIERS_PATH = "suppliers.txt"
INPUT = "nomenclature.csv"
OUTPUT = "nomenclature_with_supplier.csv"

with open(SUPPLIERS_PATH, "r", encoding="utf-8") as f:
    suppliers = [line.strip() for line in f if line.strip()]

suppliers.sort(key=lambda s: -len(s))

supplier_patterns = [(s, re.compile(re.escape(s), re.IGNORECASE)) for s in suppliers]

out_lines = []
with open(INPUT, "r", encoding="utf-8") as fin:
    for raw in fin:
        line = raw.rstrip("\n")
        best = None
        best_end = -1
        best_len = 0
        for s, pat in supplier_patterns:
            for m in pat.finditer(line):
                start, end = m.span()
                
                if end > best_end or (end == best_end and len(s) > best_len):
                    best = s
                    best_end = end
                    best_len = len(s)
        out_lines.append((line, best if best is not None else "None"))


import csv
with open(OUTPUT, "w", encoding="utf-8", newline="") as outf:
    writer = csv.writer(outf)
    writer.writerow(["product", "supplier"])
    for prod, sup in out_lines:
        writer.writerow([prod, sup])

print(f"Wrote {len(out_lines)} rows to {OUTPUT}")