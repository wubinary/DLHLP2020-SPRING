import json
import sys

infile = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w") as f:
    f.write(f"ID,Answer\n")
    for key, value in infile.items():
        f.write(f'{key},{value.replace(",","")}\n')