import sys
# Usage: python3 format.py <result csv file> <output file name>
f = open(sys.argv[1])
l = [x.strip().split('\t') for x in f.readlines()]
l = sorted(l[1:], key=lambda x: int(x[0]))
t = ["id,answer\n"]
t += [(x[0]+','+x[1]+"\n") for x in l]
fw = open(sys.argv[2], 'w')
fw.writelines(t)

