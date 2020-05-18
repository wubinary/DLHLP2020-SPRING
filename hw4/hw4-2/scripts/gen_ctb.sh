for split in train dev test;
do
  python3 scripts/convert_conll_to_raw.py data/ctb/$split.conllx > data/ctb/$split.txt
  python3 scripts/convert_raw_to_bert.py data/ctb/$split.txt data/ctb/$split.bertbase-layers.hdf5 chinese
done
