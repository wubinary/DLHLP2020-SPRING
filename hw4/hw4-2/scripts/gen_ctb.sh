for split in train dev test;
do
  python3 scripts/convert_conll_to_raw.py data/ctb/$split.conllx > /media/D/DLHLP/hw4/hw4-2/data/ctb/$split.txt
  python3 scripts/convert_raw_to_bert.py /media/D/DLHLP/hw4/hw4-2/data/ctb/$split.txt /media/D/DLHLP/hw4/hw4-2/data/ctb/$split.bertbase-layers.hdf5 chinese
done
