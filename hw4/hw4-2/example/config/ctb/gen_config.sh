for i in {1..12};
do
  cp rank_pad_ctb-BERTbase1.yaml rank_pad_ctb-BERTbase$i.yaml;
  sed -i "s/model_layer\:\ 1/model_layer\:\ $i/g" rank_pad_ctb-BERTbase$i.yaml; 
done
