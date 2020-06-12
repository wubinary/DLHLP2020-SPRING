def write_in(from_file, to_file, label):
    for line in from_file.readlines():
        to_file.write("__label__"+label+" "+line)
to_file = open("data_train_mix.txt", "w") 
imdb_pos = open("mix/train.pos","r")
imdb_neg = open("mix/train.neg","r")
yelp_pos = open("mix/train.pos","r")
yelp_neg = open("mix/train.neg","r")

write_in(imdb_pos, to_file, "pos")
write_in(imdb_neg, to_file, "neg")
write_in(yelp_pos, to_file, "pos")
write_in(yelp_neg, to_file, "neg")

    
