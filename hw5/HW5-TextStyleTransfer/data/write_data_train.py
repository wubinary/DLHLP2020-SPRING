def write_in(from_file, to_file, label):
    for line in from_file.readlines():
        to_file.write("__label__"+label+" "+line)
to_file = open("data_train_gender.txt", "w") 
gender_pos = open("gender_data/train.pos","r")
gender_neg = open("gender_data/train.neg","r")

write_in(gender_pos, to_file, "pos")
write_in(gender_neg, to_file, "neg")

    
