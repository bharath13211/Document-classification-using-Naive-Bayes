# classify.py : Classify text objects into two categories
#
# PLEASE PUT YOUR NAMES AND USER IDs HERE
#
# Based on skeleton code by D. Crandall, March 2021
#

import sys
import string

def load_file(filename):
    objects=[]
    labels=[]
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ',1)
            labels.append(parsed[0] if len(parsed)>0 else "")
            objects.append(parsed[1] if len(parsed)>1 else "")
    
    return {"objects": objects, "labels": labels, "classes": list(set(labels))}

# classifier : Train and apply a bayes net classifier
#
# This function should take a train_data dictionary that has three entries:
#        train_data["objects"] is a list of strings corresponding to documents
#        train_data["labels"] is a list of strings corresponding to ground truth labels for each document
#        train_data["classes"] is the list of possible class names (always two)
#
# and a test_data dictionary that has objects and classes entries in the same format as above. It
# should return a list of the same length as test_data["objects"], where the i-th element of the result
# list is the estimated classlabel for test_data["objects"][i]
#                                                                                                         
# Do not change the return type or parameters of this function!

table = str.maketrans('', '', string.punctuation)

def classifier(train_data, test_data):
    lab = train_data['classes']
    word_dict_W = {}
    word_dict_E = {}
    prob_W = {}
    prob_E = {}
    for i in range(len(train_data['objects'])):
        wr = train_data['objects'][i].split()
        w = [w1.translate(table) for w1 in wr]
        
        for word in w:
            if train_data['labels'][i] == lab[0]:
                if word in word_dict_W.keys():
                    word_dict_W[word] += 1
                else:
                    word_dict_W[word] = 1
            elif train_data['labels'][i] == lab[1]:
                if word in word_dict_E.keys():
                    word_dict_E[word] += 1
                else:
                    word_dict_E[word] = 1
    total_W = sum(word_dict_W.values())
    total_E = sum(word_dict_E.values())
    for i in word_dict_W:
        prob_W[i] = (word_dict_W[i])/(total_W)
    for i in word_dict_E:
        prob_E[i] = (word_dict_E[i])/(total_E)
    count_W = 0
    count_E = 0
    for i in (train_data['labels']):
        if i == lab[0]:
            count_W += 1
        else:
            count_E += 1
        
    prob_Wt = count_W/(count_W+count_E)
    prob_Et = 1-prob_Wt
    
    
    
    out = []
    #prob_test_W = {}
    for i in range(len(test_data['objects'])):
    
        wtr = test_data['objects'][i].split()
        wt = [w1.translate(table) for w1 in wtr]
        a_w=1
        a_e = 1
        alpha = 0.5
        for word in wt:
            if word in word_dict_W:
                #ww = ((word_dict_W[word])+1)/(total_W+len(word_dict_W))
                ww = ((word_dict_W[word])+alpha)/(total_W+(len(word_dict_W)*alpha))
                #a_w *= prob_W[word]
                a_w *= ww
            if word not in word_dict_W:
                #ww = 1/(total_W+len(word_dict_W))
                ww = alpha/(total_W+(len(word_dict_W)*alpha))
                a_w *= ww
            if word in word_dict_E:
                #ee = ((word_dict_E[word])+1)/(total_E+len(word_dict_E))
                ee = ((word_dict_E[word])+alpha)/(total_E+(len(word_dict_E)*alpha))
                #a_e *= prob_E[word]
                a_e *= ee
            if word not in word_dict_E:
                #ee = 1/(total_E+len(word_dict_E))
                ee = alpha/(total_E+(len(word_dict_E)*alpha))
                a_e *= ee
                
        final_w = prob_Wt * a_w
        final_e = prob_Et * a_e
        abc = (final_w+final_e)
        if (final_w/abc) > (final_e/abc):
        #if final_w > final_e:
            out.append(lab[0])
        else:
            out.append(lab[1])
    
    
    # This is just dummy code -- put yours here!
    #return [test_data["classes"][0]] * len(test_data["objects"])
    return out

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Usage: classify.py train_file.txt test_file.txt")

    (_, train_file, test_file) = sys.argv
    # Load in the training and test datasets. The file format is simple: one object
    # per line, the first word one the line is the label.
    train_data = load_file(train_file)
    test_data = load_file(test_file)
    if(train_data["classes"] != test_data["classes"] or len(test_data["classes"]) != 2):
        raise Exception("Number of classes should be 2, and must be the same in test and training data")

    # make a copy of the test data without the correct labels, so the classifier can't cheat!
    test_data_sanitized = {"objects": test_data["objects"], "classes": test_data["classes"]}

    results= classifier(train_data, test_data_sanitized)

    # calculate accuracy
    correct_ct = sum([ (results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"])) ])
    print("Classification accuracy = %5.2f%%" % (100.0 * correct_ct / len(test_data["labels"])))

        
