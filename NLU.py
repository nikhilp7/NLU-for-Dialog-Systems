import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import sys


def extract_id_name(sequence):
    split_class = sequence.split("<class")[1].strip().split()
    
    tags = []
    values = []
    for k in split_class:
        if "instructor" in k:
            break
            return {}
        taglist = k.split("=")
        if len(taglist) == 2:
            value = []
            if taglist[0] not in tags:
                tags.append(taglist[0])
                value.append(taglist[1])
            values.append(" ".join(value))
        elif len(taglist) == 1:
            value.append(taglist[0])
            values[-1] = " ".join(value)
    
    annot = {}
    for t in range(len(tags)):
        if tags[t] == "id":
            annot["id"]  = values[t]
        if tags[t] == "name":
            annot["name"] = values[t]
    return annot


def get_iob(_class):
    token = []
    for k in _class[1].values():
        splits = k.split(" ")
        for j in range(len(splits)):
            if j == 0:
                token.append((splits[j], 'B'))
            else:
                token.append((splits[j], 'I'))
    return token


word_to_number = {}
number_to_word = {}

def get_features(token, iob):
    annot = 'O'
    
    for i in range(len(iob)):
        if token == iob[i][0]:
            annot = iob[i][1]
            break
            
    lower_token = token.lower()
    if lower_token not in word_to_number:
        word_to_number[lower_token] = len(word_to_number) + 1

    value = word_to_number[lower_token]
    number_to_word[value] = token
    
    all_upper = 1 if token.isupper() else 0
    starts_with_capital = 1 if token.istitle() else 0
    length_of_token = len(token)
    consists_only_numbers = 1 if token.isdigit() else 0
    ends_with_capital = 1 if token[-1].isupper() else 0
    less_than_three = 1 if len(token) < 3 else 0
    has_special_characters = 1 if any(not char.isalnum() for char in token) else 0

    return [value, all_upper, starts_with_capital, length_of_token, consists_only_numbers,
            ends_with_capital, less_than_three, has_special_characters, annot]


def accuracy(predictions, actual):
    assert len(predictions) == len(actual)
    
    hits = 0
    for i in range(len(actual)):
        if actual[i] == predictions[i]:
            hits += 1
    return round(hits / len(actual) * 100, 3)


def extract_data(sequences):
    classes = []
    for i in range(len(sequences)):
        ex = extract_id_name(sequences[i])
        if ex != {}:
            classes.append((i, ex))
            
    dataset = []
    for i in range(len(classes)):
        iob = get_iob(classes[i])
        text = sequences[classes[i][0]].split(".")[0].split()
        for i in range(len(text)):
            annot = get_features(text[i], iob)
            dataset.append(annot)
            
    dataset = np.asarray(dataset)
    X = dataset[:, :-1]
    y = dataset[:, -1]
    return X, y


def extract_dataset(filename:str):
    data = []
    with open(filename, "r") as train_file:
        data = train_file.readlines()
        
    sequences = []
    seq = []
    for each_line in data:
        if each_line != '\n':
            seq.append(each_line)
        else:
            text = "".join(seq)
            text = text.replace("\n", " ").strip()
            text = text.replace("(", "")
            text = text.replace(")", "")
            if "<class" not in text:
                continue
            text = text.replace(">", "")
            sequences.append(text)
            seq = []
    
    classes = []
    for i in range(len(sequences)):
        ex = extract_id_name(sequences[i])
        if ex != {}:
            classes.append((i, ex))
            
    dataset = []
    for i in range(len(classes)):
        iob = get_iob(classes[i])
        text = sequences[classes[i][0]].split(".")[0].split()
        for i in range(len(text)):
            annot = get_features(text[i], iob)
            dataset.append(annot)
            
    dataset = np.asarray(dataset)
    X = dataset[:, :-1]
    y = dataset[:, -1]
    return X, y

def get_trained_model(X, y):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

def print_accuracy(model, X, y, title: str):
    predictions = model.predict(X)
    acc = accuracy(predictions, y)
    print(f"{title} accuracy:", acc)


def classify(x):
    p = model.predict(x)
    result = list(map(lambda m, a: (number_to_word[int(m)], a), x[:, 0], p))
    op = ""
    for word, a in result:
        op += f"{word}/{a}" + " "
    return op


if __name__ == "__main__":

    nargv = len(sys.argv)
    if nargv < 3:
        raise Exception("Invalid number of parameters. Try again")

    argv = sys.argv
    train = argv[1]
    test = argv[2]
    
    X, y = extract_dataset(train)
    model = get_trained_model(X, y)
    print_accuracy(model, X, y, "training")

    X_test, y_test = extract_dataset(test)
    print_accuracy(model, X_test, y_test, "testing")
    
    output = ""
    f = open(test, "r")
    records = f.read().split("\n\n")
    for each_record in records:
        sequences = [each_record]
        try:
            X, y = extract_data(sequences)
            output += classify(X) + "\n\n"
        except IndexError:
            continue
            
    output_file = open("output.txt", "w")
    output_file.write(output)
    output_file.close()

    print(">>> The result is stored in output.txt")

