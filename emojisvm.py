# required imports:
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

def load_data():
    
    all_files = []
    
    for file in glob.glob("data/*.csv"):     
        df = pd.read_csv(file, on_bad_lines="skip", engine="python")
        df = df.dropna(subset=["Text"])
        lbl = file.split("/")[-1].replace(".csv", "")
        df["label"] = lbl
  
        all_files.append(df[["Text", "label"]])
    return pd.concat(all_files, ignore_index=True)


def main():
    
    df = load_data()

   
    print("Number of classes:", df["label"].nunique())
    print("Total number of samples in dataframe:", len(df))

    
    X = df["Text"].astype(str)
    y = df["label"].astype(str)

    # applying train test split
    X_train, X_values, y_train, y_values = train_test_split(X,y,test_size=0.3,random_state=40,stratify=y )


    vectorize = TfidfVectorizer()
    X_train = vectorize.fit_transform(X_train)
    X_values2 = vectorize.transform(X_values)


    model = LinearSVC()

    model.fit(X_train, y_train)
    predictions = model.predict(X_values2)
    accuracy = accuracy_score(y_values, predictions)
    # using average= "macro" because there are many classes where emojis may vary in balance. 
    # "macro" helps with making our results more meaningful 
    f1_sc = f1_score(y_values, predictions, average= "macro")

    print("SVM Accuracy:", accuracy)
    print("SVM Macro f1 Score", f1_sc)  

main()

