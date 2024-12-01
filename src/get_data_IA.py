#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:32:35 2024

@author: pauline
"""

import stanza
import csv


path = "../generated_emails.csv"


def load_stopwords(path: str) -> list:
    """
    Loads stopwords from a file.
    
    Parameters
    ----------
    path : str
    Path to the file containing stopwords, with one stopword per line.

    Returns
    -------
    list
    A list of stopwords.
    """
    with open(path, "r") as inf:
        return inf.read().splitlines()


def read_vader(vader_path: str) -> dict[str, float]:

    vader = {}
    
    with open(vader_path, "r", encoding="utf8") as file:
        
        token_ligne = file.readlines()
        
        for ligne in token_ligne :
            info_token = ligne.split("->")
            vader[str(info_token[0])] = float(info_token[1])

    return vader



def get_features(list_token_annote, vader)->dict:
    
    features = dict.fromkeys(["NOUN", "VERB", "ADV", "ADJ",
                              "CCONJ", "EX_M", "Q_M", "POS", "NEG"], 0)
    
    for word in list_token_annote:
        # Comptage des POS non verbales
        if word.upos in features.keys() and word.upos != "VERB":
            features[word.upos]+=1
                
        # Comptage des verbes conjugés
        if word.upos == "VERB":
            if word.feats != "VerbForm=Inf":
                features["VERB"]+=1
            
        # Comptage des points d'exclamation et interrogation
        if word.upos == "PUNCT":
            if word.text == "?":
                features["Q_M"]+=1
            if word.text == "!":
                features["EX_M"]+=1
                    
        # Comptage des mots ayant une polarité
        score = vader.get(str(word.lemma),0)
        if score > 0 :
            features["POS"]+=1
        if score < 0 :
            features["NEG"]+=1
    
    # Ajout des booleens
    for pos in ["NOUN", "VERB", "ADV", "ADJ", "CCONJ"]:
        if features[pos] == 0:
            features[f"BOOL_{pos}"] = 0
        else :
            features[f"BOOL_{pos}"] = 1
    return features
                
                

def get_data(path:str, nlp, stop_words)->list[list]:
    
    dic = {
    "neutre" : 0,
    "joie" : 1,
    "tristesse" : 2,
    "colère" : 3,
    "surprise_pos" : 4,
    "surprise_neg" : 5
    }
    
    data = []
    
    with open(path, newline='') as csvfile:

        vader = read_vader("../ressources/vader_lexicon.txt")
        reader = csv.reader(csvfile, delimiter='|')
        next(reader)

        for row in reader:
            for k, item in dic.items():
                 mail_raw = [ x.strip() for x in row[dic[k]].split("***") if x != "" ]
                 mails_annotes = [ nlp(x) for x in mail_raw]
                 mails_annote_sw = [
                     [
                         token
                         for sentence in mail_annote.sentences
                         for token in sentence.words
                         if token.text.lower() not in stop_words
                         ]
                     for mail_annote in mails_annotes
                     ]
                 for mail_annote in mails_annote_sw :
                     txt_features = get_features(mail_annote, vader)
                     txt_features["LABEL"] = k
                     txt_features["TEXT"] = " ".join([x.text for x in mail_annote])
                     
                     data.append(txt_features)


            
    return data


def write_csv(data:list):

    file = "../dataIA_featurized.csv"
    with open(file, mode='w', newline='', encoding='utf-8') as fichier_csv:
        writer = csv.DictWriter(fichier_csv, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    
    return print("le csv a été créé au chemin ../dataIA_featurized.csv")


def main(path):
    nlp = stanza.Pipeline(lang='fr', processors='tokenize, mwt, pos,lemma')
    stop_words = load_stopwords("../ressources/fr-stopwords.txt")
    data = get_data(path, nlp, stop_words)
    write_csv(data)
    
    return print("done")
    
    
main(path)