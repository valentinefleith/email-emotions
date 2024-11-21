#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:41:54 2024

@author: pauline
"""
import stanza
import csv


path = "../good.csv"


def read_vader(vader_path: str) -> dict[str, float]:

    vader = {}
    
    with open(vader_path, "r", encoding="utf8") as file:
        
        token_ligne = file.readlines()
        
        for ligne in token_ligne :
            info_token = ligne.split("->")
            vader[str(info_token[0])] = float(info_token[1])

    return vader


def anonymisation(mail_annote)->str:
    doc_ner = mail_annote.ents
    mail_anon = mail_annote.text
    for token in doc_ner:
        if token.type == "PER":
            mail_anon = mail_anon[0:token.start_char] + "Laezel" + mail_anon[token.end_char+1:]
            
    return mail_anon


def get_features(mail_annote, vader)->dict:
    
    features = dict.fromkeys(["NOUN", "VERB", "ADV", "ADJ",
                              "CCONJ", "EX_M", "Q_M", "POS", "NEG"], 0)
    
    # features = {"NOUN" : 0,
    #          "VERB" : 0,
    #          "ADV" : 0,
    #          "ADJ" : 0,
    #          "CCONJ" : 0,
    #          "EX_M" : 0,
    #          "Q_M" : 0,
    #          "POS": 0,
    #          "NEG": 0
    #          }
        
    for sentence in mail_annote.sentences:
        for word in sentence.words :
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
                
            


def get_data(path:str, nlp)->list[list]:
    
    dic = {
    "neutre" : 1,
    "joie" : 2,
    "tristesse" : 3,
    "colère" : 4,
    "surprise_pos" : 5,
    "surprise_neg" : 6
    }
    
    data = []
    
    with open(path, newline='') as csvfile:

        vader = read_vader("../vader_lexicon.txt")
        reader = csv.reader(csvfile, delimiter='|')
        next(reader)

        for row in reader:
            for k, item in dic.items():
                mail_raw = [ x.strip() for x in row[dic[k]].split("***") if x != "" ]
                mails_annotes = [ nlp(x) for x in mail_raw]
                for mail_annote in mails_annotes :
                    txt_anon = anonymisation(mail_annote)
                    txt_features = get_features(mail_annote, vader)
                    txt_features["LABEL"] = k
                    txt_features["TEXT"] = txt_anon
                    
                    data.append(txt_features)

            
    return data

def write_csv(data:list):

    file = "../data_featurized.csv"
    with open(file, mode='w', newline='', encoding='utf-8') as fichier_csv:
        writer = csv.DictWriter(fichier_csv, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    
    return print("le csv a été créé au chemin ../data_featurized.csv")

def main(path):
    nlp = stanza.Pipeline(lang='fr', processors='tokenize,mwt,pos,lemma,ner')
    data = get_data(path, nlp)
    write_csv(data)
    
    return print("done")
    
    
main(path)