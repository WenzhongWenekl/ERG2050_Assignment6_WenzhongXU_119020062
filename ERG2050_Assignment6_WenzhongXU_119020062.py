#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 2021

@author: Andy
"""

import os
import numpy as np
from collections import OrderedDict
import copy


class HMM(object):
    def __init__(self, train_data, test_data, test_label):
        self.train = train_data
        self.test = test_data
        self.label = test_label
        self.total = None
        self.root = None
        self.tran = None
        self.pb = 0
        self.ps = 0
        self.dict_list = None
        self.whole = None


    def compute_start_prob(self):
        total_str = list(set(''.join(self.train) + ''.join(self.test)))
        total_str.remove(' ')
        self.total = total_str
        root_mat = []
        for i in self.train:
            root_mat.append(i.split(' '))
        self.root = root_mat
        number_B = 0
        number_S = 0
        for i in root_mat:
            if len(i[0]) == 1:
                number_S += 1
            elif len(i[0]) > 1:
                number_B += 1

        prob_B = number_B / (number_S + number_B)
        prob_S = number_S / (number_S + number_B)
        number_B = 0
        number_S = 0
        for i in root_mat:
            if len(i[0]) == 1:
                number_S += 1
            elif len(i[0]) > 1:
                number_B += 1

        prob_B = number_B / (number_S + number_B)
        prob_S = number_S / (number_S + number_B)
        self.pb = prob_B
        self.ps = prob_S



    def compute_transition(self):

        number_BI = 0
        number_BE = 0
        number_II = 0
        number_IE = 0
        number_EB = 0
        number_ES = 0
        number_SB = 0
        number_SS = 0

        for i in self.root:
            for j in i:
                if len(j) == 2:
                    number_BE += 1
                elif len(j) > 2:
                    number_BI += 1

        for i in self.root:
            for j in i:
                if len(j) >= 3:
                    number_IE += 1
                if len(j) > 3:
                    number_II += len(j) - 3

        for i in self.root:
            for j in range(len(i)-1):
                if len(i[j]) > 1 and len(i[j+1]) > 1:
                    number_EB += 1
                elif len(i[j]) > 1 and len(i[j+1]) == 1:
                    number_ES += 1

        for i in self.root:
            for j in range(len(i)-1):
                if len(i[j]) == 1 and len(i[j+1]) > 1:
                    number_SB += 1
                elif len(i[j]) == 1 and len(i[j+1]) == 1:
                    number_SS += 1

        sumB = number_BI + number_BE
        sumI = number_IE + number_II
        sumE = number_ES + number_EB
        sumS = number_SB + number_SS
        tran_mat = np.array([[0,number_BI/sumB, number_BE/sumB, 0],
                    [0,number_II/sumI, number_IE/sumI, 0],
                    [number_EB/sumE, 0, 0, number_ES/sumE],
                    [number_SB/sumS, 0, 0, number_SS/sumS]]).T
        self.tran = tran_mat

    def compute_emission(self):
        SDict = {} # count word
        for i in self.root:
            for word in i:
                if len(word) == 1:
                    if word not in SDict:
                        SDict[word] = 1 
                    elif word in SDict:
                        SDict[word] += 1

        BDict = {} 
        for i in self.root:
            for word in i:
                if len(word) > 1:
                    if word[0] not in BDict:
                        BDict[word[0]] = 1 
                    elif word[0] in BDict:
                        BDict[word[0]] += 1 
                        
        EDict = {} 
        for i in self.root:
            for word in i:
                if len(word) > 1:
                    if word[-1] not in EDict:
                        EDict[word[-1]] = 1 
                    elif word[-1] in EDict:
                        EDict[word[-1]] += 1 
                            
        IDict = {}      
        for i in self.root:
            for word in i:
                if len(word) > 2:
                    for i in range(1,len(word)-1):
                        if word[i] not in IDict:
                            IDict[word[i]] = 1 
                        elif word[i] in IDict:
                            IDict[word[i]] += 1  
        
        def full_dict(full_word, Dict):
            for i in full_word:
                if i not in Dict:
                    Dict[i] = 1  
            Dict= OrderedDict(sorted(Dict.items(), key=lambda d:d[0]))
            return Dict

        SDict = full_dict(self.total, SDict)
        BDict = full_dict(self.total, BDict)
        IDict = full_dict(self.total, IDict)
        EDict = full_dict(self.total, EDict)
        sumd1 = sum(SDict.values())
        sumd2 = sum(BDict.values())
        sumd3 = sum(IDict.values())
        sumd4 = sum(EDict.values())

        def prob(D1,D2,D3,D4):
            d1 = copy.deepcopy(D1)
            d2 = copy.deepcopy(D2)
            d3 = copy.deepcopy(D3)
            d4 = copy.deepcopy(D4)
            for i in D1:
                D1[i] = d1[i] / sumd1
            for i in D2:
                D2[i] = d2[i] / sumd2
            for i in D3:
                D3[i] = d3[i] / sumd3
            for i in D4:
                D4[i] = d4[i] / sumd4
            return D1,D2,D3,D4 

        SDict,BDict,IDict,EDict = prob(SDict,BDict,IDict,EDict)
        SDict = dict(SDict)
        BDict = dict(BDict)
        IDict = dict(IDict)
        EDict = dict(EDict)
        dict_list = [BDict,IDict,EDict,SDict]
        self.dict_list = dict_list       

    def viterbi_decoding(self):
        Label = ['B','I','E','S']
        whole_label = []
        for sentence in self.test:
            try:
                coun = 0
                empty1 = []
                empty2 = []
                sentence = sentence.strip()
                empty1.append([self.pb*self.dict_list[0][sentence[0]],0,0,self.ps*self.dict_list[3][sentence[0]]])
                for word in sentence[1:]:
                    if word == ' ':
                        continue
                    sub_empty1 = []
                    sub_empty2 = []
                    for i in range(4):
                        arr = self.tran[i] * self.dict_list[i][word]
                        arr = arr * empty1[coun]
                        arr = arr * 100
                        sub_empty1.append(np.max(arr))
                        sub_empty2.append(arr.tolist().index(np.max(arr)))
                    empty1.append(sub_empty1)
                    empty2.append(sub_empty2)
                    coun = coun + 1
                
                final_temp = empty1[-1].index(max(empty1[-1]))
                empty2_reverse = empty2[::-1]
                label_list = []
                label_list.append(final_temp)
                for i in empty2_reverse:
                    label_list.append(i[label_list[-1]])
                
                label_list = label_list[::-1]
                whole_label.append(label_list)
            except:
                pass
        self.whole = whole_label

    def word_segmentation(self):
        pred = []
        for i in range(len(self.whole)):
            sentense = self.test[i]
            order = self.whole[i] 
            new_sentense = ""
            begin =0
            for j in range(1,len(order)):
                if order[j] == 0 or order[j] == 3:
                    new_sentense += sentense[begin:j] + " "   
                    begin = j
            new_sentense = new_sentense + sentense[begin:]+'\n'
            pred.append(new_sentense.strip()) 
            with open('my_prediction.txt','a+') as f:
                f.write(new_sentense)

def main():
    train_data = open(os.path.join('data', 'train.txt')).read().splitlines()
    test_data = open(os.path.join('data', 'test.txt')).read().splitlines()
    test_label = open(os.path.join('data', 'test_gold.txt')).read().splitlines()
    hmm = HMM(train_data, test_data, test_label)
    hmm.compute_start_prob()
    hmm.compute_transition()
    hmm.compute_emission()
    hmm.viterbi_decoding()
    hmm.word_segmentation()
    fout = open('my_prediction.txt', 'r', encoding='utf-8')
    # with open('my_prediction.txt', 'r', encoding='utf-8') as f:
    my_pred = fout.read()
        # f.close()
    test_label = open(os.path.join('data', './test_gold.txt'), encoding='utf-8').read().splitlines()
    count_index = 0
    pred_count = 0
    total_count = 0
    my_pred = my_pred.split("\n")
    test_gold = test_label
    for num in range(len(my_pred)):
        sentence_divide = my_pred[num].split(" ")
        sentence_divide_perfect = test_gold[num].split(" ")
        pos = 0
        pred_pos = []
        good_pos = []
        for word in sentence_divide:
            pos_prev = pos
            if len(word) == 1 or len(word)==0:
                pos_latter = pos
            else:
                pos_latter = pos + len(word)-1
            pos = pos_latter + 1
            pred_pos.append((pos_prev,pos_latter))
        pos = 0
        for word in sentence_divide_perfect:
            pos_prev = pos
            if len(word) == 1:
                pos_latter = pos
            else:
                pos_latter = pos + len(word)-1
            pos = pos_latter + 1
            good_pos.append((pos_prev,pos_latter))
        for index in pred_pos:
            if index in good_pos:
                count_index += 1
        pred_count += len(pred_pos)
        total_count += len(good_pos)
    precision = count_index / pred_count
    recall = count_index / total_count
    f1 = (2*precision*recall) / (precision + recall)
    print("The precision is: %.2f%%, The recall is: %.2f%%, the f1 score is: %.5f."%(precision*100, recall*100, f1))

main()