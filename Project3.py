#Isaac Palacio

import os
import sys
import random
import pandas as pd
from csv import reader
from IPython.core import ultratb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

def checkInput(directory1, directory2):
    if not os.path.isfile(directory1) or not os.path.isfile(directory2):
        raise ValueError('Invalid directory.')
        
def getInfo(directory, dia, diaClass, sys, sysCLass, eda, edaClass, res, resClass):
    with open(directory, 'r') as temp_f:
        csv_reader = reader(temp_f)
        for lineList in csv_reader:
            #subjectID = lineList[0]
            dataType = lineList[1]
            painClass = lineList[2]
            data = pd.Series(lineList[3:], dtype=float)
            cleanData = data[data>0]
            if len(cleanData) == 0:
                donwSampleData =  [0] * 5000
            else:
           		donwSampleData = downSample(cleanData)        
            painCLassn = 0; #No pain
            if painClass=="Pain":
                painCLassn = 1  
            #Save Data
            if   dataType=="BP Dia_mmHg":
                dia.append(donwSampleData)
                diaClass.append(painCLassn)
            elif dataType=="LA Systolic BP_mmHg":
                sys.append(donwSampleData)
                sysCLass.append(painCLassn)
            elif dataType=="EDA_microsiemens":
                eda.append(donwSampleData)
                edaClass.append(painCLassn)
            elif dataType=="Respiration Rate_BPM":
                res.append(donwSampleData)
                resClass.append(painCLassn)

def downSample(data):
    ratio = len(data) // 5000
    result = []
    for x in range(0,5000):
        result.append(sum(data[x*ratio:(x*ratio)+ratio])/ratio)  
    series = pd.Series(result, dtype=float)
    resultNormilize = series / series.max()
    return resultNormilize

def voting(DiaPred, SysRFPred, EDARFPred, ResRFPred):
    result = []
    if len(DiaPred) == len(SysRFPred) and len(SysRFPred) == len(EDARFPred) and len(EDARFPred) == len(ResRFPred):
        for i in range(0,len(DiaPred)):
            x = DiaPred[i]
            x = x + SysRFPred[i]
            x = x + EDARFPred[i]
            x = x + ResRFPred[i]
            if x > 2:
                result.append(1)
            elif x < 2:
                result.append(0)
            else:
                result.append(random.choice([1,0]))
    else:
        raise ValueError('The prediction arrays are not the same size.')
    return result

#Make error messages colorful
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)

#Script Input
if len(sys.argv) != 3:
    raise ValueError('The command must follow the following structure: Project3.py <Training_Data> <Testing_Data>')
#Get Input
trainingDirectory = sys.argv[1]
testingDirectory = sys.argv[2]

##Manual Input
#trainingDirectory = "./data2.csv"
#testingDirectory = "./data1.csv"
checkInput(trainingDirectory, testingDirectory)

diaClassificationTraining=[]
diastolicBPTraining=[]
sysClassificationTraining=[]
systolicBPTraining=[]
edaClassificationTraining=[]
EDATraining=[]
resClassificationTraining=[]
respirationTraining=[]

print('Getting training data.')
getInfo(trainingDirectory, diastolicBPTraining, diaClassificationTraining, 
        systolicBPTraining, sysClassificationTraining, EDATraining, 
        edaClassificationTraining, respirationTraining, resClassificationTraining)
            
#Do the same with testingDirectory

diaClassificationTesting=[]
diastolicBPTesting=[]
sysClassificationTesting=[]
systolicBPTesting=[]
edaClassificationTesting=[]
EDATesting=[]
resClassificationTesting=[]
respirationTesting=[]

print('Getting validation data.')
getInfo(testingDirectory, diastolicBPTesting, diaClassificationTesting, 
        systolicBPTesting, sysClassificationTesting, EDATesting, 
        edaClassificationTesting, respirationTesting, resClassificationTesting)

#Performe score level fusion:
DiaRF = RandomForestClassifier()
SysRF = RandomForestClassifier()
EDARF = RandomForestClassifier()
ResRF = RandomForestClassifier()

DiaRF.fit(diastolicBPTraining, diaClassificationTraining)
DiaPred = DiaRF.predict(diastolicBPTesting)

SysRF.fit(systolicBPTraining, sysClassificationTraining)
SysRFPred = SysRF.predict(systolicBPTesting)

EDARF.fit(EDATraining, edaClassificationTraining)
EDARFPred = EDARF.predict(EDATesting)

ResRF.fit(respirationTraining, resClassificationTraining)
ResRFPred = ResRF.predict(respirationTesting)

MyResults = voting(DiaPred, SysRFPred, EDARFPred, ResRFPred)

#Gather results
recall = recall_score(diaClassificationTesting,MyResults)
acc_score = accuracy_score(diaClassificationTesting, MyResults)
precision = precision_score(diaClassificationTesting, MyResults)
confusion_matrices = confusion_matrix(diaClassificationTesting, MyResults)

print("\n Results \n")
#print('Recall of each fold: {}'.format(recall))
#print('Recall : {}'.format(recall))
#print('Accuracy of each fold: {}'.format(acc_score))
print('Accuracy : {}'.format(acc_score))
#print('Precision of each fold: {}'.format(precision))
#print('Precision : {}'.format(precision))
#print('Confusion Matrix of each fold:')
#print('Confusion matrix:')
#print(confusion_matrices)
#print("\n")
