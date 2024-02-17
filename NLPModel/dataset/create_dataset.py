import pandas as pd
import numpy as np

patterns = ["What herbal should I use with ", "I have ", "Herbal for ", "I need a herbal for ", "Is there a herbal for ", "My need concerns with ", "Is there a remedy for ", "Which natural remedy should I use for "]
cleanHerbs = ["JACKFRUIT", "SAMBONG", "LEMON", "JASMINE", "MANGO", "MINT", "AMPALAYA", "MALUNGGAY", "GUAVA", "LAGUNDI"]
herbs = ["JACKFRUIT\n", "SAMBONG\n", "LEMON\n", "JASMINE\n", "MANGO\n", "MINT\n", "AMPALAYA\n", "MALUNGGAY\n", "GUAVA\n", "LAGUNDI\n"]
theKeywords = 'C:/Users/KHYLE MATTHEW/OneDrive - Mapúa University/Code Projects/SL_NLP/dataset/keywords.txt'

tempHead = ""

uniqueVals = []
classify = {}
cleanVals = []
placeholder = ""
placeholder1 = ""

#Get Unique Values
with open(theKeywords, 'r') as file:
    for line in file:
        if line not in herbs:
            if line not in uniqueVals:
                uniqueVals.append(line)

#Remove next lines
for item in uniqueVals:
    placeholder = item.replace("\n", "")
    cleanVals.append(placeholder)

#Create Nested Dictionaries with Default Values for Classification Labels
for val in cleanVals:
    classify[val] = {}
    for herbz in cleanHerbs:
        classify[val][herbz] = 0

#Get Classification Labels
with open(theKeywords, 'r') as file:
    for line in file:
        if line in herbs:
            tempHead = line.replace("\n", "")
        else:
            placeholder1 = line.replace("\n", "")
            classify[placeholder1][tempHead] = 1
#Clean Again
uniqueKeys = []
for mumei in cleanVals:
    if mumei not in uniqueKeys:
        uniqueKeys.append(mumei)

#Check if Length of Keys and Dictionary is the same
print(len(classify))
print(len(uniqueKeys))

#Setup Dataframe
headings = ["SYMPTOMS", "JACKFRUIT", "SAMBONG", "LEMON", "JASMINE", "MANGO", "MINT", "AMPALAYA", "MALUNGGAY", "GUAVA", "LAGUNDI"]
df = pd.DataFrame(columns=headings)
herbals = []
for symptom in cleanVals:
    for pattern in patterns:
        a = pattern + symptom
        for herby in cleanHerbs:
            herbals.append(classify[symptom][herby])
        df.loc[len(df.index)] = ([a, herbals[0], herbals[1], herbals[2], herbals[3], herbals[4], herbals[5], herbals[6], herbals[7], herbals[8], herbals[9]])
        herbals.clear()

#Shuffle to avoid pattern order
df = df.sample(frac=1).reset_index(drop=True)

print(df)

#Save Dataset
df.to_csv('dataset.csv')

# #Split the Dataset for Training and Testing
# df['split'] = np.random.randn(df.shape[0], 1)

# msk = np.random.rand(len(df)) <= 0.7

# train = df[msk].reset_index(drop=True).drop('split', axis=1)
# test = df[~msk].reset_index(drop=True).drop('split',axis=1)

# print(train)
# print(test)

# #Save the Datasets
# train.to_csv('train.csv')
# test.to_csv('test.csv')
