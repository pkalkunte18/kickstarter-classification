# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 08:35:33 2022

@author: saipr
"""

import os #reading directories
import pandas as pd #turning things to datasets

#find all the files we need to pull from
folder = "C:\\Users\\saipr\\Desktop\\Programming\\Kickstarter Classification\\raws"
files = os.listdir(folder)
#print(files) #sweet, all our folders are here
#turn those into the actual files
files = [folder+"\\"+files[i] for i in range(0, len(files))]
keepThese = ['backers_count', 'category', 'country', 'goal',
              'is_starrable', 'spotlight', 'staff_pick',
              'state'] #columns we want (7 indep, 1 dep)

#for each folder in files
toConcat = []
for f in files:
    theseCSVs = os.listdir(f) #find the csvs in this folder
    theseCSVs = [f+"\\"+theseCSVs[i] for i in range(0, len(theseCSVs))] #turn them into full directories
    for c in theseCSVs: #for each csv file
        d = pd.read_csv(c) #read it in
        d = d[keepThese] #get rid of columns we don't want
        toConcat.append(d) #add it to our dataframe list

#now with our list, concat them all
data = pd.concat(toConcat, ignore_index=True) #reindex them when concat-ed
#print(data.head()) #check
#print(data.tail()) #check
#print(data.columns) #check
#print(len(data)) #check

#convert string dictionary to dictionary
import json #for easy conversion
cata = data.category #pull out the dictionaries in question
cata = [json.loads(c) for c in cata] #turn each string into a little dictionary
cat = [c['name'] for c in cata] #get all the subcategories
data['subcats'] = cat #we now have subcategories
print(data.subcats)#make sure it's right
#and finally drop category
data.drop('category', axis = 1, inplace = True)

#export the cleaned data for future use
data.to_csv('C:\\Users\\saipr\\Desktop\\Programming\\Kickstarter Classification\\cleaned.csv')   