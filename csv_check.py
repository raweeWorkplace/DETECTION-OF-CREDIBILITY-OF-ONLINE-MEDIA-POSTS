import sys
import csv
import pandas as pd
csv.field_size_limit(sys.maxsize)

def load_csv():
    with open('fake1.csv') as file:
        data = pd.read_csv(file)
          
        print(data.iloc[:,19])
        #print(data.iloc[:,19])
        # print(data.iloc[:1,7:9])
        # print(data.iloc[:1,10:12])
        # print(data.iloc[:1,13:15])
        # print(data.iloc[:1,16:18])
        # print(data.iloc[:1,19:20])
        text = []
        labels=[]
        count = 0
        text = data.iloc[:,5]
        labels = data.iloc[:,12]
        # for row in data:
        #     text.append("\n author:"+ data.iloc[:,2])
        return(text.astype(str),labels)
load_csv()
# import csv
# with open('text/Extra/fake1.csv','r') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for i,row in enumerate(reader):
#         print(row["author"])
#         if(i >= 9):
#             break
