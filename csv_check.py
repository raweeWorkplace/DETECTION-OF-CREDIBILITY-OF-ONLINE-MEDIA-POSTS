import sys
import csv
import pandas as pd
csv.field_size_limit(sys.maxsize) #sys.maxsize is used to restrict the output limit

def load_csv():
    with open('fake1.csv') as file:
        data = pd.read_csv(file)
        
        print('Getting the Columns of data Set')
        print(data.loc[1:1])
        print(data.head())
        text = []
        text_labels = []
        set_labels=set()
        count = 0
        text = data.iloc[:,5]
        text_labels = data.iloc[:,19]

        for row in text_labels:
            set_labels.add(row)
        print(set_labels)

        labels=[]
        for i,row in enumerate(text_labels):
            if(row =='conspiracy'):
                labels.append('1')
                
            elif(row =='fake'):
                labels.append('1')
                
            elif(row in['bs','bias']):
                if(float(data.loc[i][12])>0):
                    labels.append('1')
                else:
                    labels.append('0')
                
            elif(row =='hate'):
                labels.append('1')
                
            else:
                labels.append('0')
        # print(labels)

        return(text.astype(str),labels)
load_csv()
# import csv
# with open('text/Extra/fake1.csv','r') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for i,row in enumerate(reader):
#         print(row["author"])
#         if(i >= 9):
#             break
