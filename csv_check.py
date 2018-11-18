import csv
import sys
import pandas as pd
csv.field_size_limit(sys.maxsize)
def load_csv():
    with open('fake1.csv') as file:
        data = pd.read_csv(file)
        
        print('\n'+'*'*80 )
        print('Printing Text Data')
        print('*'*80 + '\n')
        working_data = data['text'].head()
        print(working_data)
        print('*'*80)
        text = []
        text_text = []
        text_labels = []
        set_labels=set()
        count = 0
        text_text = data.iloc[:,5]
        print(text_text.astype)
        text_labels = data.iloc[:,19]

        for row in text_labels:
            set_labels.add(row)
        
        print('\n'+'*'*80 )
        print('Printing Distinct Labels')
        print('*'*80 + '\n')
        print(set_labels)

        labels=[]
        for i,row in enumerate(text_labels):
            if(row =='conspiracy'):
                text_labels=text_labels.replace(row,'1')
                
            elif(row =='fake'):
                text_labels =text_labels.replace(row,'1')
                
            elif(row =='bs'):
                if(float(data.loc[i][12])>0):
                    text_labels =text_labels.replace(row,'1')
                else:
                    text_labels =text_labels.replace(row,'0')
            elif(row =='bias'):
                if(float(data.loc[i][12])>0):
                    text_labels =text_labels.replace(row,'1')
                else:
                    text_labels =text_labels.replace(row,'0')     
            elif(row =='hate'):
                text_labels =text_labels.replace(row,'1')
                
            else:
                text_labels =text_labels.replace(row,'0')
        
        return(text_text.astype(str),text_labels)
load_csv()