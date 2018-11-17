import pandas as pd

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
        text_labels = []
        set_labels=set()
        count = 0
        text = data.iloc[:,5]
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
        
        print('\n'+'*'*80 )
        print('Printing Labels')
        print('*'*80 + '\n')
        print(labels[:5])

        return(text.astype(str),labels)
load_csv()