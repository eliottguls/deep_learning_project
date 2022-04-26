print("===========")    
print("Data test :")
print("===========")

dtest = csv.DictReader(open('data_test.csv', 'r'),delimiter=';')

for row in dtest:
    line = dict(row)
    print(line['e'],line['x'],':',line['y'])
    print(line)
    
    #(awk -F "\"*;\"*" '{print $1}' data_train.csv)  < (awk -F "\"*;\"*" '{print $1}' data_test.csv)


#distance_donnée =  sqrt((line['x']-line['x'])**2+(line['y']-line['y'])**2) if line['x'] and line['y'] > 0 else sqrt((str(line['x'])-str(line['x']))**2+(str(line['y'])- str(line['y']))**2) ]

#Question 2
k=30       
        
p1 = {"x": "0", "y": "0"}
p2 = {"x" : "-2" , "y" : "1"}

print(knn(k,p1))
print(knn(k,p2))