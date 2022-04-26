'''
    Auteur : Eliott GUILLOSSOU
    
    Descriptif : Ce programme va definir les étiquettes ( ici A ou B ) d'un fichier 'data_test' à partir d'un fichier dont nous connaissons les étiquettes.
                 Nous y arriverons grace a notre fonction knn et notre vérification se fera visuellement en representant les points de 'data_train' dans un graphique
                 
'''
    
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
from math import *



'''Fonction Algo KNN '''
def knn(k,inconnu):
    '''Création des variables et ouverture du fichier 'data_train' '''
    lst_distance_donnee = []
    nb_A=nb_B = 0
    dTrain_knn = csv.DictReader(open('data_train.csv', 'r'),delimiter=';')
    ''' Boucle qui nous permet d'obtenir la liste de donnee trié dans l'ordre croissant en fonction de la distance'''
    for row in dTrain_knn:
        line = dict(row)
        calcul_dist = sqrt((float(line['x'])-float(inconnu['x']))**2+(float(line['y'])-float(inconnu['y']))**2)
        distance_donnee = {"distance" : calcul_dist , "donnee" : line['e']}
        lst_distance_donnee.append(distance_donnee)
    lst_distance_donnee_tri = sorted(lst_distance_donnee, key=lambda k: k['distance'])
    ''' Nous avons ici une classification et nous retournons donc l'étiquette majoritaire en regardant les k premiers éléments de notre liste 'lst_distance_donnee_tri' '''
    for i in range(k):
        donnee_ligne = lst_distance_donnee_tri[i]['donnee']
        if donnee_ligne == 'A' :
            nb_A += 1
        else :
            nb_B += 1
    if nb_B>nb_A :
        val_renv = 'B'
    else :
        val_renv = 'A'
    return val_renv


'''Création des variables pour le programme principal '''
columns = defaultdict(list)
'''Base d'apprentissage transformée en liste'''
dTrain = csv.DictReader(open('data_train.csv', 'r'),delimiter=';')

'''Transformation des coordonnées en valeurs '''
for row in dTrain: 
        for (k,v) in row.items():
            if(k=='x' or k=='y'):
                v=float(v)    
            columns[k].append(v) 

'''Nous considèrons  qu'il n'y a pas de cellule 'vide' dans le csv '''
'''Nous changeons les etiquettes pour avoir une couleur pour les points dont leur étiquette est A et une autre couleur pour B'''
for index, item in enumerate(columns['e']):
    if item=='A':
        columns['e'][index] = 'r'
    else:
         columns['e'][index] = 'b'
         

        
'''Affichage du graphique '''
plt.title( 'Graphique ')
plt.axis([0,10,0,10])
plt.scatter(columns['x'],columns['y'],c=columns['e'])
plt.xlabel("Abscisses")
plt.ylabel("Ordonnées")
plt.show()


  
''' Nous prenons les 4 premiers éléments de notre liste de distance triée '''        
k=4

''' Affichage plus clair des résultats de notre fonction knn éxecutée sur notre 'data_test' et ouverture de 'data_test' grâce  au lecteur  ''' 
dtest = csv.DictReader(open('data_test.csv', 'r'),delimiter=';')
for row in dtest:
    line = dict(row)
    print(knn(k,line),' -> (',line['x'],',',line['y'],')')
    
    
