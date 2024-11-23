import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score
data = pd.read_csv('C:\\Users\\pc\\PycharmProjects\\Data Mining\\fromage.txt', delimiter='\s+')
#Affichage des premières lignes pour explorer les données
print(data.head())
print(data.describe())
print(data.info())
# Sélectionner uniquement les colonnes numériques
numeric_data = data.drop('Fromages', axis=1)
print (numeric_data.head())
# Effectuer la CAH
Z = linkage(numeric_data, method='ward', metric='euclidean')
# Afficher le dendrogramme
plt.figure(figsize=(10, 6))
dendrogram(Z, labels=data['Fromages'].values, leaf_rotation=90)
plt.title('Dendrogramme de la CAH des fromages')
plt.xlabel('Fromages')
plt.ylabel('Distance')
plt.show()
t=300
# Découpage du dendrogramme en clusters
clusters = fcluster(Z, t, criterion='distance')
print("Clusters obtenus :", clusters)
# Ajouter les numéros de cluster au dataframe d'origine
data['Cluster'] = clusters
print("Dataframe mis à jour avec les étiquettes de cluster :")
print(data)
num_data = data.drop('Fromages', axis=1)
#Description
# Calculer les statistiques descriptives pour chaque cluster
cluster_stats = num_data.groupby('Cluster').agg(['mean', 'median'])
# Afficher les statistiques descriptives
print(cluster_stats)
# Pistes pour la détection du nombre adéquat de classes(on utilise la methode de la silhouette
# Calculer le score de silhouette pour les clusters
silhouette_avg = silhouette_score(numeric_data, clusters)
# Afficher le score de silhouette
print("Score de silhouette moyen :", silhouette_avg)
# résultat (0.31556786198229153)proche de 0 cela indique que l'observation est près de la frontière entre deux clusters
