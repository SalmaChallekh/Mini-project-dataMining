import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('C:\\Users\\pc\\PycharmProjects\\Data Mining\\fromage.txt', delimiter='\s+')
#Affichage des premières lignes pour explorer les données
print(data.head())
print(data.describe())
print(data.info())
# Sélectionner uniquement les colonnes numériques
numeric_data = data.drop('Fromages', axis=1)
print (numeric_data.head())
# Normaliser les données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
print("Données normalisées :")
print(scaled_data)
k = 4  # Exemple : nombre de clusters souhaité
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(scaled_data)

# Obtenir les étiquettes de cluster pour chaque observation
cluster_labels = kmeans.labels_
print("Étiquettes de cluster pour chaque observation :")
print(cluster_labels)

# Ajouter les étiquettes de cluster au dataframe d'origine
data['Cluster'] = cluster_labels
print("Dataframe mis à jour avec les étiquettes de cluster :")
print(data)
num_data = data.drop('Fromages', axis=1)
print(num_data)
#Description
# Calculer les statistiques descriptives pour chaque groupe
grouped_stats = num_data.groupby('Cluster').agg(['mean', 'std', 'median'])

# Afficher les statistiques descriptives
print(grouped_stats)

# Pistes pour la détection du nombre adéquat de classes(on utilise la methode de la silhouette
# Liste pour stocker les valeurs de score de silhouette
silhouette_scores = []

# Calculer le score de silhouette pour différents nombres de clusters
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    silhouette_scores.append(silhouette_avg)
# Trouver le nombre optimal de clusters qui maximise le coefficient de silhouette moyen
optimal_clusters = np.argmax(silhouette_scores) + 2  # Ajouter 2 car nous avons commencé à partir de 2 clusters
# Afficher le nombre optimal de clusters
print("Nombre optimal de clusters selon le coefficient de silhouette :", optimal_clusters)
# Tracer le graphique du score de silhouette en fonction du nombre de clusters
plt.plot(range(2, 10), silhouette_scores, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Score de silhouette moyen')
plt.title('Méthode de la silhouette')
plt.show()


