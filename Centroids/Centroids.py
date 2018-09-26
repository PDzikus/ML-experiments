# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

class Centroids:

    centroids: pd.DataFrame

    def __init__(self):
        self.epsilon = 1.5
        self.alpha = 0.70
        self.beta = 0.5

    def initialize_random(self, dataset):
        column = dataset.shape[1] - 1
        ds_min = dataset.groupby([column]).min()
        ds_max = dataset.groupby([column]).max()
        random_seed = np.random.random(ds_min.shape)
        final = (ds_max - ds_min) * random_seed + ds_min
        del final.index.name
        
        return final

    def calculate_distances(self, centroids, dataset):
        dataset = dataset.drop(dataset.columns[-1], axis=1)
        distances = pd.DataFrame(columns = ["Closest", "Lower", "Upper"], index=dataset.index)
        col = 0
        for idx in centroids.index:
            row = centroids.loc[idx]
            dist = np.sqrt(np.sum((dataset - row).T**2))
            distances.insert(loc=col, column = idx, value=dist)
            col += 1
        
        distances["Lower"] = distances.iloc[:,:-3].min(axis=1)
        distances["Upper"] = distances["Lower"] * 1.5
        distances["Closest"] = distances.iloc[:,:-3].idxmin(axis=1)
        
        rough = pd.DataFrame( index = dataset.index)
        for column in centroids.index:
            rough[column + "Lower"] = distances[column] <= distances["Lower"]
            rough[column + "Upper"] = distances[column] <= distances["Upper"]
        rough = rough * 1
        for column in centroids.index:
            rough[column + "Upper"] = rough[column + "Upper"]  - rough[column + "Lower"]

        return distances, rough
        
    def calculate_centroids(self, dataset, classes, rough_sets):
        ds = dataset.drop(dataset.columns[-1], axis=1)
        cd = pd.DataFrame(columns=dataset.columns[:-1], index=classes)
        print (np.sum(rough_sets))
        for column in classes:
            means_lower = ((ds.T * rough_sets[column + "Lower"]).T).sum()
            if rough_sets[column + "Lower"].sum() > 0:
                means_lower /= rough_sets[column + "Lower"].sum()
            means_upper = ((ds.T * rough_sets[column + "Upper"]).T).sum()
            if rough_sets[column + "Upper"].sum() > 0:
                means_upper /= rough_sets[column + "Upper"].sum()
            
            if rough_sets[column + "Upper"].sum() == 0:
                cd.loc[column, :] = means_lower
            elif rough_sets[column + "Lower"].sum() == 0:
                cd.loc[column, :] = means_upper
            else:
                cd.loc[column, :] = means_lower * self.alpha + means_upper * (1 - self.alpha)

        return cd

    def quality(self, distances):
        i = 0
        s = 0
        group_count = 0
        for name, group in distances.groupby("Closest"):
            group_count += 1

            group = group.drop("Closest", axis=1)

            means = group.mean()
            i += means.at[name]
            s += (means.sum() - means.at[name]) / (len(means.index) - 1)

        i /= group_count
        s /= group_count
        return i / s if s != 0 else 0

    def fit(self, ds):
        dataset = ds.copy()
        # nazwy kategorii        
        classes = dataset.loc[:, dataset.columns[-1]].astype("category").cat.categories
        # losowe pierwsze centroidy
        centroids = self.initialize_random(dataset)
        old_centroids = centroids
        distances, rough = self.calculate_distances(old_centroids, dataset)
        old_quality = 0
        quality = 0
        iteracja = 1
        
        while True:
            print()
            print("Iteracja {}".format(iteracja))
            iteracja += 1
            # czyszcze dane z ostatniej kolumny
            dataset.loc[:, dataset.columns[-1]] = distances.loc[:, "Closest"]
            old_centroids = centroids
            centroids = self.calculate_centroids(dataset, classes, rough)
            distances, rough = self.calculate_distances(centroids, dataset)
            old_quality, quality = quality, self.quality(distances)
            
            print(centroids)
            if quality <= old_quality:
                break

        self.centroids = old_centroids
        
    def print(self):
        print(self.centroids)

    def calculate_distance(self, centroids, row):
        distances = pd.Series(index=centroids.index)

        for centroid_index, centroid_row in centroids.iterrows():
            centroid_row = centroid_row - row
            distance = centroid_row.apply(lambda x: x ** 2).sum() ** 1/2
            distances.at[centroid_index] = distance

        return distances
    
    def predict(self, array):
        distances = self.calculate_distance(self.centroids, pd.Series(array))
        return distances.where(lambda x: x == distances.min()).dropna().index[0]

    def accuracy(self, dataset):
        wrong = 0
        for index, row in dataset.iterrows():
            out = self.predict(row[:-1])
            if (out != row[4]):
                wrong+=1
        return (dataset.shape[0]-wrong)/1.5

def main():
    iris_ds = pd.read_csv("iris.txt", header=None)
    centroids = Centroids()
    centroids.fit(iris_ds)
    print ()
    print ("Ostateczny wynik - centroidy:")
    centroids.print()
    print("Accuracy {}%".format(centroids.accuracy(iris_ds)))
if __name__ == "__main__":
    main()
