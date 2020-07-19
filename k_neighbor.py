"""
Daniel Maidment

Tue Jun  4 11:45:29 2019
"""
"""##################################################################"""
import spyder_utilities as su
import numpy as np
import matplotlib.pyplot as plt
from numpy import abs
from numpy import max
from numpy.linalg import norm
"""##################################################################"""


class kNearestNeighbor:
    """
    This class classifies data by the k-nearest neighbor algorithm.
    Args:
        k [int]:
            The number of neighbors to select vote from (default 3).
        measure [str]:
            The distance method used. Select from:
            "euclidean" or "manhatten" (default "euclidean").
        rows [int]:
            The number of rows in the array generated for visualisation 
            (default 10).
        cols [int]:
            The number of columns in the array generated for
            visualisation (default 10).
    """
    
    def __init__(self, k=3, measure="euclidean", shape=(10, 10)):
        if(k>0):
            self.k = int(k)
        else:
            print("k must be a positive integer greater than 1")
            self.k=3
        
        self.order = 2
        if(measure=="euclidean"):
            self.order = 2
        elif(measure=="manhatten"):
             self.order = 1
        else:
            print("measure must be either 'euclidean' or 'manhatten'")
            print("defaulting to euclidean")
        self.M = abs(int(shape[0]))
        self.N = abs(int(shape[1]))
        self.kmap = np.empty((self.M, self.N), dtype = float)
        
    def train(self, tdata=None, debug=False):
        """
        Train a map of size MxN on data passed. Using the k-nearest
        neighbor algorithm. The map is traveresed and for each point on
        the map, a list of the relative distances is passed to an array
        which is then sorted and passed to the _vote() method which
        decides the k-nearest neighbor, and assisgns the label to that
        point.
        
        Args:
            tdata [numpy array]:
                An array containing the training data. The data should
                be of shape (N, 3) where N is the number of datapoints
                and each point should contain the variables, {x,y,label}
                (default None).
        """
        
        self.tdata = tdata
        # L is the "L i" distance measure, either i=1 for manhatten 
        # distance or i=2 for euclidean distance.
        L = self.order
        if(debug):print(np.shape(self.tdata))
        P = np.shape(self.tdata)[0]
        distance_list = np.empty((P, 2), dtype=float)
        for i in range(self.M):
            for j in range(self.N):
                # Create an array containing the cooridinates of the
                # observed unit from which to measure a distance.
                cur_unit = np.array([j, i], dtype = int)
                for k in range(P):
                    unit = np.array(self.tdata[k, 0:2])
                    distance_list[k, 0] = norm(cur_unit-unit, L)
                    distance_list[k, 1] = self.tdata[k, 2]
                winner = self._vote(distance_list, debug=debug)
                self.kmap[i, j] = winner
        return self.kmap
    
    def _k_neighbor(self, sarr=None, k=3, debug=False):
        """
        Takes a sorted array of distaces, and the number of those that get
        voting rights "k".
        """
        if(debug):print("\n\ncalled _k_neighbor method")
        if(debug):print("k={}".format(k))
        # Select k of the closest elements.
        k_neighbors_arr = sarr[0:k, :]  
        if(debug):print("k neighbors:\n", k_neighbors_arr)
        # Find the unique labels, and the count of each label with in
        # the k neighbors.
        label, count = np.unique(k_neighbors_arr[:, 1],
                                 return_counts=True)
        
        if(debug):print("labels and counts:\n", label, count)
        vote_mx = max(count)
        if(debug):print("max votes: ", vote_mx)
        winner_idxs = np.where(count==vote_mx)[0]
        vote = -1
        if(len(winner_idxs)>1):
            if(debug):print("==> {} way tie".format(len(winner_idxs)))
            # Since there is more than one winner, call _k_neighbor
            # recursively, but reducing k. This should repeat until only
            # one winner is left in the observed space.
            vote = self._k_neighbor(k_neighbors_arr, k-1, debug)
        else:
            vote = label[winner_idxs[0]]
        
        return int(vote)
        
    def _vote(self, distances=None, debug=False):
        # A lambda that selects for the first element in the array for
        # sorting.
        sort_key = lambda x:x[0]
        # Sort the selected distances in ascending order.
        sorted_distances = np.array(sorted(distances, key=sort_key))
        if(debug):print("sorted distances:\n", sorted_distances)
        
        vote = self._k_neighbor(sarr=sorted_distances,
                                k=self.k,
                                debug=debug
                                )
        if(debug):print(vote)
        return vote
    
if __name__ == "__main__":
    #%% get data
    import SelfOrganisingMap as SOM
    tdata = SOM.readData(fname="KN_trainingdata.txt", debug=False)
    
    #%% train on data
    testCase = kNearestNeighbor(k=2, measure="manhatten", shape=(9,9))
    collection = testCase.train(tdata=tdata)
    
    #%% plot 2
    
    fig2, ax2  = plt.subplots(1, 1, figsize = (5, 5))
    
    im0 = ax2.imshow(collection, cmap = 'bone_r')
    
    ax2 = su.config_axis(ax=ax2, x_lim=(-1, 9), y_lim=(-1, 9))
    fig2.colorbar(im0, ax = ax2)
    plt.show()
    
                

