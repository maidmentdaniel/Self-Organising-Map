"""
Daniel Maidment

Mon May 27 13:59:07 2019
"""
import numpy as np
from numpy import pi
from numpy import sqrt
from numpy import exp
from numpy.linalg import norm
import matplotlib.pyplot as plt
import spyder_utilities as su

########################################################################


def readData(fname, debug=False):
    # Read data file into data array.
    data = []
    file = open(fname, 'r')
    for line in file:
        data_line = line.strip().split(',')
        if(debug):
            print(data_line)
        data.append([float(j) for j in data_line])
    file.close()
    data = np.array(data, dtype=int)

    return data


def writeData(fname, data):
    file = open(fname, 'w')
    M = np.shape(data)[0]
    N = np.shape(data)[1]
    for i in range(M):
        for j in range(N):
            file.write(str(data[i, j]))
            if(j < N-1):
                file.write(',')
        if(i < M-1):
            file.write('\n')
    file.close()

    return None


def normalise_data(data, debug=False):
    """
    Should normalise each feature of 'data' but not the lables.


    Args:
        data [numpy array]: Shape = (number of data points, number of
        features + integer label).
    """
    shape = np.shape(data)
    norm_data = np.zeros(shape)

    mu_arr = np.mean(data[:, 0:shape[1]-1], axis=0)
    std_arr = np.std(data[:, 0:shape[1]-1], axis=0)

    for j in range(shape[1]-1):
        norm_data[:, j] = (data[:, j]/std_arr[j])-mu_arr[j]

    norm_data[:, -1] = data[:, -1]

    if(debug):
        for i in range(shape[0]):
            if(i % (shape[0]/5) == 0):
                print(data[i, :])
                print(norm_data[i, :], '\n')

    return norm_data


def most_common(lst, n):
    if len(lst) == 0:
        return -1
    counts = np.zeros(shape=n, dtype=np.int)
    for i in range(len(lst)):
        counts[int(lst[i])] += 1

    return np.argmax(counts)


def calc_euclid_matrix(map_mat):
    M, N, d = np.shape(map_mat)
    out_mat = np.zeros((M, N))
    euclid_dist_sum = 0
    n = 0
    for j in range(M):
        for k in range(N):
            euclid_dist_sum = 0
            n = 0
            cur_map_vec = map_mat[j, k, :]
            if(j-1 >= 0):  # North
                euclid_dist_sum += norm(cur_map_vec
                                        - map_mat[j-1, k, :])
                n += 1
            if(k+1 < N):  # East
                euclid_dist_sum += norm(cur_map_vec
                                        - map_mat[j, k+1, :])
                n += 1
            if(j+1 < M):  # South
                euclid_dist_sum += norm(cur_map_vec
                                        - map_mat[j+1, k, :])
                n += 1
            if(k-1 >= 0):  # West
                euclid_dist_sum += norm(cur_map_vec
                                        - map_mat[j, k-1, :])
                n += 1

            out_mat[j][k] = euclid_dist_sum/n

    return out_mat


class selfOrganisingMap:
    """
    The class generates a self organising map.

    Args:
        gShape [int tuple]:
            The size of the SOM that is generated.
        lam [int]:
            The number of training iterations that the SOM will
            undergo.
        rate_mx [float]:
            The learning rate of the SOM. Should be between 0 and 1.
        sigma [float]:
            The standard deviation used in the gaussian neighborhood
            function.
        data:
            The numpy array containing the training data. Should be of
            a format where each row is an input vector. The last element
            of which is the lable. The lable should be encoded as an
            integer.
        neighborfunction ["gaussian", "linear"]:
            Select the neighborhood function.
        randomseed [int]:
            If TRUE, this will seed the random number denerator with
            randomseed.
        debug [bool]:
            Set true for debugging lines to execute.


    """

    def __init__(self, gShape=(5, 5),
                 lam=5000, rate_mx=0.5, sigma=1,
                 data=None, neighbor="gaussian",
                 randomseed=None, debug=False):
        self.sigma = sigma
        if(randomseed):
            np.random.seed(randomseed)
        self.gShape = gShape
        self.M, self.N = self.gShape
        self.lam = lam
        self.range_mx = self.M+self.N
        self.rate_mx = rate_mx
        self.data = data
        self.N_data = np.shape(self.data)[0]
        self.d = np.shape(self.data)[1]-1
        self.debug = debug
        self.SOM = np.random.sample((self.M, self.N, self.d))
        self.cur_range = self.range_mx
        self.pc_left = 1.0
        self.data_vec = None
        self.BMU = (None, None)
        if(neighbor == "gaussian"):
            self.neighbor = self._gaussian_neighbor
        elif(neighbor == "linear"):
            self.neighbor = self._linear_neighbor
        else:
            print("you selected a non-existing neighborhood function",
                  "\n defaulting to a gaussian.")
            self.neighbor = self._gaussian_neighbor

        return None

    def train(self):
        """
        This function is the main function called to train the SOM.
        It should run for lam training cylces.
        """
        # Run through lam training cycles.
        print("Started Training")
        for s in range(self.lam):
            if(not self.debug and s % (self.lam/4) == 0):
                print("\rCompleted: {:.2%}".format(s/self.lam),
                      end="\r")
            elif(self.debug and s % (self.lam/10) == 0):
                print("\rCompleted: {:.2%}".format(s/self.lam),
                      end="\r")

            # The percentage of iterations left.
            self.pc_left = 1-(s/self.lam)
            # The learning rate as a function of the number of remaining
            # iterations.
            self.cur_rate = self.rate_mx*(1/(1+10*((s/self.lam)**2)))
            # Choose a random training input vector.
            t = np.random.randint(0, self.N_data)
            self.data_vec = self.data[t, 0:self.d]

            self._find_BMU(self.data_vec)
            self._update_SOM()

        print("\nFinished Training")
        return None

    def _find_BMU(self, vec=None):
        # Traverse map and find BMU.
        temp_min_dist = self.M+self.N
        for j in range(self.M):
            for k in range(self.N):
                cur_dist = norm(vec-self.SOM[j, k, :])
                # Find SOM point with smallest euclidean distance from
                # site.
                if(cur_dist <= temp_min_dist):  # i
                    temp_min_dist = cur_dist
                    self.BMU = (j, k)
        return self.BMU

    def _update_SOM(self):
        for j in range(self.M):
            for k in range(self.N):
                cur_unit = np.array((j, k))
                BMU_arr = np.array(self.BMU)
                theta_i = norm(cur_unit-BMU_arr, 1)

                theta = self.neighbor(theta_i)

                self.SOM[j, k, :] = (self.SOM[j, k, :]
                                     + theta*self.cur_rate
                                     * (self.data_vec-self.SOM[j, k, :])
                                     )

        return None

    def _gaussian_neighbor(self, x):
        """
        Given some l1 length x, this should return:
            y=(1/(sigma*sqrt(2*pi)))*exp((-1/2)*(x/sigma)**2)
            which is used to reduce the leanring rate of the neuron as
            a function of the distance from the current BMU.
            "TODO"
            Test Sigma shrinking over time.
        Args:
            x [float or int]:
                Some measure of distance from the current BMU to some
                coordinate.
        Returns:
            y [float]:
                A value determined by a Gaussian distribution that
                should effect the learning rate.

        """
        # Here the idea is that over time sigma will approach half it's
        # original value.
        sig = self.sigma*(0.5+0.5*self.pc_left)
        y = (1/(sig*sqrt(2*pi))) * exp((-1/2)*(x/sig)**2)

        return y

    def _linear_neighbor(self, x):
        """
        Here x is some l1 measure of distance from the current BMU to
        the current unit of obeservation. The idea is that the learning
        rate should decrease linearly as function of distance to the BMU
        within some obeserved radius. The radius shrinks over time.
        Args:
            x [float or int]:
                Some measure of distance from the current BMU to some
                coordinate.
        Returns:
            y [float]:
                A value determined by a linear function of the distance
                that should effect the learning rate.
        """
        cur_range = int(self.range_mx*self.pc_left)+1
        y = 0
        if(x<=cur_range):
            y = x/cur_range
        return y
        
#Helper functions
    def visualise_data(self, n = 3):
        """
        Generate a map of the data on an MxN grid.
        """
        self.count_map = np.zeros((self.M, self.N), dtype=object)
        self.display_map = np.zeros((self.M, self.N), dtype=int)
        for i in range(self.M):
            for j in range(self.N):
                self.count_map[i][j] = []
        # Iterate through all the training data.
        for i in range(len(self.data)):
            self.data_vec = self.data[i, 0:self.d]
            self._find_BMU(self.data_vec)
            r, c = self.BMU
            self.count_map[r, c].append(self.data[i, -1])
            
        for i in range(self.M):
            for j in range(self.N):
                val = most_common(self.count_map[i][j], n)
                self.display_map[i][j] = val
        return self.display_map
    
    def get_markers(self, n=3, debug=False):
        """
        Generate an array containing coloured markers and the respective
        x,y coordiantes associated with the labels. Generated from the
        visualise_data function.
        Args:
            n [int]:
                Is the number of labels (default 3).
        Returns:
            marker_arr [numpy array]:
                Is the array of shape (M*N, 3), where each row is of the
                form [x_coord, y_coord, str_marker].
        """
        display_map = self.visualise_data(n=n)
        M = self.M
        N = self.N
        
        marker_arr = []
        for i in range(M):
            for j in range(N):
                label = int(display_map[i, j])
                if(label > -1):
                    marker_arr.append([j, i, label])
        return np.array(marker_arr, dtype=int)
    
    def visualise_markers(self,
                          n=3,
                          style="label",
                          target=None,
                          debug=False
                          ):
        """
        Args:
            n [int]:
                Is the number of labels (default 3).
            style [str]:
                The style of marker returned. If "label" then the
                integer label of "${}$".format(label) is replaced for
                each (x, y) element. Otherwise  if "symbol" a symbol
                marker and color is associated with each (x, y) element.
                Else if "target" then the marker is asscotiated with a
                passed dictionary of labels (default "label").
            target [dictionary]: 
                Associates a specfic label with some target string
                (default None).
        Returns:
            marker_list [list]:
                Of the same shape and structure as marker_arr, but with
                a string in place of label.
        """
        marker_arr = self.get_markers(n=n)
        marker_list = []
        if(style=="symbol"):
            marker = ["bo", "go", "ro", "co", "mo", "yo", "ko",
                      "b*", "g*", "r*", "c*", "m*", "y*", "k*",
                      "bX", "gX", "rX", "cX", "mX", "yX", "kX",
                      "bv", "gv", "rv", "cv", "mv", "yv", "kv",
                      "bs", "gs", "rs", "cs", "ms", "ys", "ks"]
            for i in range(len(marker_arr)):
                y = marker_arr[i, 0]
                x = marker_arr[i, 1]
                z = marker[marker_arr[i, 2]]
                marker_list.append([y, x,z])
                
        elif(style=="label"):
            for i in range(len(marker_arr)):
                y = marker_arr[i, 0]
                x = marker_arr[i, 1]
                z = marker_arr[i, 2]
                str_val = str("$"+"{}".format(z)+"$")
                if(debug):print(str_val)
                marker_list.append([y, x, str_val])
                
        elif(style=="target"):
            for i in range(len(marker_arr)):
                y = marker_arr[i, 0]
                x = marker_arr[i, 1]
                z = marker_arr[i, 2]
                t = target[z]
                str_val = str("$"+"{}".format(t)+"$")
                marker_list.append([y, x, str_val])
        else:
            print("{} for style argument passed to\n" 
                  +"SelfOrganisingMap.visualise_markers method doesn't"
                  +"\nexist, defaulting to 'symbol' method.")
            marker = ["bo", "go", "ro", "co", "mo", "yo", "ko",
                      "b*", "g*", "r*", "c*", "m*", "y*", "k*",
                      "bX", "gX", "rX", "cX", "mX", "yX", "kX",
                      "bv", "gv", "rv", "cv", "mv", "yv", "kv",
                      "bs", "gs", "rs", "cs", "ms", "ys", "ks"]
            for i in range(len(marker_arr)):
                y = marker_arr[i, 0]
                x = marker_arr[i, 1]
                marker_list.append(marker[marker_arr[i, 2]])
        return marker_list
                                
    
    def set_range_mx(self, r_mx):
        if(r_mx< self.M+self.N):
            self.range_mx = r_mx
        else:
            print("The maximum range for the neighborhood function")

if __name__ == "__main__":
    
    dataFile = readData('iris_data.txt')
#    ndataFile = normalise_data(dataFile, False)
    testSOM = selfOrganisingMap(gShape=(10, 10),
                                lam=1000, rate_mx=1, 
                                neighbor="gaussian",
                                data=dataFile, randomseed=12,
                                debug=True)

    testSOM.train()
    som_map = testSOM.SOM
    U_mat = calc_euclid_matrix(som_map)
#%%    
    target = {0: "x", 1: "y", 2: "z"}
    marker_arr = testSOM.visualise_markers(n=3,
                                           style="label",
                                           target=target,
                                           debug=False
                                           )
    
#%% plot 1
    plt.style.use("myPaper_grey")
    fig, ax  = plt.subplots(1, 1, figsize = (5, 5))
    
    im0 = ax.imshow(U_mat, cmap = 'bone_r')
    
    for k in range(len(marker_arr)):
        ax.plot(marker_arr[k][0], marker_arr[k][1],
                marker=marker_arr[k][2], color='r',
                markersize=8
                )
    ax = su.config_axis( ax=ax, 
                        x_lim=(0, 9), X_0=10,
                        y_lim=(0, 9), Y_0=10,
                        grd=False,
                        mult_x=0.1, mult_y=0.1
                        )
    fig.colorbar(im0, ax = ax)
    plt.show()

#%% k nearest neighbor
    import k_neighbor as KN
    
    kn_training_data = testSOM.get_markers(n=3)
#    writeData("KN_trainingdata.txt", kn_training_data)
#    print(kn_training_data)
    k_map = KN.kNearestNeighbor(k=3, measure="manhatten", shape=(30,30))
    collection = k_map.train(tdata=kn_training_data)
    
#%% plot 2
    
    fig2, ax2  = plt.subplots(1, 1, figsize = (5, 5))
    
    im0 = ax2.imshow(collection, cmap = 'bone_r')
    
    for k in range(len(marker_arr)):
        ax2.plot(marker_arr[k][0], marker_arr[k][1],
                marker_arr[k][2],
                markersize=8
                )
    ax2 = su.config_axis(ax=ax2, x_lim=(0, 30-1), y_lim=(0, 30-1))
    fig2.colorbar(im0, ax = ax2)
    plt.show()
