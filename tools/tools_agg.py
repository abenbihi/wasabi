"""Implements primitives for local feature aggregations."""
import numpy as np

def load_cbk_Flickr100k():
    """Returns vlad codebook trained on Flicker100k."""
    return fvecs_read('meta/words/holidays/clust_k64.fvecs')

def load_cbk_delf_par1024():
    """Returns delf codebook trained on Paris6k with 1024 centroids."""
    return np.loadtxt("meta/k1024_paris.txt")

class Codebook():
    """Loads various codebook.
        
        NW: Number of visual Words.
        DW: Dimension of visual Words.
    """
    def __init__(self):
        self.d = {}
        # vlad # (NW, DW) = (64,128)
        self.d["flickr100k"] = load_cbk_Flickr100k

        # delf
        self.d["delf_k1024_paris"] = load_cbk_delf_par1024
    
    def load(self, key):
        loader = self.d[key]
        if not loader:
            raise ValueError("Unknown codebook: %s"%key)
        return loader()


def normalise_vlad(des_vlad, vlad_norm='l2'):
    """
    Flatten and normalise a vlad descriptor.
    Args:
        des_vlad: (vlad_dim,)
        vlad_norm: {l2, ssr}
    """
    if vlad_norm=='l2':
        des_vlad = des_vlad.flatten()
        norm = np.sqrt(np.sum(des_vlad**2))
        if norm > 1e-8:
            des_vlad /= np.sqrt(np.sum(des_vlad**2))
        else:
            des_vlad = np.ones(des_vlad.size)
    elif vlad_norm=='ssr':
        des_vlad = des_vlad.flatten()
        # power normalization, also called square-rooting normalization
        des_vlad = np.sign(des_vlad)*np.sqrt(np.abs(des_vlad))
        # L2 normalization
        des_vlad = des_vlad/np.sqrt(np.dot(des_vlad,des_vlad))
    else:
        raise ValueError("Unknown norm: %s"%vlad_norm)
    return des_vlad


def lf2vlad(lf, centroids, vlad_norm):
    """Computes vlad descriptors from the local features.
    Args:
        lf: (N, D) np array. N Local Features of dimension D.
        centroids: (NW, DW) np array. NW visual words of dimension DW.
    """
    NW, DW = centroids.shape[:2]
    N, D = lf.shape[:2] # dimension of local img feature
    if DW != D:
        raise ValueError("Error: D != DW."
                "Local descriptor dim is different than the visual word's one.")

    # cluster assignment
    c = np.expand_dims(centroids, 1) # row
    x = np.expand_dims(lf, 0) # col
    # euclidean distance between all pairs of local descriptors and visual
    # words/cluster
    d = np.linalg.norm(c - x, ord=None, axis=2) 
    x2c = np.argmin(d, axis=0) # x2c[j] = cluster of j-th local descriptor
    # one hot encoding of the cluster assignment
    x2c_hot = np.eye(NW)[x2c].T # x2c_host[i,j]=1 if x_j belongs to c_i

    # cluster distance
    gap = x - c # gap[i,j] = c_i - x_j

    # sum of eq 1 of vlad paper: 
    # aggregating local descriptors into a compact image representation
    toto = gap * np.expand_dims(x2c_hot, 2)
    v = np.sum(toto, axis=1)
    
    # normalisation
    v = normalise_vlad(v, vlad_norm)
    return v


def lf2bow(lf, centroids, vlad_norm):
    """
    Computes bow descriptors from the local features.
    Args:
        lf: (N, D) np array. N Local Features of dimension D.
        centroids: (NW, DW) np array. NW visual words of dimension DW.
    """
    NW, DW = centroids.shape[:2]
    N, D = lf.shape[:2] # dimension of local img feature
    if DW != D:
        raise ValueError("Error: D != DW."
                "Local descriptor dim is different than the visual word's one.")

    c = np.expand_dims(centroids, 1) # row
    x = np.expand_dims(lf, 0) # col
    
    # cluster assignment
    d = np.linalg.norm(c - x, ord=None, axis=2) # distance between clusters and des
    x2c = np.argmin(d, axis=0) # x2c[j] = cluster of j-th descriptor

    #create histogram
    unique, counts = np.unique(x2c, return_counts=True)
    hist = np.zeros(NW)
    hist[unique] = counts
    
    # normalisation
    v = normalise_vlad(hist, vlad_norm)

    return v

