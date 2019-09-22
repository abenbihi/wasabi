"""Implementation of metrics.

Mostly copy-pasted from other repos:
- mean Average Precision from:
    http://lear.inrialpes.fr/people/jegou/code/eval_holidays.tgz.
- recall@N from:
    Copy-pasted from https://github.com/Nanne/pytorch-NetVlad
"""
import numpy as np

def get_order(order, gt_idx_v):
    """
    Look for rank of gt_idx in order.  And yes, I know that np.in1d exists but
    this mtf np.in1d can not bother to preserve the order 
    WARNING: Assumes both order and gt_idx have no duplicates.
    
    Args:
        order: np array of ints. List of db indices from nearest to
            furthest to the current query in desriptor space.
        gt_idx: array of int. Subset of db indices to find in order.
    
    Returns:
        The rank of each element of gt_idx_v in order.
    """
    o = np.tile(order, (gt_idx_v.shape[0],1))
    g = np.expand_dims(gt_idx_v, 1)
    o = o - g
    l, c = np.where(o==0)
    return c # order[c[0]] = gt_idx_v[0]
    

def recallN(order_l, gt_idx_l, n_values):
    """Computes the recall@N metric for N in n_values.
    
    Copy-pasted from https://github.com/Nanne/pytorch-NetVlad
    
    Args:
        order_l: list of np array of ints. order_l[i]: for the i-th query,
            array of db img indices retrieved from nearest to furthest in 
            descriptor space.
        gt_idx_l: list of np array of ints. gt_idx_l[i]: for the i-th query, 
            ground-truth order of the db img indices from nearest to furthest
            in descriptor space. 
    """
    correct_at_n = np.zeros(len(n_values))
    #TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(order_l):
        for i,n in enumerate(n_values):
        # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt_idx_l[qIx])):
                correct_at_n[i:] += 1
                break
    numQ = len(order_l) # for when I do partial retrieval
    recall_at_n = correct_at_n /numQ
    
    recall_l = []
    for i,n in enumerate(n_values):
        recall_l.append(recall_at_n[i])
        #print("- Recall@%d: %.4f"%(n, recall_at_n[i]))
    return recall_l


def parse_results_from_file(fname):
    """Parse the "rank.txt" and returns the rank and the filenames of the
        top_k retrieved database images.

    Copied from http://lear.inrialpes.fr/people/jegou/code/eval_holidays.tgz.

    Returns:
        A pair (rank, filename).
    """
    for l in open(fname,"r"):
        fields=l.split()
        query_name=fields[0]
        ranks=[int(rank) for rank in fields[1::2]]
        yield (query_name, list(zip(ranks,fields[2::2])) )


def parse_results_from_list(retrieved_l):
    """Parse the "rank.txt" and returns the rank and the filenames of the
        top_k retrieved database images.

    Copied from http://lear.inrialpes.fr/people/jegou/code/eval_holidays.tgz.

    Returns:
        A pair (rank, filename).
    """
    for l in retrieved_l:
        #fields = l.split()
        query_name = l[0]
        ranks = [int(rank) for rank in l[1::2]]
        yield (query_name, list(zip(ranks, l[2::2])) )

def score_ap_from_ranks_1(ranks, nres):
    """Compute the average precision of one search.

    Copied from http://lear.inrialpes.fr/people/jegou/code/eval_holidays.tgz. 
    
    Args:
        ranks = ordered list of ranks of true positives.
        nres  = total number of positives in dataset.
    """
    if nres == 0:
        raise ValueError("This query does not have a matching db img."
                "Remove it from the dataset and re-run.")

    ap=0.0 # accumulate trapezoids in PR-plot
    recall_step=1.0/nres # All have an x-size of:
    absc = np.linspace(0, 1, nres)
    for ntp, rank in enumerate(ranks):
        # y-size on left side of trapezoid:
        # ntp = nb of true positives so far
        if rank == 0: # rank = nb of retrieved items so far
            precision_0 = 1.0
        else:
            precision_0 = ntp/float(rank)

        # y-size on right side of trapezoid
        # ntp and rank are increased by one
        precision_1=(ntp+1)/float(rank+1)
        ap+=(precision_1+precision_0)*recall_step/2.0
    return ap

def mAP(rank_l, gt_name_d):
    """Computes the mean Average Precision of this retrieval.
    
    Copied from http://lear.inrialpes.fr/people/jegou/code/eval_holidays.tgz. 
    
    Args:
        rank_fn: file with the top_k retrieved images for each query (and
            the rank of the ground-truth database images if outside of the 
            top_k retrieved images). One line per query.
        gt_name_d: dictionnary of filenames. gt_idx_l[q_fn]: for the query
            which image name is q_fn, ground-truth order of the db img filanes 
            from nearest to furthest in descriptor space. 
    """
    sum_ap = 0. # sum of average precisions
    n = 0.
    gt = gt_name_d.copy()
    
    # loop over result lines
    for query_name,results in parse_results_from_list(rank_l):
        if results[0][0] == -1: # failed to match this query
            print('Pop this motherfucker')
            # this query failes
            sum_ap += 0
            gt_results=gt.pop(query_name) # ground truth
            continue

        results.sort() # sort results by increasing rank
        gt_results = gt.pop(query_name) # ground truth
        tp_ranks = [] # ranks of true positives (not including the query)
        rank_shift = 0 # apply this shift to ignore null results
        
        for rank,returned_name in results:
            if returned_name == query_name:
              rank_shift=-1
            elif returned_name in gt_results:
              tp_ranks.append(rank+rank_shift)
        local_ap = score_ap_from_ranks_1(tp_ranks,len(gt_results))
        sum_ap += local_ap
        n += 1
    if gt:
      # some queries left
      print("WARNING: no result for queries",gt.keys())
    mAP = sum_ap / n
    return mAP
