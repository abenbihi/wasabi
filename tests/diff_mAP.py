"""Compute the diff in the mAP between the icra submission and now. I made an
error in an index while writing the retrieval results in rank.txt.
Here are the average diff for each xp: 
    - park: 3.18 %
"""
import numpy as np

survey_num = 10

old_trial = 7
old_fn = "/home/abenbihi/ws/tf/fuck_retrieval/res/fuck/%d/mAP.txt"%old_trial

new_trial = 0
new_fn = "/home/abenbihi/ws/tf/wasabi/res/wasabi/%d/mAP.txt"%new_trial

# i ignore the previous 10-th survey because it holds too few images (~ < 20)
# so image retrieval has no meaning
old_mAP = np.loadtxt(old_fn)[:survey_num, :]

new_mAP = np.loadtxt(new_fn)

diff = np.abs(old_mAP - new_mAP)
avg_diff = np.sum(diff) / np.prod(diff.shape)
print("avg_diff: %.2f%%"%(100*avg_diff))

