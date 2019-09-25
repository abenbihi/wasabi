
import cv2
import numpy as np

import tools.SurveyLake as Survey


slice_id = -1
cam_id = -1

W, H = 511, 383

def proc(img): # resize to same size as cmu for concat
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    return img

col_l = []

# col: 0, 1
survey_id_l = [6,8]
survey_d = {}
for survey_id in survey_id_l:
    survey_d[survey_id] = Survey.Survey(slice_id, cam_id, survey_id)
pose_d = {}
for survey_id in survey_id_l:
    pose_d[survey_id] = survey_d[survey_id].pose_v

survey0 = survey_id_l[0]
survey1 = survey_id_l[1]
d = np.linalg.norm(np.expand_dims(pose_d[survey0], 1) -
        np.expand_dims(pose_d[survey1], 0), ord=None, axis=2)
order = np.argsort(d, axis=1)
matches = order[:,0]

#for idx0, idx1 in enumerate(matches):# show img for me to chose
for idx0, idx1 in [(111, matches[111]), (141, matches[141])]:
    print(idx0)
    img0 = survey_d[survey0].get_img(idx0)
    img1 = survey_d[survey1].get_img(idx1)
    sem_img0 = survey_d[survey0].get_sem_img(idx0)
    sem_img1 = survey_d[survey1].get_sem_img(idx1)
    
    img0, img1 = proc(img0), proc(img1)
    sem_img0, sem_img1 = proc(sem_img0), proc(sem_img1)

    line0 = np.vstack((img0, img1))
    line1 = np.vstack((sem_img0, sem_img1))
    out = np.vstack((line0, line1))
    
    h,w = img0.shape[:2]

    #out[:,w] = 255
    out[h,:] = 255

    out = cv2.resize(out, None, fx=0.5, fy=0.5,
            interpolation=cv2.INTER_NEAREST)
    col_l.append(out)
    
    #cv2.imshow('out', out)
    ##cv2.imwrite('plots/img/sunglare_lake.png', out)
    #if (cv2.waitKey(0) & 0xFF) == ord("q"):
    #    exit(0)


###############################################################################
# col: 2
survey_id_l = [0,2]
survey_d = {}
for survey_id in survey_id_l:
    survey_d[survey_id] = Survey.Survey(slice_id, cam_id, survey_id)
pose_d = {}
for survey_id in survey_id_l:
    pose_d[survey_id] = survey_d[survey_id].pose_v

survey0 = survey_id_l[0]
survey1 = survey_id_l[1]
d = np.linalg.norm(np.expand_dims(pose_d[survey0], 1) -
        np.expand_dims(pose_d[survey1], 0), ord=None, axis=2)
order = np.argsort(d, axis=1)
matches = order[:,0]

#for idx0, idx1 in enumerate(matches):# show img for me to chose
for idx0, idx1 in [(50, matches[50])]:
    print(idx0)
    img0 = survey_d[survey0].get_img(idx0)
    img1 = survey_d[survey1].get_img(idx1)
    sem_img0 = survey_d[survey0].get_sem_img(idx0)
    sem_img1 = survey_d[survey1].get_sem_img(idx1)
    
    img0, img1 = proc(img0), proc(img1)
    sem_img0, sem_img1 = proc(sem_img0), proc(sem_img1)

    line0 = np.vstack((img0, img1))
    line1 = np.vstack((sem_img0, sem_img1))
    out = np.vstack((line0, line1))
    
    h,w = img0.shape[:2]

    #out[:,w] = 255
    out[h,:] = 255

    out = cv2.resize(out, None, fx=0.5, fy=0.5,
            interpolation=cv2.INTER_NEAREST)
    col_l.append(out)
    
    #cv2.imshow('out', out)
    ##cv2.imwrite('plots/img/sunglare_lake.png', out)
    #if (cv2.waitKey(0) & 0xFF) == ord("q"):
    #    exit(0)

out_cmu = cv2.imread('plots/img/sunglare.png')
out = np.hstack(col_l)

print(out_cmu.shape)
print(out.shape)

out= np.hstack((out_cmu, out))
print(out.shape)


for i in range(4):
    out[int(i*H/2),:] = 255

for j in range(4):
    out[:,int(W*j/2)] = 255

cv2.imshow('out', out)
cv2.waitKey(0)

cv2.imwrite('plots/img/seg_fails.png', out)

