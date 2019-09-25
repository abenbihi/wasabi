
import cv2
import numpy as np

import tools.cst as cst
import tools.SurveyLake as Survey

match_id = np.loadtxt('meta/lake/retrieval_img_ids.txt', dtype=np.int32)
match_survey = np.loadtxt('meta/lake/retrieval_surveys.txt', dtype=np.int32)
survey_num = match_id.shape[0]

img_dir = cst.LAKE_EXT_IMG_DIR
seg_dir = cst.LAKE_SEG_DIR
MODE = 'img'

img_id_l = [0, 10, 24, 50, 124, 100, 120, 160]

keep_d = {}
keep_d[0] = [0,2,3,4,6]
keep_d[10] = [0,9,6,4,5]
keep_d[24]  = [0,2,4,5,6]
keep_d[50]  = [0,3,4,7,8]
keep_d[124] = [0,3,5,6,9]
keep_d[100] = [1,9,8,2,6]
keep_d[120] = [0,3,5,7,2]
keep_d[160] = [0,1,2,5,6]


col_l = []

for img_id in img_id_l:
    #if img_id != 124:
    #    continue
    print(img_id)
    if MODE == 'img':
        img_fn_l = ['%s/%d/%04d/%04d.jpg'%(
            img_dir, 
            match_survey[j, img_id],
            match_id[j, img_id]/1000,
            match_id[j, img_id]%1000)
            for j in keep_d[img_id]]
    elif MODE == 'seg':
        img_fn_l = ['%s/%d/%04d/%04d.png'%(
            seg_dir, 
            match_survey[j, img_id],
            match_id[j, img_id]/1000,
            match_id[j, img_id]%1000)
            for j in keep_d[img_id]]
        #print(seg_dir)
        #print(img_fn_l)
    else:
        print("Error: Go fuck yourself")
        exit(1)

    img_l = [cv2.imread(img_fn) for img_fn in img_fn_l]
    img_l = [cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) 
            for img in img_l]
    out = np.vstack(img_l)
    col_l.append(out)
    
    #cv2.imshow('out', out)
    #if (cv2.waitKey(0) & 0xFF) == ord("q"):
    #    exit(0)

slice_num = 8
survey_num= 5
out = np.hstack(col_l)
if MODE == 'img':
    h,w = 240, 352
else:
    h,w = 240, 350
for j in range(1, slice_num):
    for i in range(1, survey_num):
        hh = h*i
        ww = w*j
        #print('hh: %d\tww: %d'%(hh, ww))
        if MODE == 'img':
            out[:,ww] = 255
            out[hh,:] = 255
        elif MODE == 'seg':
            out[:,ww] = 0
            out[hh,:] = 0
        else:
            out[:,ww] = 0
            out[hh,:] = 0

out = cv2.resize(out, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
cv2.imshow('out', out)
cv2.waitKey(0)

if MODE == 'img':
    hh,ww = out.shape[:2]
    h = int(hh/slice_num)
    w = int(ww/slice_num)
    img_size = out.shape[1]
    img = np.ones((h,img_size,3), np.uint8)*255
    
    for i, id_ in enumerate(img_id_l):
        text_pos_y = int(h/2)
        begin = i*w
        text_pos_x = begin+int(w/3)
        print(text_pos_x, text_pos_y)
        cv2.putText(img, 'Pos. %d'%i, (text_pos_x,text_pos_y), fontFace=0, fontScale=0.5, color=(0,0,0), thickness=1)
    
    cv2.imshow('img', img)
    cv2.waitKey(0)
    
    out = np.vstack((img, out))
    #cv2.imshow('out', out)
    #cv2.waitKey(0)

# lines text
hhh,www = out.shape[:2]
h = int(hhh/(survey_num + 1))
w = int(www/slice_num)

#h = 40
#num_class = len(label_name)
img_size = out.shape[0]

#h*num_class
img = np.ones((img_size,w,3), np.uint8)*255

#for i, survey_id in enumerate(survey_l):
for i in range(len(keep_d[0])):
    if MODE == 'seg':
        begin = i*h + int(0.5*h)
    elif MODE == 'img':
        begin = i*h + h
    else:
        print("Fuck you")
        exit(1)
    end = begin + h
    text_pos_x = h
    text_pos_y = begin+int(0.5*h)
    cv2.putText(img, 'T %d'%i, (text_pos_x,text_pos_y), fontFace=0, fontScale=0.5, color=(0,0,0))



cv2.imshow('img', img)
out = np.hstack((img, out))
cv2.imshow('out', out)
cv2.waitKey(0)


cv2.imwrite('plots/img/mosaic_lake_%s.png'%MODE, out)


