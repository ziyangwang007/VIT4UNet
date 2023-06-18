
import numpy as np
import os




d = np.load('test3D.npy')
d = np.squeeze(d)
print(np.shape(d))
d = np.where(d>0,1,0)


a = np.load('results_3dunet.npy')
a = np.squeeze(a)
print(np.shape(a))
a = np.where(a>0,1,0)


dice = np.sum(d[a==1])*2.0 / (np.sum(d) + np.sum(a))
print('Dice similarity score is {}'.format(dice))


IOU = dice/(2-dice)
print('IOU score is {}'.format(IOU))



true_labels = a
pred_labels = d
# True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
 
# True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
 
# False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
 
# False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

print('TP:{}, FP:{}, TN:{}, FN:{}'.format(TP,FP,TN,FN))
TP=float(TP)
FP=float(FP)
TN=float(TN)
FN=float(FN)
 
accuracy = (TP+TN)/(TP+FN+FP+TN)
print('accuracy:{}'.format(accuracy))

precision = TP/(TP+FP)
print('precision:{}'.format(precision))

recall = TP/(TP+FN)
print('recall:{}'.format(recall))

specificity = TN/(TN+FP)
print('specificity:{}'.format(specificity))

