import torch

def calc_val_data_dice(TR,PR):
    """
    Calculating pixel-wise validation metrics 
    TR = true mask
    PR = pred mask
    """
    class_lables = [0,255]
    TP = (TR == class_lables[1]) & (TR == PR)
    TN = (TR == class_lables[0]) & (TR == PR)
    FP = (PR == class_lables[1]) & (TR != PR)
    FN = (PR == class_lables[0]) & (TR != PR)
    return TP,TN,FP,FN

def calc_val_loss_dice(TP,TN,FP,FN,eps=1e-8):
    """
    Dice score calculating
    """
    tp,tn,fp,fn = TP.sum(),TN.sum(),FP.sum(),FN.sum()
    dice_score = 2*tp / (2*tp+fp+fn + eps)
    return dice_score

def calc_val_data_IoU(TR,PR):
    """
    Calculating of intersection,union and target
    """
    class_lables = [0,255]
    I = torch.zeros([2])
    U = torch.zeros([2])
    T = torch.zeros([2])
    for i,j in enumerate(class_lables):
        I[i] = ((TR == j) & (PR == j)).sum()
        U[i] = ((TR == j) | (PR == j)).sum()
        T[i] = (TR == j).sum()
    return I,U,T

def calc_val_loss_IoU(I,U,T,eps=1e-8):
    """
    IoU and accuracy calculating
    """
    IoU = I/(U + eps)
    Acc = I/(T + eps)
    return IoU,Acc