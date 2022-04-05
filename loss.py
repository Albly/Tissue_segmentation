def calc_val_data(TR,PR):
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

def calc_val_loss(TP,TN,FP,FN, eps = 1e-7):
    """
    Dice score calculating
    """
    tp,tn,fp,fn = TP.sum(),TN.sum(),FP.sum(),FN.sum()
    dice_score = 2*tp / (2*tp+fp+fn+eps)
    return dice_score

# size = 3
# TR = torch.randint(0,2,[size,size],dtype=torch.uint8)*255
# PR = torch.randint(0,2,[size,size],dtype=torch.uint8)*255

# TP,TN,FP,FN = calc_val_data(TR, PR)
# dice_score = calc_val_loss(TP,TN,FP,FN,eps=1e-7)
# print(TR)
# print(PR)
# TP,TN,FP,FN