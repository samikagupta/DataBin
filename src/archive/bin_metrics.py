def frac_equal_bin_error(num_bins, clf, test_x, test_label, col, bin):
    """
    Compute the maximum fraction of misclassification error across bins. 
    This is the MRE descirbed in the paper 

    This function evaluates a trained classifier on test data and calculates,
    for each bin (defined by a column), the fraction of incorrect predictions.
    It returns the maximum error fraction across all bins (i.e., worst-case bin error).

    Args:
        num_bins (int): Number of bins used (not directly used in computation).
        clf: Trained classification model with a `predict` method.
        test_x (pandas.DataFrame): Test feature set.
        test_label (pandas.Series): True labels for the test set.
        col (str): Column name in `test_x` that contains bin assignments.
        bin (array-like): List of bin identifiers or boundaries.

    Returns:
        float: Maximum fraction of misclassified samples in any bin.
    """

    # get the predicted values 
    predicted = clf.predict(test_x)
    count  = 0

    # set up both counting dicts 
    both_dict = {}
    total_dict = {}
    missed = 0
    all_val = []
    for i in bin[:-1]:
        both_dict[i] = {}
    
    for tx in predicted:
        try:
             # count total
            total_dict[test_x.iloc[count][col]] += 1
        except:
            # if doesn't exist key, add
            total_dict[test_x.iloc[count][col]] = 1
        
         # if misclassified
        if tx != test_label.iloc[count]:
            missed += 1
            try:
                test_x.iloc[count][col]
            except:
                continue
            if test_x.iloc[count][col] not in both_dict: 
                both_dict[test_x.iloc[count][col]] = {}
            try:
                both_dict[test_x.iloc[count][col]][test_label.iloc[count]] += 1
            except:
                both_dict[test_x.iloc[count][col]][test_label.iloc[count]] = 1
        count += 1

    # sort dictionary so that calculate errors linearly, same as method of binning 
    both_dict_lab = sorted(both_dict)
    total_dict_lab = sorted(total_dict)

    frac_list = []

     # iterate over all values in order
    for b in total_dict_lab:
        try:
            frac_list.append(sum(both_dict[b].values())/total_dict[b])
        except:
            frac_list.append(0)
    
    # returns MRE, this is a decimal value, for %, use the function below as ref.
    return max(frac_list)

def frac_equal_bin_error_full(num_bins, clf, test_x, test_label, col, bin):
    '''
    Get the list of errors per bin (not just the Max Relative Bin Error, the list to derive the Error)
    MRE is basically is the accuracy within the bin, the maximum value. 

    Returns:
        new_list: list, this is the errors in a list. 
            By taking the max of this list, you will get the Maximum relative error 
    '''
    
    predicted = clf.predict(test_x)
  
    count  = 0
    both_dict = {}
    total_dict = {}
    missed = 0
    all_val = []

    # set up both dictionary to count
    for i in bin[:-1]:
        both_dict[i] = {}
        total_dict[i] = 0

    
    for tx in predicted:
        try:
            # count total
            total_dict[test_x.iloc[count][col]] += 1
        except:
            # if doesn't exist key, add
            total_dict[test_x.iloc[count][col]] = 1

        # if misclassified
        if tx != test_label.iloc[count]:
            missed += 1
            try:
                test_x.iloc[count][col]
            except:
                continue
            if test_x.iloc[count][col] not in both_dict: 
                both_dict[test_x.iloc[count][col]] = {}
            try:
                both_dict[test_x.iloc[count][col]][test_label.iloc[count]] += 1
            except:
                both_dict[test_x.iloc[count][col]][test_label.iloc[count]] = 1
        count += 1

    # sort dictionary so that calculate errors linearly, same as method of binning 
    both_dict_lab = sorted(both_dict)
    total_dict_lab = sorted(total_dict)

    frac_list = []

    # tracking numerator and denom. changes 
    all_top = []
    all_bottom = []

    # iterate over all values in order
    for b in total_dict_lab:
        try:
            frac_list.append(sum(both_dict[b].values())/total_dict[b])
            all_top.append(sum(both_dict[b].values()))
            all_bottom.append(total_dict[b])
        except:
            frac_list.append(0)
            all_top.append(0)
            all_bottom.append(total_dict[b])

    multiplier  = 100
    new_list = [item * multiplier for item in frac_list]
    # returns the % 
    return new_list

