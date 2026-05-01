import copy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import pandas as pd
from sklearn.neural_network import MLPClassifier



def sample_data(data,y_name,method="stratified",test_size=0.8,random_state=42):
    """
    Generic sampling function supporting multiple strategies.

    Args:
        data: pd.DataFrame
        y_name: str, Target class of data
        random_state: int, should be randomly generated
        method: str, chosen str of the method, must be lower case
        test_size: float (between 0-1), determines the sampling ratio

    The strategies are:
    1. Stratified: used for the main experiments, sample such that the target
        class is evenly distributed in the sample. 

    2. random: regular uniform sampling. 
        This is generally the second best approach to take.

    3. undersampling: under samples the majority class in the dataset to compensate for imbalance.

    4. oversampling: over samples the minority class in the dataset to compensate for the imbalance

    5. None: no sampling occurs, returns full copy of the data.
    """

    # uses sklearn libraries 
    if method == "stratified":
        sample, _ = train_test_split(
            data,
            test_size=test_size,
            stratify=data[y_name],
            random_state=random_state
        )

    # uses sklearn libraries 
    elif method == "random":
        sample, _ = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state
        )

    # using own logic in pandas 
    elif method == "undersample":
        # balance classes by undersampling
        min_count = data[y_name].value_counts().min()

        sample = (
            data.groupby(y_name)
            .sample(n=min_count, random_state=random_state)
        )

    # using own logic in pandas 
    elif method == "oversample":
        max_count = data[y_name].value_counts().max()

        sample = (
            data.groupby(y_name)
            .sample(
                n=max_count,
                replace=True,
                random_state=random_state
            )
        )

    # entire dataset (no sampling)
    elif method == "none":
        sample = data.copy()

    else:
        raise ValueError(f"Unknown sampling method: {method}")

    return sample

def get_bin_model_stratified(data, y_name,bin_col, method="stratified", test_size_sampling=0.8, test_size_model=0.9, random_state=42):
    """
    Use a weak learner model to train data that has been sampled using stratified
    on target class of the data. (updated: can now take multiple sampling methods)

    Args:
        data: pd.DataFrame
        y_name: str, Target class of data
        bin_col: str, Binned feature class in the data
        random_state: int, should be randomly generated
        method: str, chosen str of the method, must be lower case
        test_size_sampling: float (between 0-1), determines the sampling ratio r
        test_size_model: float (between 0-1), determines the proxy model test size x%

    Returns:
        Trained classifier (clf) and test_train_split data
    """
    data2 = copy.deepcopy(data)

    min_val = data2[bin_col].min()
    max_val = data2[bin_col].max()

    min_row = data2[data2[bin_col] == min_val]
    max_row = data2[data2[bin_col] == max_val]

    # Combine the special rows into a guaranteed set
    guaranteed_samples = pd.concat([min_row, max_row]).drop_duplicates()
    remaining_data = data2.drop(guaranteed_samples.index)
    
    # old code to only do straitfied 
    # sample, _ = train_test_split(
    #     remaining_data, 
    #     test_size=test_size_stratified,  # Keep 20%, discard 80%
    #     stratify=remaining_data[y_name],
    #     random_state=random_state
    # )

    # use this function (details above) to do selection of the sampling type 
    sample = sample_data(data,y_name,method=method,test_size=test_size_sampling,random_state=42)
    sample = pd.concat([guaranteed_samples, sample])
    

    y = sample[y_name]
    X = sample.drop(columns=[y_name]) 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_model, random_state=random_state
    )


    # can use different proxy models, uncomment which one you want. 
    # clf = LogisticRegression().fit(X_train, y_train)
    clf = DecisionTreeClassifier(
        random_state=random_state, criterion="entropy",
    )
    # clf = MLPClassifier(random_state=random_state, max_iter=1)
    # clf = AdaBoostClassifier(n_estimators=100, learning_rate=1.0)

    clf.fit(X_train, y_train)
    return clf, X_train, X_test, y_train, y_test

def get_error_dict(clf, X_test, y_test, X_train, bin_name):
    """

    After deriving sampled values from a proxy model, generate both the error_count dict (p_i.e) 
    and the correlated total_count dict (p_i.c)

    Similar logic to getting MRE 

    Args:
        Trained classifier (clf) and test_train_split data (should be derived from sampling)

    Returns:
        numerator_dict: dict, derived from error_dict func, error_counts p_i.e
        denominator dict: dict , derived from error_dict func, total_counts p_i.c
    """

    # Derive predictions from trained model
    y_pred = list(clf.predict(X_test))
    y_test = list(y_test)

    unique_values = sorted(X_test[bin_name].unique())

    misclass = {}
    total_errors = 0
    errors = []
    # iterate over the predictions to get error_counts (numerator)
    for y in range(len(y_pred)):
        if y_pred[y] != y_test[y]:
            errors.append(1)
            if X_test[bin_name].iloc[y] not in misclass:
                misclass[X_test[bin_name].iloc[y]] = 0
            misclass[X_test[bin_name].iloc[y]] += 1
            total_errors += 1
        errors.append(0)
    
    result = [key for key, count in misclass.items() for _ in range(count)]

    # include the min and max values for the entire dataset 
    # (in case the test-train split doesn't include all values)
    global_min = min(min(X_train[bin_name]), min(X_test[bin_name]))
    global_max = max(max(X_train[bin_name]), max(X_test[bin_name]))

    if min(X_test[bin_name]) != global_min:
        result.append(global_min)
    if max(X_test[bin_name]) != global_max:
        result.append(global_max)
    counts1 = X_test[bin_name].value_counts()
    if global_min not in counts1:
        counts1[global_min] = 1
    if global_max not in counts1:
        counts1[global_max] = 1
    
    # total_counts (denom dict)
    numerator_counts = Counter(result)

    return numerator_counts, counts1


def repeated_stratified( data, y_name, bin_name, rand_states, test_size_sampling=0.8, test_size_model=0.9):
    '''
    This is the modified final version used in the code.

    For the Fair Bin algorithm, we use repeated straitfied sampling as
      there is fair too much variance otherwise with a single sample, however sampling returns better results 

    Uses the function of the get_bin_model_stratified, and repeats and stores the error over multiple iterations.

    Usually anything above T=3 is decent, T=5 is good. Overfitting with too much reps. is bad. 

    No need to use the error_dict function anymore, as it is not needed. 
    computation for this is done in this function due to repeating instances. 

    Args:
        data: pd.DataFrame
        y_name: str, Target class of data
        bin_col: str, Binned feature class in the data
        random_state: int, should be randomly generated
        method: str, chosen str of the method, must be lower case
        test_size_sampling: float (between 0-1), determines the sampling ratio r
        test_size_model: float (between 0-1), determines the proxy model test size x%

    Returns:
        Trained classifier (clf) and test_train_split data, numerator_dict and denominator_dict. 


    '''
    misclass = {}
    denominator_counts = Counter()
    # rep x times (this is the repeated stratified sampling part)
    for r in rand_states:
        clf, X_train, X_test, y_train, y_test = get_bin_model_stratified(data, y_name,bin_name, method="stratified", test_size_sampling=test_size_sampling, test_size_model=test_size_model, random_state=r)
        # print(X_train)
        y_pred = list(clf.predict(X_test))
        y_test = list(y_test)

        # iterate over the errors from the proxy model 
        for y in range(len(y_pred)):
            if y_pred[y] != y_test[y]:
                if X_test[bin_name].iloc[y] not in misclass:
                    misclass[X_test[bin_name].iloc[y]] = 0
                misclass[X_test[bin_name].iloc[y]] += 1
        
        temp_counts = Counter(X_test[bin_name])

        # add to counter dictionary 
        denominator_counts = temp_counts + denominator_counts 
  
    
    result = [key for key, count in misclass.items() for _ in range(count)]
    # print(len(result))

    # print(X_train)
    global_min = min(min(X_train[bin_name]), min(X_test[bin_name]))
    global_max = max(max(X_train[bin_name]), max(X_test[bin_name]))
    if min(X_test[bin_name]) != global_min:
        result.append(global_min)
    if max(X_test[bin_name]) != global_max:
        result.append(global_max)

    
    # ensure that min and max are present in the data
    if global_min not in denominator_counts:
        denominator_counts[global_min] = 1
    if global_max not in denominator_counts:
        denominator_counts[global_max] = 1
    
    numerator_counts = Counter(result)

    return numerator_counts, denominator_counts, clf, X_train, X_test, y_train, y_test


def min_max_range_split_dp(numerator_dict, denominator_dict, num_splits):
    """
    Find optimal split points that minimize the MAXIMUM range value using DP.

    Uses tabulation, bottom up DP method, using a precomputed array before iterating to find optimal solution.

    Use backtracking to retrive splits from idx

    Args:
        numerator_dict: dict, derived from error_dict func, error_counts p_i.e
        denominator dict: dict , derived from error_dict func, total_counts p_i.c
        num_splits: int (must be greater than 0 for valid output), 
            desired number of bins (in this case k-1 in comparison to the paper writing)
    
    Returns:
        max_range_value: The maximum range value (minimized)
        final_split: List [min_pos, internal_splits..., max_pos] (no duplicates, as defined by binning def)
    """
    # Other possible options for positions:

    """
    positions were previously determined more statically, 
    now we use adaptive merge so this distinction doesnt matter anymore
    """
    positions = sorted(denominator_dict.keys())
    # positions = sorted(numerator_dict.keys())
    # positions = sorted(numerator_dict.keys() | denominator_dict.keys())

    # logic to ensure that max and min values are included in the final bin choice. 
    denom_max = max(denominator_dict.keys())
    denom_min = min(denominator_dict.keys())
    if max(positions) != denom_max:
        positions.append(denom_max)
        numerator_dict[denom_max] = 0
    if min(positions) != denom_min:
        positions.insert(0,denom_min)
        numerator_dict[denom_min] = 0

    # Fill missing values with 0
    for pos in positions:
        # Validation, ensure that all errors can be computed correctly 
        if numerator_dict[pos] > denominator_dict[pos]:
            raise ValueError(
                f"Invalid input: numerator ({numerator_dict[pos], pos}) > denominator "
                f"({denominator_dict[pos]}) at position {pos}."
            )
    n = len(positions)
    min_pos = positions[0]
    max_pos = positions[-1]

        
    # more checks for valid splits 
    if num_splits >= n:
        raise ValueError(f"Cannot create {num_splits} splits from {n} positions")
    
    # V(i, j) calc
    def calculate_range_value(start_idx, end_idx):
        """Calculate value for range."""
        total_num = sum(numerator_dict[positions[i]] for i in range(start_idx, end_idx + 1))
        total_den = sum(denominator_dict[positions[i]] for i in range(start_idx, end_idx + 1))
        return total_num / total_den if total_den != 0 else float('inf')
    
    # Precompute all range values
    # creating the array for the values
    range_val = {}
    for i in range(n):
        for j in range(i, n):
            range_val[(i, j)] = calculate_range_value(i, j)
    
    num_groups = num_splits + 1
    INF = float('inf')
    
    # dp[i][g] = minimum achievable maximum when partitioning first i positions into g groups
    dp = [[INF] * (num_groups + 1) for _ in range(n + 1)]
    # store the actual selected partitions
    parent = [[-1] * (num_groups + 1) for _ in range(n + 1)]
    
    # Base Case
    dp[0][0] = 0
    
    # Fill DP table
    # start from one because we dont want to go to 0th index here 
    for i in range(1, n + 1):
        for g in range(1, min(i, num_groups) + 1):
            # Constrain k to prevent last position from being a split
            # k can be from g-1 to i-1, but exclude n-1 (last position index)
            # When i == n, we don't want k to be n-1
            # this is the recur rel (decide the smaller betwen then the two)
            max_k = min(i - 1, n - 2) if i == n else i - 1
            
            for k in range(g - 1, max_k + 1):
                if dp[k][g-1] < INF:
                    # grab computed array value
                    # not included logic in the pseudocode, previous referred to as lmax 
                    last_group_val = range_val[(k, i-1)]
                    # compare prev with curr, this is the max part of the recurrence rel
                    candidate = max(dp[k][g-1], last_group_val)
                    
                    # replace if candidate is better 
                    if candidate < dp[i][g]:
                        dp[i][g] = candidate
                        parent[i][g] = k

    # if there is nothing in the final array instance, no valid solution
    if dp[n][num_groups] == INF:
        raise ValueError("No valid solution found")
    
    # Backtrack to get split indices
    # parent only contains idx not actual numerical positions 
    split_indices = []
    i, g = n, num_groups
    
    while g > 1:
        k = parent[i][g]
        if k > 0:  # Exclude index 0 (will be min_pos boundary)
            split_indices.append(k)
        i = k
        g -= 1
    
    split_indices.reverse()
    
    # Convert indices to positions
    internal_splits = [positions[idx] for idx in split_indices]
    
    # Verify we have the correct number of internal splits
    if len(internal_splits) != num_splits:
        raise ValueError(
            f"Error: Expected {num_splits} internal splits but got {len(internal_splits)}. "
            f"Split indices: {split_indices}, Internal splits: {internal_splits}, "
            f"min_pos: {min_pos}, max_pos: {max_pos}, n: {n}"
        )
    
    # Build final splits: [min_pos, internal_splits..., max_pos]
    final_splits = [min_pos] + internal_splits + [max_pos]
    
    return dp[n][num_groups], final_splits



def adaptive_merge(numerator_dict, denominator_dict, target_groups=30):
    """
    Adaptively merge positions to create approximately target_groups.
    Merges more in dense regions, preserves structure in sparse regions.

    Can use the error_dict function to generate the initial numerator and denominator dict

    Args:
        numerator_dict: dict(), error counts
        denominator_dict: dict(), total counts
        target_groups: int, must be greater than num_bins for merging

    Returns:
        Merged dict with at maximum target_groups keys (merged_num, merged_denom) and mapping of group positions 
    """
    common_positions = sorted(set(numerator_dict.keys()) & set(denominator_dict.keys()))
    n = len(common_positions)
    
    if n <= target_groups:
        # Already few enough positions, return the same count mappings 
        return numerator_dict.copy(), denominator_dict.copy(), {p: [p] for p in common_positions}
    
    # Calculate position gaps
    gaps = []
    for i in range(len(common_positions) - 1):
        # get the gap by taking the difference between the current and the subsequent value
        gap = common_positions[i + 1] - common_positions[i]
        gaps.append((i, gap))
    
    # Sort by gap size (largest gaps first)
    gaps.sort(key=lambda x: x[1], reverse=True)
    
    # Select split points at largest gaps
    num_splits_needed = target_groups - 1
    split_indices = sorted([gaps[i][0] for i in range(num_splits_needed)])
    
    # Create groups based on splits
    merged_num = {}
    merged_den = {}
    position_mapping = {}

    # ensure that the max and min keys from the denom dict are maintained in selected dict
    max_den = max(denominator_dict.keys())
    min_den = min(denominator_dict.keys())
    merged_den[max_den] = 1
    merged_den[min_den] = 1
    merged_num[max_den] = 0
    merged_num[min_den] = 0

    
    start_idx = 0
    # iterate and create new mapping based on gaps 
    for split_idx in split_indices + [len(common_positions) - 1]:
        end_idx = split_idx + 1

        # identify all positions between choosen split indices
        group_positions = common_positions[start_idx:end_idx]
        
        # if there are no positions between indices, continue
        if not group_positions:
            continue
        
        # create new mapping by summing counts for the group positions
        group_key = group_positions[0]
        merged_num[group_key] = sum(numerator_dict[p] for p in group_positions)
        merged_den[group_key] = sum(denominator_dict[p] for p in group_positions)
        position_mapping[group_key] = group_positions
        
        # move to next idx
        start_idx = end_idx
    
    return merged_num, merged_den, position_mapping
