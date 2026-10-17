# Fair Bin: Minimizing Classification Disparity in Data Discretization

Discretization is widely used in machine learning, for example because it improves class separability by inducing piecewise-constant partitions of the feature space. Existing data binning methods are agnostic to downstream classification error, yet bin boundaries can directly influence which data groups face disproportionately high misclassification. While fairness-aware learning has been extensively studied at the model level, the impact of preprocessing decisions, such as binning, on group-level fairness has been largely overlooked. We introduce Fair Binning, a novel preprocessing framework that explicitly minimizes downstream classification disparity across bin groups. Central to our approach is Maximum Relative Bin Error (MRE), a new fairness metric that evaluates binning quality per bin rather than globally. To optimize bin boundaries with respect to MRE, we introduce the first sampling-based error estimation pipeline for data binning. Using out-of-fold proxy models, we generate per-value estimated errors independent of any bin configuration and select bin boundaries that minimize the error corrected by upper bound Bonferroni confidence score. Evaluated on UCI datasets, Fair Binning consistently reduces MRE compared to existing methods including Equal Width, Frequency, Entropy, Jenks, and Kmeans binning, with minimal impact on global classification accuracy, demonstrating that fairness-aware binning is both achievable and practical as a preprocessing step.

## Prerequisites
To install the package and prepare for use, run:
<pre><code>git clone https://github.com/githubaccount/DataBin.git

pip install -r requirements.txt
</code></pre>

The following python packages are required to run the code: pandas, sklearn, numpy.
