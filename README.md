# Fair Bin: Minimizing Classification Disparity in Data Discretization

Abstract: Discretization is widely used in machine learning, for example because  they improve class separability by inducing piecewise-constant partitions of the feature space. Existing data binning methods are agnostic to downstream classification error, yet bin boundaries can directly influence which data groups face disproportionately high misclassification. While fairness-aware learning has been extensively studied at the model level, the impact of preprocessing decisions such as binning on group-level fairness has been largely overlooked. We introduce Fair Binning, a novel preprocessing framework that explicitly minimizes downstream classification disparity across bin groups. Central to our approach is Relative Maximum Bin Error ($\maxRelError$), a new fairness metric that evaluates binning quality per bin rather than globally. To optimize bin boundaries with respect to $\maxRelError$, we introduce the first sampling-based error estimation pipeline for data binning. A proxy model trained on stratified samples derives per-value error estimates independent of any specific bin configuration, enabling efficient and generalizable bin boundary selection. Evaluated on six UCI datasets, Fair Binning consistently reduces $\maxRelError$ compared to existing methods including Equal Width, Frequency, Entropy, Jenks, and Kmeans binning, with minimal impact on global classification accuracy, demonstrating that fairness-aware binning is both achievable and practical as a preprocessing step.

## Prerequisites
To install the package and prepare for use, run:
<pre><code>git clone https://github.com/samikagupta/DataBin.git

pip install -r requirements.txt
</code></pre>

The following python packages are required to run the code: pandas, sklearn, numpy.
