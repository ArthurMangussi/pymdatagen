## Documentation about Univariate mechanisms

**missing-data-generator** has a univariate approach that refers to only one feature in the dataset containing missing values. From this concept, the literature presents three mechanisms: Missing Completely at Random (MCAR), Missing Not at Random (MNAR), and Missing at Random (MAR). Each mechanism has different strategies for choosing the missing locations, which are implemented in this Python package.

The strategies to generate artificial missing data are described as follows:

### Missng At Random (MAR)

- lowest: Method to generate missing values in the feature `x_miss` by selecting the lowest values from an observed feature based on a specified missing rate;

- rank: A rank is created for the observed feature; this rank serves as the criterion for identifying the missing locations in the feature `x_miss`. While the original paper proposed a rank determined by the sum of all ranks, in the mdatagen package, we employ the maximum rank plus 1 to determine whether the index will be missing. New random numbers are generated to facilitate continued searching if the target missing rate is not achieved after 50 iterations.

- median: This function generates missing data in the feature `x_miss` by utilizing the median of an observed feature `x_obs`. The median of `x_obs` results in two groups—those equal to or higher than the median and those lower than the median. The group with values higher or equal to the median is chosen with a nine times greater probability based on a specified missing rate.

- highest: This function generates missing values in the feature `x_miss` by selecting the highest values from an observed feature.

- mix: This function generates missing values in the feature `x_miss` by incorporating the N/2 lowest values and N/2 highest values from an observed feature, where N is the missing data rate multiplied by the patterns from the dataset.

### Missing Not at Random (MNAR) 
- run: Method to generate missing values in the feature `x_miss` by selecting the threshold to choose values from an unobserved feature or feature itself. The threshold is a float between 0 and 1. If the threshold equals 0, the lowest values from an unobserved/observed feature will be selected to determine the missing locations in `x_miss`. Otherwise, if the threshold is 1, the highest values will be selected. This strategy is a generic implementation in the literature, and the user can employ various methodologies. The unobserved feature is not in the dataset; it consists of a range of random numbers with the same length as the patterns.

### Missing Completly at Random (MCAR)
- random: Method to randomly select missing locations in the feature `x_miss`.

- binomial: Method to determine feature `x_miss` locations to be missing using a Bernoulli distribution. In this method, we implement the Bernoulli distribution using `numpy.binomial`; occasionally, this method may not precisely generate the missing rate specified by the user.
