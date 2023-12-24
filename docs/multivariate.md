## Documentation about Multivariate mechanisms

**missing-data-generator** has multivariate approach that it refers about only one feature in dataset containing missing values. From this concept, the literature presents three mechanisms: Missing Completly at Random (MCAR), Missing Not at Random (MNAR) e Missing at Random (MAR). For each mechanism are different strategies to choose the miss locations, and all these are implemented in this Python package. For all methods in multivariate, the missing data rate is calculated for entire dataset. 

The strategies to generate artificial missing data are described as follows:

### Missing at Random (MAR)
- random: This method generates artificial missing data in `n_xmiss` features, selected randomly. If `n_xmiss` is not specified, it defaults to two. However, depending on the length of the input dataset, an error may occur. Both the observed feature and `x_miss` are randomly chosen. The missing locations in `x_miss` are determined by the lower values in the observed feature for each corresponding `x_miss`.

- correlated: This method generates missing data in features from the dataset, excluding the class (target). The strategy involves creating pairs/triples of features based on their correlation. For each pair, the most correlated feature with the class becomes `x_miss`, while the remaining one serves as the observed feature (`x_obs`) determining missing locations in `x_miss`. In the case of triples, the first and second features most correlated with the class are designated as `x_miss`, and the third is `x_obs`. The selection criterion involves choosing the lowest values from the observed feature. Given that only one feature in a pair experiences missing data, the missing rate is twice the input missing rate. For triples, it is 1.5 times the missing rate.

- median: This method generates artificial missing data in the dataset, excluding the class (target). The strategy entails creating pairs/triples of features based on their correlation. For each pair/triple, an observed feature (`x_obs`) is randomly selected. The median of x_obs establishes two groups—those lower and those equal or higher than the median. Subsequently, a group is randomly chosen, and the lowest values within it determine the missing locations in `x_miss`. It's essential to note that for triples, two features will be designated as `x_miss`. Similar to the correlated method, the missing rate is twice the input rate for pairs and 1.5 times for triples rate.

### Missing Not at Random (MNAR) 
- random: Method to randomly choose the n_xmiss features x_miss in the dataset to generate missing data. The missing locations in x_miss are determined by the lower values based on an unobserved or feature itself.

- correlated: The correlated method mirrors the approach used in the Correlated Method of the Missing at Random (MAR). However, in the case of Missing Not at Random (MNAR), this technique diverges by avoiding the use of observed features from the dataset. Instead, MNAR employs an unobserved random feature that is not present in the dataset or it uses the feature values itself.

- median: The median method shares similarities with the Median Method in the Missing at Random (MAR).  However, in the case of Missing Not at Random (MNAR), this technique diverges by avoiding the use of observed features from the dataset. Instead, MNAR employs an unobserved random feature that is not present in the dataset or it uses the feature values itself.

- MBOUV: Method to generate missing data based on the Missingness Based on Own and Unobserved Values (MBOUV), and it describe as follows: MBUV is applied to all nominal features and to half of the continuous ones, while MBOV (with lower values removal) is applied to the remaining half of the features. Both approaches are applied iteratively, and the split of continuous features is performed randomly.

- MBOV_randomness: Method to generate missing data based on Missingness Based on Own Values (MBOV) using randomness to choose missing locations in each input feature. Randomness is a float between 0 and 0.5, introducing stochasticity to generate the missing locations in x_miss. If randomness is equal to 0, only the lowest values will be selected.

- MBOV_median: Method to generate missing data based on Missingness Based on Own Values (MBOV) using the median to choose missing locations in each input feature. For this method, object types are not allowed. The missing locations are closer to the median of each feature. We utilize np.argsort of the difference between the current value and the median of the feature. The N lowest values are selected to be missing.

- MBIR: Method to generate missing data based on Missingness Based on Intra-Relation (MBIR). MBIR is a novel approach to generate missing data by the MNAR mechanism. This method is based on the MAR strategy and involves finding the lowest values from an observed feature x_obs. Then, x_miss receives the missing values, and an auxiliary indicator is created, which is 1 for missing and 0 otherwise. The user can select the statistical method to evaluate if there is evidence of a significant difference. Finally, the feature with the most statistically significant differences is selected to determine the missing locations, and after that, it is removed from the dataset. This feature is dropped because MNAR uses an unobserved feature from the dataset. It is important to clarify that if the user inputs all columns in the dataset, the entire dataset will be dropped.

### Missing Completly at Random (MCAR)
- random: Method to randomly generate missing data in all dataset. 

- binomial: Function to generate missing data in input columns by Bernoulli distribution for each attribute informed. It is important to clarify, similar to univariate MCAR binomial, occasionally, this method may not exactly generate the missing rate specified by the user.