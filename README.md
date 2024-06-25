**Done by:**

- FadyMounir 49 - 0716
- IngyFam 49 - 2499
- MiraEmad 49 - 4627
- GeorgeReda 49 - 0301
- TonyAmir 49 - 10132

## Machine Learning Project


##### Machine learning has become an

##### essential part of our daily lives.


### Table of contents

```
Background
Context and motivation for the
project, Description of the
problem
```
```
Description of the data
being used
```
```
what the project aims to
achieve and how we did it.
```
```
Summarizing the findings, discuss how
the results can be improved
```

**Dataset**

**Objective/Goal & Method Discussion and Future Insight**


# 01

Context and motivation for the project, Description of the
problem

## Background


#### Background

```
The challenge is to assess the eligibility of
individuals or organizations for loans,
considering factors like credit score,
income, employment status, loan term,
amount, assets, and past loan
performance. Solving this problem matters
because it streamlines the lending process,
enhances decision-making, and promotes
fair, inclusive, and efficient lending
practices, ultimately benefiting both
applicants and the lending institutions.
```
"Imagine building a machine learning
project like crafting a digital superhero.
Each bit of code becomes a superpower,
helping us teach computers to learn from
data, predict outcomes, and solve
problems. It's like giving our technology the
ability to understand, adapt, and make
smart choices. The motivation? To use this
tech superhero to make things easier, solve
puzzles, and make our digital world a bit
more awesome each day!"


#### — Geoffery Moore

##### “Without big data analytics,

##### companies are blind and deaf,

##### wandering out on the web like deer

##### on a freeway”


# 02

Description of the data being used

## Dataset


#### Dataset

```
The loan approval dataset is a robust financial resource for evaluating loan eligibility,
featuring 13 columns and 4269 rows. The columns encompass a mix of continuous
numbers, discrete variables, and object/string data types. Essential details, such as
credit score, income, employment status, loan term, amount, and asset values, are
captured within columns like loan_id, no_of_dependents, education, self_employed,
income_annum, loan_amount, loan_term, cibil_score, residential_assets_value,
commercial_assets_value, luxury_assets_value, bank_asset_value, and loan_status.
This dataset is instrumental in developing predictive models and algorithms to
determine the likelihood of loan approval based on provided features.
```
###### The loan approval dataset


#### Dataset

###### The loan approval dataset

```
true and false are
0% as the attributes
are yes and no.
```

### Objective/Goal & Method

The primary objective of applying machine
learning models to the loan approval dataset is
to **predict the outcome of the "loan_status"**
column. This entails developing loan predictive
algorithms that leverage various features within
the dataset to accurately forecast **whether a
loan will be approved or denied.** By focusing on
this key column, the goal is to **create models** that
contribute to more informed **decision-making in
the lending process, enhancing the efficiency
and effectiveness of loan approvals.**

###### Objective/Goal Method

```
In the upcoming slides , we will
delve into the methodologies
employed to achieve our
objective. The use of machine
learning techniques and data
analysis to harness the
information embedded in the
dataset. We'll explore the process
of model development, feature
engineering, and the evaluation of
model performance
```

# 03

what the project aims to achieve and how we did it.

## Objective/Goal &

## Method


```
1 - PCA dataset with target column
2 - PCA dataset without target column
```
#### Methodologies

###### Pre-processing Clustering

```
1 - For normalized dataset
2 - For PCA dataset(without target
column)
```
###### Classification

```
1 - Data Cleaning
2 - Mapping(encoding)
3 - Normalization
```

#### 1 - Preprocessing

- Checking against duplicates and null values in the dataset however,
    the loan approval data set was clean
- Mapping discrete columns to 0s and 1s:
    1 - education -> Graduated: 1 , Not Graducated: 0
    2 - self_employed: Yes:1 , No: 0
    3 - loan_status: Approved:1 , Rejected: 0

```
Then, the new columns for the mapped values are education_numeric,
self_eployed_numeric, and loan_status_numeric, all added to the dataset
that will be used later. (df)
```
###### Mapping:

###### Data Cleaning:


###### Dataset before mapping:

###### Dataset after mapping:


#### 1 - Preprocessing

- QQ-plot for dataset (df) :

###### Normalization:


#### 1 - Preprocessing

- As the graphs illustrate, all columns need to be normalised because of the
    strong deviation from the line.
- A snippet of the dataset after normalisation (all values lie from 0 to 1)

###### Normalization:


#### Dimensionality Reduction

###### Principle Component Analysis:

- PCA reduces the dimensionality of data while preserving essential information
- PCA was done on the dataset without target column(loan_status_numeric)
    and with the target column ( pca_rawand X_pca).
- Number of components was determined using the cumulative variance.
- Number of components for PCA with target is 8 and 7 for PCA without target.


###### PCA with target column

###### PCA without target column


#### Dimensionality Reduction

###### Principle Component Analysis:

**- Variance bar graph: (with target)**
    The graph illustrates how much
variance each PC contributes
to the overall dataset.

```
It helps in deciding how many principal
components to retain based on the
amount of variance explained
```

#### Feature extraction and selection:

###### Correlated features:

- Table illustrates the correlation between features.


#### 2 - Clustering

**a) K-means:**

```
Elbow method Silhouette method Davis-Bouldin
```
###### 1 - PCA dataset with the target column


#### 2 - Clustering

**a) K-means:**

###### 1 - PCA dataset with the target column


#### 2 - Clustering

```
b) K-medoids:
```
###### 1 - PCA dataset with the target column


#### 2 - Clustering

```
c) DBSCAN
```
###### 1 - PCA dataset with the target column


#### 2 - Clustering

```
d) K-modes
```
###### 1 - PCA dataset with the target column


#### 2 - Clustering

```
e) Hierarchical:
```
###### 1 - PCA dataset with the target column


#### 2 - Clustering

```
a) K-means:
```
###### 2 - PCA dataset withOUT the target column


#### 2 - Clustering

```
b) K-medoids
```
###### 2 - PCA dataset withOUT the target column


#### 2 - Clustering

```
c) DBSCAN
```
###### 2 - PCA dataset withOUT the target column


#### 2 - Clustering

```
d) K-modes
```
###### 2 - PCA dataset withOUT the target column


#### 2 - Clustering

```
e) Hierarchical :
```
###### 2 - PCA dataset withOUT the target column


#### 2 - Clustering

```
1 - K-means:
Method: Divides data into K clusters based on mean values.
Centroids: Utilizes cluster centroids (means) to represent each cluster.
Outlier Sensitivity: Sensitive to outliers as they heavily impact the mean
calculation.
Applicability: Works well for globular-shaped clusters but struggles with outliers
and non-linear shapes.
```
###### Comparison between kmeans, kmedoids, DBSCAN:


#### 2 - Clustering

```
2 - K-medoids:
```
```
Method: Divides data into K clusters based on medoid points (data points with the
least average dissimilarity to all other points in the cluster).
Centroids: Uses actual data points as cluster representatives, providing
robustness against outliers.
Outlier Sensitivity: Less sensitive to outliers compared to K-means due to the use
of medoids.
Applicability: Suitable for non-Euclidean distances and arbitrary-shaped clusters
```
###### Comparison between kmeans, kmedoids, DBSCAN:


#### 2 - Clustering

```
3 - DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
```
```
Method: Identifies clusters based on the density of data points.
Cluster Shape: Capable of identifying clusters of arbitrary shapes and sizes, and it
detects noise as well.
Outlier Sensitivity: Less sensitive to outliers as it defines clusters based on density.
Applicability: Effective for datasets with varied cluster densities and shapes,
resistant to noise, and capable of discovering clusters of irregular shapes.
```
###### Comparison between kmeans, kmedoids, DBSCAN:


#### Visualization of clustering

#### results:

###### Use PCA to reduce

###### dimensions of original

###### data to 2


#### 3 - Classification

###### Classification on normalised dataset:

```
Splitting data: 70% training (2988 rows)
```
```
30% testing (1281 rows)
```
```
We used 3 Classification techniques:
```
- Logistic Regression
- Naïve Bayes
- Random Forest


#### 3 - Classification

###### Why did we use these types of Classification

```
Logistic Regression:
```
```
Used for binary classification, predicting the probability of an observation belonging to a particular
class. It models the relationship between a dependent binary variable and one or more independent
variables by estimating probabilities using a logistic function.
Best choice for problems where the outcome is binary or categorical.
```

#### 3 - Classification

###### Why did we use these types of Classification

```
Naïve Bayes:
```
```
probabilistic classification algorithm based on Bayes' theorem with an assumption of
independence between predictors. It calculates the probability that a data point
belongs to a certain class given the presence of certain features.
Pros:
It's fast, efficient, and performs surprisingly well in many real-world applications
```

#### 3 - Classification

###### Why did we use these types of Classification

```
Random Forest :
```
```
learning method that constructs a multitude of decision trees during training and
outputs the mode of the classes for classification tasks.
Pros:
Robust, less prone to overfitting, and capable of handling complex relationships in the
data. It excels in capturing diverse patterns, handling large datasets, and providing
high accuracy, making it a versatile and powerful tool for various machine learning
tasks.
```

#### 3 - Classification

###### Classification on normalised dataset:

Confusion Matrix:

```
Logistic Regression Naïve Bayes Random Forest
```

#### 3 - Classification

```
Logistic
Regression
```
**Naïve Bayes Random Forest**

```
● Accuracy: 91.10 %
● Precision: 91.11 %
● Recall: 91.10%
● F1 score: 91.10
● ROC AUC: 96.3%
```
```
● Accuracy: 93.29 %
● Precision: 93.46%
● Recall: 93.29%
● F1 score: 93.34%
● ROC AUC: 96.35%
```
```
● Accuracy: 96.41 %
● Precision: 96.40%
● Recall: 96.41%
● F1 score: 96.41%
● ROC AUC: 99.40
```
###### Classification on normalised dataset:


#### 3 - Classification

###### Classification on normalised dataset:

```
Comparison between 3 algorithms:
```

#### 3 - Classification

###### Classification on normalized dataset:

```
Visualization of the decision boundary along with the original data points :SVM
```

#### 3 - Classification

###### Classification on PCA dataset (without target

###### column) (raw_pca dataset) :

```
Splitting data: 70% training
```
```
30% testing
```
- Test data: The y-train and y-test were taken from normalized dataset as the
    PCA dataset used does not include target column (y value).
- Therefore, the PCA dataset only contains x values, which were divided into 30%
    for testing and 70% for training.


#### 3 - Classification

###### Classification on PCA dataset:

Confusion Matrix:

```
Logistic Regression Naïve Bayes Random Forest
```

#### 3 - Classification

```
Logistic
Regression
```
**Naïve Bayes Random Forest**

● Accuracy: 90.94 %
● Precision: 90.95 %
● Recall: 90.94%
● F1 score: 90.95%
● ROC AUC: 96.13%

```
● Accuracy: 92.43 %
● Precision: 92.60%
● Recall: 92.43%
● F1 score: 92.47%
● ROC AUC: 96.39%
```
```
● Accuracy: 94.30 %
● Precision: 94.35%
● Recall: 94.30%
● F1 score: 94.25%
● ROC AUC: 98.38%
```
###### Classification on PCA dataset:


#### 3 - Classification

###### Classification on PCA dataset:

```
Comparison between 3 algorithms:
```

#### 3 - Classification

###### Classification on PCA dataset:

```
Visualization of the decision boundary along with the original data points :SVM
```

# 04

```
Summarizing the findings, discuss how the results can be
improved
```
## Discussion and

## Future Insight


### 4 - Discussion

###### Classification on Normalized dataset:

```
Results showcase the performance of three classification algorithms—Logistic
Regression, Naïve Bayes, and Random Forest—applied to a dataset that has been
normalized. Random Forest demonstrated the best performance among the
three methods, with higher accuracy, precision, recall, and F1 score around 96%.
Indicating its exceptional predictive Prowers compared to both Logistic
Regression and Naïve Bayes in this specific scenario.
```
```
ROC AUC score was high for all classifications algorithms:
```
- Logistic regression: 96.3 %
- Naïve Bayes: 96.35 %
- Random Forest: 99.4%


### 4 - Discussion

###### Classification on PCA dataset:

```
Results indicate the performance of three different classification algorithms—
Logistic Regression, Naïve Bayes, and Random Forest—on a dataset that was pre-
processed using PCA (Principal Component Analysis) for dimensionality
reduction. Random Forest outperformed both Logistic Regression and Naïve
Bayes with higher accuracy, precision, recall, and F1 score around 94-95%.This
suggests that the Random Forest algorithm yielded the best performance
among the three methods on this PCA-transformed dataset, offering higher
accuracy and predictive power for the classification task at hand.
```
```
ROC AUC score was high for all classifications algorithms:
```
- Logistic regression: 96.13 %
- Naïve Bayes: 96.39 %
- Random Forest: 98.38%


### 4 - Discussion

###### Comparing Results between Normalized and PCA

###### datasets

- **Performance Improvement:** Across all algorithms, there's a consistent improvement in
    performance metrics when using the **normalized dataset compared to the PCA dataset.**
- **Random Forest:** Random Forest exhibits **significantly higher performance on both datasets** ,
    with a notable boost in accuracy, precision, recall, and F1 score on the normalized dataset
    compared to the PCA dataset, 94% compared to 96%.


### 4 - Discussion

###### Conclusion of Classification results

- **Data Scaling:** Normalization scales the data to a common range, facilitating better
    convergence for various algorithms. It helps different algorithms by making all features
    equally important in the decision-making process.
- **PCA's Dimensionality Reduction:** PCA reduces features, possibly discarding some information
    crucial for classification, impacting the performance of algorithms that rely on the full feature
    set.
- **Impact of Preprocessing:** Normalization standardizes the range of features, aiding algorithms
    to make more informed decisions. PCA, while reducing dimensions, might discard some
    variance which could impact performance.



