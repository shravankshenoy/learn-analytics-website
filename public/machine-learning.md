
### Basics
* Feature

* Weights

* Model



### Statistics
* Sample :

* Population : 

* Estimator : A rule/formula for calculating an estimate of a population parameter based on sample of data eg. To estimate the average height of all students at a large university, estimator would be mean formula. Estimator is just a formula

* Expected Value : Weighted average of all possible outcomes

```
Example

Imagine a simple game where you roll a six-sided die, and win $2 if you roll a 1, and lose $1 if you roll any other number. Expected Value = (1/6) * $2 - (5/6) * $1 = -$ 0.5

Meaning : Expected value is not about any single game. Instead, over a large number of plays, the average outcome per game will approach -0.5 dollars. For example, if you play 100 games, you can expect to lose approximately $50 overall.


```

* Bias : Bias of an estimator is the difference between this estimator's expected value and the true value of the parameter being estimated

* Variance : How data is spread around its mean

* Variance (machine learning) : How sensitive a model is to change in data point

* Bias-Variance tradeoff



### Model Development Life Cycle

1. Define requirements
- What are we trying to optimize?
- What are latency requirements?

2. Analyze data sources
-  What kind of data do we have? (tabular, time series, supervised or unsupervised)
-  What is the refresh frquency? (streaming or batch)
-  What kind of problem? (supervised or unsupervised, regression or classification, balanced or imbalanced data)

3. Feature engineering
- Find inter-feature correlation and correlation bw feature and target variable (if supervised problem)
    - Feature-target correlation
    - Inter-feature correlation    
    - Creating Interaction Features: If two features are individually weakly correlated with the target but together have a strong relationship, you can create interaction terms. eg. using bmi instead of weight and height


- Create features using
    - difference (profit = selling price - cost price)
    - ratio (such as bmi = height/(weight)^2)
    - if-else condition 
    - log
    - square root
    - grouping (number of counts in a 3 minute window)
    - extraction (getting day of week, time of day from a timestamp)


4. Model building


5. Model deployment 

6. Model monitoring




### Feature Engineering

* Some important questions to answer in a dataset are
    * Target Variable Identification
    * Do the features contain information that helps predict the target?

* To identify if features contain information one approach we can use is **shuffling the target**. If the target is truly related to the features, a model like XGBoost can find patterns and reduce error, if target is just noise, performance will look same if we shuffle the target. Shufflling breaks relationship between features and target. The approach is as follows
    * Train an XGB on the original data features and target (if there were non-linear signals, then XGB would find them. If XGB cannot find any signal in the original dataset it indicates that it isn't "low signal" or "noisy signal", instead it is just random numbers with "no signal")
    * Shuffle data 100 times and train XGB on each shuffle and calculated CV RMSE. You get a distribution of scores under the no signal assumption i.e. null hypothesis is no relationship between features and target
    * Compare the CV RMSE scores of the XGB trained with original target versus the XGB trained with the random target. If the CV score from using the original target is within z-score -2 to 2 of the CV score from using random targets, we conclude that the original data is just random numbers.

```
def root_mean_squared_error(true,pred):
    m = np.sqrt(np.mean( (true-pred)**2.))
    return m


## Code for shuffling the target
def train_xgb(
    orig,
    FEATURES,
    TARGET,
    repeats=10,
    folds=5,
    seed=42,
    **kwargs,  
):
    
    params = {
        "objective": "reg:squarederror",   
        "eval_metric": "rmse",                       
        "learning_rate": 0.3,
        "max_depth": 6,                    
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": seed,      
        "alpha": 2.0,                      
        "min_child_weight": 10,
    }
    
    scores = []
    print(f"Training {repeats+1} XGB KFold CVs. ")
    for repeat in range(repeats+1):

        # FIRST ITERATION USES ORIGINAL TARGET
        # SUBSEQUENT ITERATIONS USE RANDOM TARGETS
        
        train = orig.copy()
        
        # Shuffle the target
        if repeat>0:
            t = orig[TARGET].values
            np.random.shuffle(t)
            train[TARGET] = t

        # KFOLD CV
        oof_preds = np.zeros(len(train))
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train)):

            # TRAIN AND VALID DATA SPLITS
            X_train = train.iloc[train_idx][FEATURES].copy()
            y_train = train.iloc[train_idx][TARGET]
            X_valid = train.iloc[val_idx][FEATURES].copy()
            y_valid = train.iloc[val_idx][TARGET]
            dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
            dval   = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

            # TRAIN XGB
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=10_000,
                evals=[(dtrain, "train"), (dval, "valid")],
                early_stopping_rounds=100,
                verbose_eval= 0,
            )

            # INFER XGB
            oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))

        # COMPUTE OOF CV SCORE
        m = root_mean_squared_error(train[TARGET],oof_preds)
        if repeat==0:
            print(f"When using original target CV RMSE = {m:.2f}")
        elif repeat==1:
            print(f"When using random target CV RMSE = {m:.2f}\nAdditional random trials... ",end="")
        else: 
            print(f"{repeat-1}, ",end="")
        scores.append(m)

    print(); print()
    return scores

def display_result(
    scores,
    name="",
    **kwargs, 
):
    s = np.std(scores[1:])
    m = np.mean(scores[1:])
    z = (scores[0]-m)/s
    
    print(f"z-score = {z:.2f} of Original Target CV vs. Random Target CVs")
    plt.hist(scores, bins=100, label='random targets')
    ymax = plt.ylim()[1]
    plt.plot([scores[0],scores[0]],[0,ymax/2.],color='black',linewidth=5,label='original target')
    plt.legend()


orig = train_df
TARGET = "BeatsPerMinute"
FEATURES = list( orig.columns )
FEATURES = [f for f in FEATURES if f != TARGET]
scores = train_xgb(orig, FEATURES,TARGET)
display_result(scores)

```

* We generate ~100 CV RMSE scores from random targets. Then compare the original CV RMSE against this distribution. For z-scores, the common cutoff is -2, 2 (≈ 95% confidence). t critical at 95% confidence for n =100 i.e. df = 99 ≈ ±1.984 (almost the same as z = ±1.96).

* z-score in a z-test tells you “how extreme is my observed value compared to what I’d expect if the null were true, measured in units of standard deviations?” If z-score is between -2 and 2, this indicates the z-score is not extreme enough to provide statistically significant evidence to reject the null hypothesis, suggesting the observed difference is likely due to random chance, not a true effect. In other words, the orignal cv rmse falls within the distribution of random cvs rmse(since it is less than 2 SD away), hence we believe the original data also has random targets since it falls within that distribution

* Population Distribution vs Sampling Distribution : If we take height of 150 students in class, population distribution is just plotting the histogram of 150 heights, whereas sampling distribution of mean is we take multiple samples of 15 students, for each sample, find the mean and then plot histogram of the means obtained from different samples. Hence population distribution focuses on individual data points from one large group, while the sampling distribution focuses on the variability of a statistic calculated from many smaller groups (samples). 

* Point estimate :  Single specific numerical value derived from a sample, used as the best guess for an unknown population parameter. For example, 62 is the average mark achieved by a sample of 15 students randomly collected from a class of 150 students, which is considered the mean mark of the entire class. Since it is in the single numeric form, it is a point estimator.

* Interval estimate : If the mean (or any population statistic) computed from a single sample is x, can you give an interval such that you are 95% sure the mean of the population lies in this interval

* Intuition behind hypothesis testing : Is your sample data "extreme" or "unlikely" enough to cast doubt on initial, default assumption (the null hypothesis). We evaluate how unusual our results are against the baseline expectation to decide if a real difference exists, rather than just random chance. (For example we reject the alternate hypothesis because there is a chance we get a sample where the mean is greater than 90mph although true mean is less than 90mph, so we got a sample which proves alternate hypothesis just because we were lucky rather than reflecting the actual population)

* Given the null hypothesis is true, how likely is it that random sampling would give us a mean as far away from population mean as our observed sample mean.
```
Imagine you think the average IQ of a population = 100.
You take a sample of 25 people, and get:

sample mean = 103
sample stddev = 15

Expected standard error = 𝑠 / sqrt(25) = 15/5 = 3
Difference from hypothesized mean = 103 − 100 = 3

So t= 3 / 3 = 1

Interpretation: The observed mean is only 1 standard deviation away from 100 (std dev here refers to that of sampling distribution).
That’s not unusual in a sampling distribution → so we don’t reject H₀.

But if sample mean = 110
Then t= (110−100)/ 3 = 3.33.
Now the sample mean is 3+ SEs away → very unlikely under H₀ → reject H₀.

```

* **Standard error (SE) is the standard deviation of the sampling distribution of a statistic** As number of samples increases, std dev of the sampling distribution decreases as we get more and more values closer to each other

* For both regression and classification problems, we need to analyze the target variable distribution for better model performance

* In regression, if target variable is highly skewed, we can apply transformations like
    * Log transformation
    * Box cox transformation
    * Square root transformation


```
import matplotlib.pyplot as plt
from scipy.stats import probplot, boxcox

log_transformed  = np.log1p(df[target])

boxcox_transformed = boxcox(df[target])

sqrt_transformed = np.sqrt(df[target])

## Visualizing impact of transformation
sns.histplot(log_transformed, kde=True, bins=30)
probplot(log_transformed, dist=norm, plot=plt)

```

* Reasons to transform target variable include
    * Improve the results of a machine learning model when the target variable is skewed.
    * Reduce the impact of outliers in the target variable
    * Using the mean absolute error (MAE) for an algorithm that only minimizes the mean squared error (MSE)

* XGBoost can work with skewed target variables, but performance will likely be poor if skewness is extreme, especially for regression

* To check if distribution is skewed we can
    * Plot the distribution or a quantile-quantile plot
    * Use quantitative metrics like skewness, kurtosis
    * Use normality tests like Kolmogorov-Smirnov Test

```
### Plot distribution
plt.figure(figsize=(8,5))
sns.histplot(df[target], kde=True, bins=30)
plt.title(f"Distribution of {target}")
plt.show()

### PLot Q-Q plot
# Compares sample quantiles of your data against theoretical quantiles of a normal distribution.
probplot(log_transformed, dist=norm, plot=plt)


### Using skewness
from scipy.stats import skew
skewness_value = skew(df[target])

```

* probplot generates a probability plot, which should not be confused with a Q-Q or a P-P plot
    * ppplot : Probability-Probability plot Compares the sample and theoretical probabilities (percentiles)
    * qqplot :
    * probplot : 



* ANOVA assumes data is normally distributed


* Quantiles : Divides data into equal parts. Examples of quantiles are median, quartiles and percentiles

* Percentile - Rank relationship : Percentile = ((Rank - 1)/ (Sample Size -1)) x 100. For example for 11,12,31,41,51, the rank of 41 is 4, the percentile is (4-1)/(5-1) = 0.8

* Z-score : represents the number of standard deviations a data point is from the mean in a normal distribution. (z-score of mean is zero)

* Z-scores can be used for non-normal distributions because they are a way to standardize any dataset by measuring how many standard deviations a value is from the mean. But you cannot use a standard normal distribution table to find percentile ranks with a z-score from a non-normal distribution. This is where qq plot is helpful

* For non-normal distribution, it is hard to make any direct interpretation of the z-score

* QQ plot (Quantile-Quantile plot) : Used to see if you data matches a certain distribution

* KDE (Kernel Density Estimate) plot : Provides a smoothed, continuous representation of the underlying probability density function of a variable

* Interpreting skewness metric
    * Skewness ≈ 0 -> Approximately symmetric distribution
    * Skewness > 0.5 and < 1 -> Moderately Right-skewed (long tail to the right)
    * Skewness > 1 -> Heavily right skewed
    * Skewness > - 0.5 and < 0 -> Moderately Left-skewed (long tail to the left)

* Interpreting a Q-Q plot - refer 4 video by John Barosso
    * Heavy tails → Points bend away at the ends.
    * Skewness → Points curve systematically above/below the line.

* For categorical variable - if the data is highly imbalanced, then some feature engineering required such as 
    * Sampling techniques (oversampling, smote)
    *

* Most oversampling techniques like SMOTE are ineffective in improving Random Forest. An alternative approach is
    1. Tune model hyperparameters so the model performs well in terms of Average Precision (AP) (which is suitable for imbalanced classification problems)
    2. Set class_weight="balanced" to give more weight to minority class (aka defaulters)
    3. Cross-validation with StratifiedKFold
    4. Optimize the decision threshold by maximizing the F2 score (which weighs recall higher than precision, useful for detecting defaulters)

* Once you optimize thresholds and class weights, SMOTE often adds noise instead of signal and better avoided. Weighting the underrepresented class working better than oversampling methods. Even better, tune the sample weights.

* SMOTE is really only useful when minority-class points have dense local neighborhoods that don’t cross into majority territory, and where linear interpolation between neighbors makes sense in the feature space you’re working in. Higher dimensionality, non-linearities, gaps etc all make it result in worse performance. 

* How does stratified kfold work for imbalanced dataset?


* Feature-target correlation
    * If low : There is no strong linear relationship between that feature and the target. It does not mean the feature is useless for prediction, since non-linear relation may exist (quadratic, exponential)
    * If high : There is a linear relationship

* If no feature has high correlation with target -> **Linear regression may not explain much variance**. In simpler words, it means, Linear regresion model not able to capture relationship bw target and features well. Mathematically speaking, models R-squared value is low 

* R-squared : Measures amount of variation explained by (least squares) linear regression

* Feature-feature correlation : If two features are highly correlated, they carry almost the same information, and one of the features can be removed (removing multicollinearity) eg. height_cm and height_inch

* Outlier detection : Has different techniques such as
    * Grubbs test
    * Z-score method
    * IQR method
    * Winsorization
    * Dbscan
    * Isolation forest
    * Visualizing the data (using box plot)

* Winsorization Method / Percentile Capping is the better outlier detection technique amongst the above. In this approach, extreme values in a dataset replaced with less extreme ones

```
features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality']

### Plotting boxplot
for col in features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Outliers in {col}")
    plt.show()

features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality']

### Winsorization
for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df.loc[df[col] < lower, col] = lower
    df.loc[df[col] > upper, col] = upper
    # df[col] = df[col].clip(lower=lower, upper=upper)



```



### Model Building

* Cross-validation : A resampling technique in machine learning used to evaluate the performance of a model. Using train-test split, we can evaluate the model only once, while in this approach, the model can be evaluated multiple times using same data

* K Fold Cross Validation : A technique that divides a dataset into K equal-sized subsets (called "folds") for evaluating a model's performance. In each iteration, one fold serves as the test set while the remaining K-1 folds are used for training the model. After each iteration, evaluation score is retained and model is discarded

* OOF (Out of Fold) Predictions: In K fold CV, after training, the model makes predictions on the held-out validation fold. These predictions are called "out-of-fold" predictions (happens for each iteration)

* Understanding how OOF predictions are done : Consider 5 fold CV. During 1st iteration, folds 1 to 4 are train and fold 5 is test, so we get OOF prediction for only 20% of the dataset i.e. the rows corresponding to the 5th fold. In the second iteration, we get OOF prediction for another 20% of the dataset i.e. rows corresponding to 4th fold. Only after all 5 iterations do we get OOF prediction for all the rows

```
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import numpy as np
import xgboost as xgb

# Initialize OOF predictions
oof_preds = np.zeros(len(train))

# Set up KFold
kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

# Loop through folds

# train_idx = all rows part of train for this iteration
# val_idx = all rows part of validation for this iteration
for fold, (train_idx, val_idx) in enumerate(kf.split(train), 1):
    print(f"Training fold {fold}...")

    # Split data
    X_train, y_train = train.iloc[train_idx][FEATURES], train.iloc[train_idx][TARGET]
    X_val,   y_val   = train.iloc[val_idx][FEATURES], train.iloc[val_idx][TARGET]

    # Convert to DMatrix (optimized XGBoost format)
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval   = xgb.DMatrix(X_val,   label=y_val,   enable_categorical=True)

    # Train model
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=10_000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=100,
        verbose_eval=False
    )

    # Store out-of-fold predictions
    oof_preds[val_idx] = model.predict(
        dval, iteration_range=(0, model.best_iteration + 1)
    )

# Compute overall OOF RMSE
cv_score = root_mean_squared_error(train[TARGET], oof_preds)
print(f"OOF RMSE: {cv_score:.5f}")

```

* GridSearchCrossValidation : 

* In xgboost, the model’s predictive power comes from all trees up to that round. XGBoost builds models additively: Each boosting round add one new tree, and predictions are made by summing the outputs of all trees created upto that round

* Early stopping in XGBoost is a regularization technique designed to prevent overfitting and optimize training time. It works by monitoring the model's performance on a separate validation set during the training process and halting training when performance on this set stops improving for a specified number of rounds. For `model = xgb.train(..., num_boost_round=10_000, early_stopping_rounds=100)` if the best iteration was 1234 and no improvement happened for 100 rounds, XGBoost will actually stop at round 1334 — not at 1234. To overcome this we use iteration_range. `iteration_range=(0, model.best_iteration + 1)` which means: “Make predictions using all trees from the beginning up to and including the best iteration found during training.”

* Why early stopping is considered data leakage : Early stopping looks at the validation loss to decide how long to train (i.e., optimal number of boosting rounds). But in cross-validation, the validation set is supposed to represent unseen data. If we use it to tune training hyperparameters (like num_boost_round), then the validation set is no longer “purely unseen” → it influenced the training process. This is why it is called a data leak (mild one unless you begin using 100 or 1000 folds, then leak becomes more influencial)

* R² is not only for linear models — you can compute it for any regression model.

* Plot distribution of y_pred vs y_test. 

* We can use mean of target variable as prediction and compare your model to mean baseline. This is where R2 comes, R² measures how well your model explains the variation in the target variable compared to a simple baseline (just predicting the mean). This can easily be understood from the formula 
![R2 formula](r2_formula.png)



### Questions
1. Variance in statistics vs variance in machine learning? 
2. If bias is for a single value, how do we apply it to multiple data points?
3. How does stratified kfold work for imbalanced dataset?
4. Why does creation of a Q–Q plot in Excel need an adjustment by 0.5? (to make the distribution symmetrical)
5. In xgboost, why cant we use only the best iteration instead of using all the iterations from 0 till the best iteration (in the case of early stopping)?


### References
1. https://datasciencewithchris.com/transform-the-target-variable/
2. https://www.linkedin.com/posts/astrosica_we-made-it-92-recall-without-smote-resampling-activity-7368651053564719109-_a9f?utm_source=share&utm_medium=member_desktop&rcm=ACoAACH01PIBBVfmHJKmclThlgIPLNnlISIVIAA
3. https://www.reddit.com/r/learnmachinelearning/comments/1n0x1kp/advice_for_becoming_a_top_tier_mle/
4. https://www.youtube.com/watch?v=IgIN9-4-kSg
5. https://www.kaggle.com/code/nareshbhat/outlier-the-silent-killer
6. https://www.kaggle.com/code/cdeotte/analyze-original-dataset-from-kaggle-playgrounds#Playground-E5-S9:-Predicting-the-Beats-per-Minute-of-Songs
7. https://stats.stackexchange.com/questions/463870/eval-set-in-xgboost-and-validation-data