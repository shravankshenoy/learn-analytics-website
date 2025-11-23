
### Basics
* Feature

* Weights

* Model

* Types of machine learning problems for structured/tabular data:
    * Regression
    * Classification (2 class and multi-class)
    * Time series forecasting
    * Clustering
    * Anamoly detection

* Types of machine learning problems for text data
    * Text matching (fuzzy matching)

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

* A statonary process is mean reverting and shocks fade, whereas a non stationary process has persistent trend and shocks accumulate


### Machine Learning Development Life Cycle

* Difference stages in MLDLC include
    1. Requirement gathering
    2. Data source analysis 
    3. Annotation
    4. Data pipeline development
    5. Model development (including feature engineering, model building)
    6. Model testing (including sensitivity analysis, identifying evaluation metrics)
    7. Model deployment 
    8. Model monitoring


1. Define requirements
- What are we trying to optimize?
- What are latency requirements?
- Do we need an ML model? Or can we go for other approaches?

2. Analyze data sources
-  What kind of data do we have? (tabular, time series, supervised or unsupervised)
-  Do we have the right kind of data for building an ML model? If so how do we formulate the problem statement?
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


* Different stages of a time series forecasting problem
    1. Requirement gathering 
        * Identify the target column (especially if multiple date columns)
        * Univariate or multivariate time series problem
        * If other dimensions present, which dimensions we want to aggregate at (eg. take single time series, or one time series for each line of business and forecast each of the time series separately)
        * How many day forecast
        * Frequency of forecast (daily, weekly, monthly)
        * Prepare data into right format (datetime index)
    

    2. Exploratory time series analysis
        * Plot the data at different frequencis
        * STL decomposition (decompose into level, trend, seasonality, residuals)
        * ACF and PACF plots (with proper interpretation)
        * Stationary check (using Augmented Dickey-Fuller(ADF) or plotting rolling statistics like mean/stddev)
        * Fourier analysis

    3. Data Preprocessing
        * Outlier detection and removal
        * Making data stationary (using differencing/seasonal differencing, transforms like log)
        * Missing value treatment (removal, imputation)

    4. Feature Engineering
    
    5. Modeling
        * Build a baseline model using Exponential smoothing or simple ARIMA/SARIMA
        * For more complex series, convert to regression problem


### Feature Engineering

* Some important questions to answer in a dataset are
    * Target Variable Identification
    * Do the features contain information that helps predict the target?

* Feature engineering for a time series
    * lag variables
    * 

```
df['lag_1'] = df['value'].shift(1)

```

* Detecting signal in target : When we say the target has "signal", we usually mean - There’s a systematic, predictable relationship between the features (X) and the target (y), not just random noise.

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
probplot(log_transformed, dist='norm', plot=plt)

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


* Handling missing values : Missing values can be in original features as well as in derived values. Everything must be handled in separate ways

```
null_count = train_df.isnull().sum()

```

* Is feature engineering required? : Feature Engineering involves 3 things
    * Feature selection
    * Feature extraction
    * Adding features through domain expertise
Xgboost only does feature selection. The other 2 has to be done by us (only a deep learning model could replace feature extraction for you) Xgboost would have hard time on picking relations such as a*b, a/b and a+b for features a and b (refer 10)

* Tree-Based Models Learn by Splitting, Not Algebra. Hence they can learn additive relationship easily but harder time with multiplicate and division. And even harder time with non-linear transforms

* Ways to create new features for numerical columns include
    * Multiplication/Interaction features 
    * Non-linear/Polynomial like log, square 
    * Division/Pairwise Ratios (safe division)
    * Binning

```
df_new['Rhythm_Energy'] = df_new['RhythmScore'] * df_new['Energy']

df_new['Energy_Squared'] = df_new['Energy'] ** 2` 
df_new['Log_Duration'] = np.log1p(df_new['TrackDurationMs'])

df_new['Acoustic_Instrumental_Ratio'] = df_new['AcousticQuality'] / (df_new['InstrumentalScore'] + 0.01)

df["DurationBin"] = pd.qcut(df["TrackDurationMin"], q=10, duplicates='drop').cat.codes


```

* Interaction features are new features created by combining two or more existing features, usually to allow the model to capture relationships that aren't obvious when features are considered independently. Ways to create interaction features include
    * Manual numeric-numeric interaction (multiply/divide/binning)
    * Automatic numeric-numeric interaction (using sklearn.preprocessing.PolynomialFeatures)
    * Categorical-categorical interactions (combine 2 categorical columns into a compound column)
    * Categorical-numerical interaction (label/one-hot encode cat column, then multiply with numerical column)

```

df = pd.DataFrame({
    'road_type': ['urban', 'highway', 'urban', 'rural'],
    'lighting': ['daylight', 'night', 'night', 'dim']
})

df['road_lighting_interaction'] = df['road_type'] + '_' + df['lighting']
# You can then Label Encode or One Hot Encode this interaction column for modeling.

```

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

* Feature engineering for categorical and boolean features
    * Categorical : Label encoding, target encoding, one-hot encoding
    * Boolean : Convert to int

```
for col in BOOLEAN_FEATURES:
    train_df[col] = train_df[col].astype(int)
    test_df[col] = test_df[col].astype(int)

for col in CATEGORICAL_FEATURES:    
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])


```

* As a best practice, transform both train and test dataframes at the same place to avoid mistakes like 
    1. Not applying train dataframe transformation to test dataframe
    2. Use fit_transform for the test dataframe


* Time series is made up of 3 components
    1. Trend : Long term direction of data
    2. Seasonality : Pattern that repeat at fixed interval (week, month, year)
    3. Cyclic variations : Longer-term fluctuations that are not of a fixed period (eg. recession cycle)


* Below are code snippets to prepare data for time series analysis

```
### Refer link 12 i.e. Kaggle link for dataset. Below snippets are from Trend notebook
### Approach 1 (prepare dataset for time series)
# parse_dates instructs Pandas to interpret specified columns as datetime objects during the data loading process
# to_period has to be applied on column with datetime type
retail_sales = pd.read_csv(
    "us-retail-sales.csv",
    parse_dates=['Month'],
    index_col='Month'
).to_period('D')


### Approach 2 (prepare dataset for time series)
retail_sales = pd.read_csv(
    "us-retail-sales.csv",
    parse_dates=['Month']
)
retail_sales = retail_sales.set_index('Month').to_period('M')

food_sales = retail_sales[:, 'FoodAndBevarage']

### Approach 3 (create your own data)
dates = pd.date_range(start='2025-01-01', periods=20, freq='D')
time_series_df = pd.DataFrame(
    {
        'date':'dates',
        'value': 10 + np.arange(1,21)
    }
)
time_series_df = time_series_df.set_index('date')
```


* Below are code snippets to plot trend

```
# takes mean of 12 points, since center=True, mean is computed by taking 6 points to left and 5 points to right and that point. Also if number of observations are less than 6, then it gives NA 
# ser = pd.Series(range(1, 100)) # mock series
# ser.rolling(window=4).mean()
# ser.rolling(window=5, center=True).mean()

# Computing trend
trend = food_sales.rolling(
    window=12,
    center=True,
    min_periods=6
).mean()

# Plot trend against time series
ax = food_sales.plot(alpha=0.5)
ax = trend.plot(ax=ax, linewidth=3)

```
* Rolling average/moving average is used to find the trend of a time series because it helps smooth out short-term fluctuations and thus highlights the underlying long-term pattern. On averaging nearby points, sudden spikes and dips are softened and random noise average out

* Mathematically, rolling average is a low-pass filter (i.e. keeps low frequncy component i.e trend and remove high frequency component i.e. seasonality)

* Stationarity : A time series is stationary if it has no long term trend or seasonality. In mathematical terms:
    * Constant mean through time
    * Constant variance through time
 Can be checked via Augmented Dickey-Fuller (ADF) test

* Stationarity matters because if time series is not stationary, every data point has its own variance, which means each data point belongs to a different distribution. This makes it hard to build model, because model assumes some consistent underlying distribution

* Common type of aggregations include count, distinct count, minimum, max, average, std dev, ratio

* Autocorrelation means how much current values depend on past values. Correlation of a variable with itself at a different point

$$ \rho_k = \frac{\sum_{t=k+1}^{n} (Y_t - \bar{Y})(Y_{t-k} - \bar{Y})}{\sum_{t=1}^{n} (Y_t - \bar{Y})^2} 


$$

Y_t is the observation at time t.
$\bar{Y}$ is the sample mean of the time series.

* Complex Autocorrelation Structure: When the dependence is spread across many lags and possibly
seasonal lags.

* What an ACF plot can reveal:
    * If all the bars are within the confidence interval (except for lag 0) -> the series is likely random and has no autocorrelation. 
    * A slow decay or a steep linear decay in the ACF plot -> indicates a trend (this is because the correlation remains high for many lags and decreases slowly)
    * Wave like pattern with significant spikes at regular intervals (e.g., every 12 months for monthly data) -> indicates a seasonal pattern
    * An ACF plot that decays quickly to zero suggests a stationary time series (A slow decay can indicate non-stationarity
    
* Statistical Significance in ACF: The shaded region on the plot is a confidence band. Bars extending beyond this band represent statistically significant correlations, meaning the correlation is unlikely to be due to random chance.

* Partial autocorrelation - Correlation of a time series with delayed copy itself (effect of a lag after removing the effect of the intermediate lags (for example, effect of lag 3 after
removing the effects of lag 1 and 2)) 

* If a PACF shows a significant spike at lag 3, it means $Y(t)$ is significantly correlated with $Y(t-3)$ even after accounting the effects of lag 1

* Cut off point: Point where plot drops to or stays within confidence levels, meaning higher lags do not have significant contribution. A slowly decaying PACF plot may indicate presence of seasonality in data. If PACF has no clear cut off, it means AR might not be a good model, or additional preprocessing required. 

* Time series can have AR or MA signatures (refer 13): 
    * An AR signature corresponds to a PACF plot displaying a sharp cut-off and a more slowly decaying ACF
    * An MA signature corresponds to an ACF plot displaying a sharp cut-off and a PACF
plot that decays more slowly.

* For a time series with a linear trend:
    * The ACF will have high positive autocorrelation at lag 1 and decay slowly as the lag increases. This slow decay happens because the trend creates persistent correlation across time.
    * PACF will typically show a very strong spike at lag 1, a smaller spike at lag 2 and drops of very quickly. This is because once you account for the first lag, the additional lags don't add much explanatory power for a linear trend.

* How to distinguish trend vs seasonality using ACF : For trend only, ACF decreases gradually and smoothly without a clear repeating pattern. For seasonality, ACF shows a wave-like pattern with significant spikes at seasonal lags (e.g lag 12 for monthly data with yearly seasonality)

* For an AR(p) model:
    * PACF cuts off after lag p (partial autocorrelations become zero beyond p)
    * ACF tails off gradually.

* For an MA(q) model:
    * ACF cuts off after lag q (autocorrelations become zero beyond q)
    * PACF tails off gradually.

* If ACF cuts off and PACF tails -> likely MA model. If PACF cuts off and ACF tails -> likely AR model. If neither shows a clear cutoff -> consider ARMA or ARIMA.

* Since PACF helps assess how many lags contribute directly to series, it is helpful in identifying order of Autoregressive model.

* Time series based features
    1. Lag and lead features
    2. Window features
    3. Ratio of current to window features such as 
        * Ratio of observed day activity to last 8/15 days average activity for that customer
        * Ratio of observed day activity (minus mean) to last 8/15 days std deviation
        * Ratio of observed day activity to last 8/15 days minimum/maximum
    4. Derived feature from lead lag features (min-max ratio)
        
        
* The intuition behind `Ratio of observed day activity to last 8/15 days average activity` is the mean gives the baseline, what is normal for that customer, and we compare current behaviour against baseline behaviour
    * Ratio > 1 : sudden spike in activity which could imply fraud
    * Ratio < 1 : sudden drop in activity which could indicate churn, disengagement

* The intuition behind `Ratio of observed day activity to last 8/15 days minimum/maximum` If current activity greater than historical max or less than historical min, then abnormal behaviour







### Model Building

* Ordinary Least Squares : https://www.youtube.com/shorts/TrAFh3Onf3E

* To use XGBoost for a time series with a trend, you should always pre-process the data to remove the trend, allowing XGBoost to focus on modeling the non-linear, high-frequency components (trend is a low frequency component) (refer next point for reason)

* Tree-based models like XGBoost, Random Forest, and Decision Trees generally cannot extrapolate beyond the range of training data because of how they work (and hence called **interpolation model**). Prediction is based on splits, not equations.
    * Trees partition the feature space into regions based on observed values.
    * Each leaf node stores the average target value of training samples in that region.
    * If a new input falls outside the training range, the tree still assigns it to the closest existing leaf - prediction stays within the range of seen values.
If your training data for Value ranges from 10 to 100, and you ask the model to predict for a scenario where lag features suggest a value of 150, the model will likely predict
something close to 100 (the max seen in training), not 150.

* To handle time series with rising trend 
    * Use models that assume a functional form (Linear Regression, Polynomial Regression, ARIMA / SARIMA) 
    * Detrend the time series (make time series stationary)

* Cross-validation : A resampling technique in machine learning used to evaluate the performance of a model. Using train-test split, we can evaluate the model only once, while in this approach, the model can be evaluated multiple times using same data

* K Fold Cross Validation : A technique that divides a dataset into K equal-sized subsets (called "folds") for evaluating a model's performance. In each iteration, one fold serves as the test set while the remaining K-1 folds are used for training the model. After each iteration, evaluation score is retrained and model is discarded

* Another advantage of k fold cv verses train-test split is that we can get prediction on entire train data instead of a small portion of train data.

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

* GridSearchCrossValidation : A method for hyperparameter tuning. Tries all possible combination of hyperparameter values, and for each combination it uses cross validation strategy (like k-fold) to evaluate the performance of that combination
```

param_dist = {
    'max_depth': [3, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'min_child_weight': [3, 5],
    'n_estimators': [200, 400],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0],
}

xgb = XGBRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=xgb, 
    param_grid=param_dist, 
    cv=3, 
    n_jobs=-1
)

```

* GridSearchCV can be computationally very expensive when there are a lot of hyperparamters. The alternatives for lesser computational load are
    * RandomizedSearchCV
    * Bayesian Optimization using a library like Optuna or HyperOpt


```
### Using Optuna for hyperparamter tuning of Xgboost regressor
N_SPLITS = 7
kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

def objective(trial):
    params = {
        'n_estimators' : trial.suggest_int('n_estimators', 3, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }

    oof = np.zeros(len(train_df))

    for train_idx, val_idx in kfold.split(X):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            early_stopping_rounds=200,
            verbose=False
        )
        oof[val_idx] = model.predict(X_fold_val)

    rmse = np.sqrt(mean_squared_error(y_train, oof))
    return rmse

# ==========================
# Run Optuna study
# ==========================
print("🔍 Running Optuna optimization...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)

print("✅ Best trial:")
print(study.best_trial.params)
best_params = study.best_trial.params

```

* Refit on full training : This refers to the practice of retraining a machine learning model on the entire available dataset after the model's architecture and hyperparameters have been finalized through processes like cross-validation or a train-validation-test split.

* An alternative to “refit on full” is to increase the number of folds. When we use 5, 10, 20 KFold, each model is trained with 80%, 90%, 95% data respectively. (So “refit on full” is like having 100+ folds)

* In xgboost, the model’s predictive power comes from all trees up to that round. XGBoost builds models additively: Each boosting round add one new tree, and predictions are made by summing the outputs of all trees created upto that round

* For xgboost, when running experiments, it is often helpful to use a larger learning rate like LR=0.3 or LR=0.1. This lets us perform faster experiments. After we find GBDT models to include in our ensemble, we can usually boost their CV and LB a little more by decreasing the learning rate and training them longer. We can train the same models with LR=0.01 or LR=0.005

* Early stopping in XGBoost is a regularization technique designed to prevent overfitting and optimize training time. It works by monitoring the model's performance on a separate validation set during the training process and halting training when performance on this set stops improving for a specified number of rounds. For `model = xgb.train(..., num_boost_round=10_000, early_stopping_rounds=100)` if the best iteration was 1234 and no improvement happened for 100 rounds, XGBoost will actually stop at round 1334 — not at 1234. To overcome this we use iteration_range. `iteration_range=(0, model.best_iteration + 1)` which means: “Make predictions using all trees from the beginning up to and including the best iteration found during training.”

* Purpose of eval_set in Xgboost
    * Help to implement early stopping. When combined with the early_stopping_rounds parameter, XGBoost will automatically stop training if the performance on the specified evaluation set does not improve for a certain number of consecutive rounds. This prevents overfitting
    * Observe how the model's performance metrics (e.g., RMSE for regression) evolve over boosting rounds

* Why early stopping is considered data leakage : Early stopping looks at the validation loss to decide how long to train (i.e., optimal number of boosting rounds). But in cross-validation, the validation set is supposed to represent unseen data. If we use it to tune training hyperparameters (like num_boost_round), then the validation set is no longer “purely unseen” → it influenced the training process. This is why it is called a data leak (mild one unless you begin using 100 or 1000 folds, then leak becomes more influencial)

* Using early stopping + refit with full : Lets say we train 7 KFold above with early stopping, and optimal number of iterations for each KFold are 2700, 2232, 2652, 2000, 2327, 2288, 1842 respectively. The average is 2292. Assuming early_stopping_rounds=200, average is actually 2092. When training with 100% train data (i.e. refit on full), we need to use K/(K-1) more iterations. So we use 7/6 * 2092 = 2440. We will now train with 100% train data using fixed 2440 iterations.


* R² is not only for linear models — you can compute it for any regression model.

* R2 value tell about amount of variability in target explained by model

* We can use mean of target variable in training data as prediction and compare your model to mean baseline. This is where R2 comes, R² measures how well your model explains the variation in the target variable compared to a simple baseline (just predicting the mean). This can easily be understood from the formula 
![R2 formula](r2_formula.png)

* A low R2 value does not necessarily imply a good model. For example in the Predicting the Beats-per-Minute of Songs challenge, target values had a normal distribution with a mean ~120.Predictions have the same general distribution, just a lot narrower. The residuals may not be large because the range of target values is small to begin with, but that doesn't mean that anyone has a good model even if R2 is low. Thus a good R2 in low variance target does not mean much

* Scenarios where a high R2 does not mean much
    * Low variance targets
    * Overfitting (R2 does not tell if model fits well out of sample)
    

```
# Low variance target example
y_true = [1000, 1010, 990, 1005, 995]
y_pred = [1000, 1000, 1000, 1000, 1000]
R² = 1 - 250/2500 = 0.90. High R2 but the model is useless

```

* R2 cannot check if there is enough signal in the target

* Suppose we generate synthetic data using a known function f(x) plus random noise. We then train a regression model that is exactly f(x), and get an R² ≈ 0.92. Then no trained model can get an R2 better than 0.92. This is because we injected Gaussian noise, so out of total variance in base signal, 8% of the variance is due to noise, which no model can explain (unless model overfits on noise). If we substitute formula it is R2 = 1 - (Noise variance/Total variance)

* In stacking (stacked ensembling), instead of training the second-level model (the meta-model) on the original input features (e.g., age, salary, pixels…), we train it on the predictions of the first-level models (the base learners). These predictions of first-level models become new features for the second-level model, hence the name meta-features. Meta-features = features generated by models, not by the original data. 

* Approach to do stacking:
    1. Train a model (say xgboost) on folds of train data and predict on the out-of-fold to get model's unbiased prediction on training data (the shape of this prediction will be (train_num_rows, 1))
    2. Repeat step 1 using other models (say catboost, random forest, lgbm). For each model you will get a prediction on training data of shape (train_num_rows, 1)
    3. Concatenate all of these predictions, you will get a matrix of shape (train_num_rows, 4) (assuming we have trained using 4 different models). These prediction of model will serve as features for the meta-model
    4. Train a lasso model with the above predictions/metafeatures matrix as input and the y_train of training data as output. This meta-model will decide how much weightage to be given to each of the base models


* An alternate stacking approach is when we train model using cross validation approach, we store each of the models. So if we have 5 folds, we will have 5 xgboost models (each xgboost model trained on 4 of the 5 folds), 5 catboost models, 5 lgbm, etc. We then use each of these models to predict on entire training data and then take average of that for each model type. For example we will have 5 xgboost predictions of shape (train_num_rows, 5), which we then average to get a prediction of shape (train_num_rows, 1). Similarly for catboost, lgbm etc. We then concatenate the prediction of different model types to get the final meta-features matrix, and train a meta model like lasso on that. The below code does the same (from reference 9)

```
def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train a LightGBM model"""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_SEED,
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=callbacks
    )
    
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train an XGBoost model"""
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 1000,
        'random_state': RANDOM_SEED,
        'verbosity': 0
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=False
    )
    
    return model

def train_catboost(X_train, y_train, X_val, y_val):
    """Train a CatBoost model"""
    params = {
        'loss_function': 'RMSE',
        'learning_rate': 0.05,
        'depth': 6,
        'iterations': 1000,
        'random_seed': RANDOM_SEED,
        'l2_leaf_reg': 3,
        'bootstrap_type': 'Bayesian',
        'verbose': False
    }
    
    model = cb.CatBoost(params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=False
    )
    
    return model

# Train and evaluate models across folds
for fold_idx, (train_idx, val_idx) in enumerate(folds):
    print(f"\nTraining fold {fold_idx + 1}/{n_folds}")
    
    # Split data for this fold
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train models
    print("Training LightGBM...")
    lgb_model = train_lightgbm(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
    
    print("Training XGBoost...")
    xgb_model = train_xgboost(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
    
    print("Training CatBoost...")
    cb_model = train_catboost(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
    
    # Random Forest as an additional diverse model
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    rf_model.fit(X_fold_train, y_fold_train)
    
    # Make predictions on validation fold
    lgb_preds = lgb_model.predict(X_fold_val)
    xgb_preds = xgb_model.predict(X_fold_val)
    cb_preds = cb_model.predict(X_fold_val)
    rf_preds = rf_model.predict(X_fold_val)
    
    # Create a weighted average of predictions
    # We give higher weights to models that generally perform better
    blend_preds = 0.35 * lgb_preds + 0.35 * xgb_preds + 0.2 * cb_preds + 0.1 * rf_preds
    
    # Store out-of-fold predictions
    oof_predictions[val_idx] = blend_preds
    
    # Make predictions on test set
    lgb_test_preds = lgb_model.predict(X_test)
    xgb_test_preds = xgb_model.predict(X_test)
    cb_test_preds = cb_model.predict(X_test)
    rf_test_preds = rf_model.predict(X_test)
    
    # Average test predictions from this fold
    fold_test_preds = 0.35 * lgb_test_preds + 0.35 * xgb_test_preds + 0.2 * cb_test_preds + 0.1 * rf_test_preds
    test_predictions += fold_test_preds / n_folds
    
    # Calculate and display fold metrics
    lgb_rmse = rmse(y_fold_val, lgb_preds)
    xgb_rmse = rmse(y_fold_val, xgb_preds)
    cb_rmse = rmse(y_fold_val, cb_preds)
    rf_rmse = rmse(y_fold_val, rf_preds)
    blend_rmse = rmse(y_fold_val, blend_preds)
    
    print(f"Fold {fold_idx + 1} Results:")
    print(f"LightGBM RMSE: {lgb_rmse:.5f}")
    print(f"XGBoost RMSE: {xgb_rmse:.5f}")
    print(f"CatBoost RMSE: {cb_rmse:.5f}")
    print(f"Random Forest RMSE: {rf_rmse:.5f}")
    print(f"Blended RMSE: {blend_rmse:.5f}")
    
    # Store models for this fold
    models.append({
        'fold': fold_idx,
        'lgb_model': lgb_model,
        'xgb_model': xgb_model,
        'cb_model': cb_model,
        'rf_model': rf_model
    })

# Calculate overall cross-validation score
cv_score = rmse(y_train, oof_predictions)
print(f"\nOverall CV RMSE: {cv_score:.5f}")


# Filter models if any failed to train
valid_models = []
for model in models:
    if all(m is not None for m in [model['lgb_model'], model['xgb_model'], model['cb_model'], model['rf_model']]):
        valid_models.append(model)

if len(valid_models) > 0:
    # Create meta-features for stacking
    X_meta_train = np.column_stack([
        np.array([model['lgb_model'].predict(X_train) for model in valid_models]).mean(axis=0),
        np.array([model['xgb_model'].predict(X_train) for model in valid_models]).mean(axis=0),
        np.array([model['cb_model'].predict(X_train) for model in valid_models]).mean(axis=0),
        np.array([model['rf_model'].predict(X_train) for model in valid_models]).mean(axis=0)
    ])

    # Create meta-features for test set
    X_meta_test = np.column_stack([
        np.array([model['lgb_model'].predict(X_test) for model in valid_models]).mean(axis=0),
        np.array([model['xgb_model'].predict(X_test) for model in valid_models]).mean(axis=0),
        np.array([model['cb_model'].predict(X_test) for model in valid_models]).mean(axis=0),
        np.array([model['rf_model'].predict(X_test) for model in valid_models]).mean(axis=0)
    ])

    # Train a Ridge meta-model
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_meta_train, y_train)

    # Make final predictions
    stacking_predictions = meta_model.predict(X_meta_test)

    # Analyze the performance of the stacking model
    stacking_oof_preds = meta_model.predict(X_meta_train)
    stacking_cv_score = rmse(y_train, stacking_oof_preds)
    print(f"Stacking Ensemble CV RMSE: {stacking_cv_score:.5f}")

    # Compare with the simple average approach
    print(f"Simple Average Ensemble CV RMSE: {cv_score:.5f}")

    # Select the better performing approach for final predictions
    if stacking_cv_score < cv_score:
        print("Using stacking ensemble for final predictions")
        final_predictions = stacking_predictions
    else:
        print("Using simple average ensemble for final predictions")
        final_predictions = test_predictions
else:
    print("Not enough valid models for stacking. Using simple average ensemble.")
    final_predictions = test_predictions


```

* One advatage of k-fold cross validation over splitting data into train and eval is in train-eval split, we only predict on the eval set, but in k-fold, we get predict for the entire training data. For example if we do a 80-20 split, we train on 80% of data and predict on the 20% eval dataset. Instead if we do a 5-fold cross validation, we have 5 iterations, and in each iteration, we predict on a different 20% of the data, so we end up predicting on the entire data and can then compare that against y_train

* Applying bounds to prediction : If the prediction of model is too low or too high, we replace it with some fixed value
```
min_bpm = max(60, y_train.min())  # Most songs have at least 60 BPM
max_bpm = min(200, y_train.max())  # Most songs have at most 200 BPM
bounded_predictions = np.clip(final_predictions, min_bpm, max_bpm)

```

* Some ways to compare prediction values with actual values are
    * Plot distribution of y_pred vs y_test
    * Scatter plot between y_pred and y_test
    * Residual plot of y_pred vs y_residual (y_residual = y_test-y_pred)
Below is code for all of the above (from reference 9)
```
# Distribution plot
plt.figure(figsize=(12, 6))
plt.hist(y_train, alpha=0.5, label='Actual BPM', bins=50)
plt.hist(oof_predictions, alpha=0.5, label='Predicted BPM', bins=50)

# Scatter plot
plt.scatter(y_train, oof_predictions, alpha=0.3, s=10)

# Residual plot
residuals = y_train - oof_predictions
plt.figure(figsize=(12, 6))
plt.scatter(oof_predictions, residuals, alpha=0.3, s=10)

```

* Why automated machine learning is not easy : Lets say you ask a non-DS to go pull customer data and predict likelihood of churn. 
    * What data do they need? 
    * Which SQL tables do they need to query?
    * What are the relevant features? 
    * Is it a time series problem? 
    * How about engineering features based on historical behaviour? 
    * How do you identify the target column? 
    * Now let's say they manage all of that, but the model scores 40% accuracy, now what?

* Steps to use xgboost on GPU
    1. Set TREE_METHOD = 'gpu_hist' in xgboost parameters
    2. Set environment to use GPU (GPU T4 x2 or GPU P100 on Kaggle)


* GPU T4 x2 gives you an option to parallelize your work while the p100 does not provide this option. "GPU T4 x2" refers to a computing environment with two NVIDIA Tesla T4 GPUs. The T4 GPU is based on NVIDIA's Turing architecture, which features Tensor Cores for accelerating AI tasks

* Time series models in order of complexity
    * ExponentialSmoothing models (Holt-Winter model)
    * ARIMA/SARIMA models
    * Using ml regression techniques like xgboost

* Exponential smoothing models at their core give more weightage to recent observations and less on historical observations. 


$$ \hat{y}_{t+1} = \alpha y_t + (1 - \alpha)\hat{y}_t $$

$$ \alpha = smoothing parameter $$

* Called exponential because older observations have exponentially smaller influence (say alpha = 0.8 then $ \alpha * (1-\alpha) = 0.2 * 0.8 = 0.16 $ and $ (1-\alpha) * (1-\alpha) = 0.2 * 0.2 = 0.04 $, cube is 0.008)

$$ {y}_{4} = \alpha {y}_{3} + (1-\alpha) \hat{y}_{3}  \\
           = \alpha {y}_{3} + (1-\alpha) (\alpha {y}_{2} + (1-\alpha) \hat{y}_{2})  $$

* Time series is made up of following components
    * Level : Average value of time series, a constant value
    * Trend :  Direction/Slope of time series
    * Seasonality 
    * Cyclicity
    * Residuals

* Exponential Smoothing models are of 3 types
    * Simple Exponential model : Takes into account only level, does not take trend or seasonality into account
    * Double Exponential Smoothing model (Holt's Linear Trend model) : Takes level and trend into account
    * Triple Exponential Smoothing model (Holt Winter model) : Considers level, trend and seasonality

Simple Exponential Smoothing
$$

\hat{y}_{t} = {l}_{t} = \alpha * {y}_{t} + (1-\alpha) * {l}_{t-1}

$$


Double Exponential Smoothing (b stands for trend)
$$

\hat{y}_{t} = {l}_{t} + h * {b}_{t} \\

{l}_{t} = \alpha * {y}_{t} + (1-\alpha) * ({l}_{t-1} + {b}_{t-1})

$$

Holt Winter
```
level_t = α * y_t + (1 − α) * (level_{t−1} + trend_{t−1})
trend_t = β * (level_t − level_{t−1}) + (1 − β) * trend_{t−1}
season_t = γ * (y_t − level_t) + (1 − γ) * season_{t−s})
```

* Autoregressive (AR) Model : Current value of variable  expressed as linear combination of its past values plus error. AR models capture temporal dependence in data

* Moving Average (MA) Model : Uses residuals for forecasting. Instead of using previous observations, we forecast using past errors instead. Intution is, sometimes series does not strongly depend on its past values, but rather on unexpected changes that occured

* MA model is NOT a rolling mean model which is also called moving average model

* ARIMA has 3 parameters:
    * p : order of autoregressor, can be deduced using PACF
    * d : order of differencing, can be deduced using Augmented Dickey-Fuller (ADF) test
    * q : order of moving average, can be deduced using autocorrelation

* Another approach to determine p,d,q is simply by iterating over all the possible combinations and choose a model with best score against a metric such as AIC(Akaike's Information Criterion) or BIC (Bayesian Information Criterion)

* SARIMA has 6 parameters, 3 for trend, 3 for seasonal

* An AR model is probabilistic because of the random noise term. This random shock makes the model stochastic, not deterministic.

* Holt Winter vs SARIMA
    * Deterministic vs Probabilistic : Holt Winter is just a weighted average of past values, it just smoothes the past.
    * Prediction vs Prediction Interval

* Point Forecast vs Multistep forecast:
    * Point forecast : predicts a single future value
    * Multistep forecast : Predicts a series of future values (next 60 days). Can be achieved through direct forecasting (a separate model for each step) or recursive forecasting (using prevoius forecast as input for next one). More prone to error accumulation

* Forecast horizon : How many days/time steps into future we are predicting - https://stats.stackexchange.com/questions/586244/what-does-it-mean-forecast-horizon-in-time-series-forecasting

Creating horizon indicator - Instead of training 30 separate models, you can add a horizon feature to indicate which step ahead you are predicting. This allows a single model to learn pattrn for different horizons

Hence for each time point, we create 30 rows - one for each horizon



### Model testing

* Sensitivity analysis : How much model output changes when we slightly change the input paramters

* Match ratio : How many predictions match when we change the input parameter (for example one feature all zeros). Higher the match ratio, more robust the model

```
## Sensitivity analysis : Match ratio per feature
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get baseline predictions
y_pred_original = model.predict(X_test)

# Create perturbed version of test data
X_test_perturbed = X_test.copy()
X_test_perturbed[:, 0] = 0

# Get new predictions on perturbed data
y_pred_perturbed = model.predict(X_test_perturbed)

# Compute Match Ratio
matches = np.sum(y_pred_original == y_pred_perturbed)
total = len(y_pred_original)
match_ratio = matches / total

print(f"Match Ratio: {match_ratio:.3f}")
print(f"Number of predictions unchanged: {matches}/{total}")


```

* Interpretability : Can we understand the reasoning behind model's decisions

* Partial dependence plot : Used to explain how model predictions is impacted by a feature. A flat pdp low dependence on feature, a sharp rise or fall indicates high dependence

* Steps to create a 1-way partial dependence plot:
    1. Select a feature you want to analyze
    2. Get list of all distinct values of that feature
    3. Copy training data
    4. Replace feature column with fixed value (create multiple copies of training data, such that each copy has only 1 value from step 2 for the selected feature)
    5. Predict outcomes and take mean prediction (regression)/mean class probability(classification)

```


# Load dataset
data = load_iris()
X, y = data.data, data.target
feature_names = data.feature_names

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Choose petal length feature
feature_index = 2 

x_feature = X_train[:, feature_index]

feature_distinct_values = x_feature.distinct()
grid = np.linspace(x_feature.min(), x_feature.max(), num=50)  # 50 evenly spaced values


pdp_values = []
# use either grid or feature_distinct_values
for v in feature_distinct_values:
    X_temp = X_train.copy()
    X_temp[:, feature_index] = v
    preds = model.predict_proba(X_temp)[:, 0]
    pdp_values.append(preds.mean())

pdp_values = np.array(pdp_values)


# Plot the pdp charts
plt.figure(figsize=(6,4))
plt.plot(feature_distinct_values, pdp_values, linewidth=2) # use either grid or feature_distinct_values
plt.xlabel(feature_names[feature_index])
plt.ylabel("Average Predicted Probability (class 0)")
plt.title("Partial Dependence Plot (built from scratch)")
plt.grid(True)
plt.show()


```

* Time-series aware cross-validation: Unlike standard cross-validation (which randomly splits data), time series CV respects temporal order. This means future data is never
used to predict the past, avoiding data leakage (includes concepts like expanding window, validation gap)
    
* Expanding window: Start with an initial training set. For each fold, add more recent data to the training set while moving the validation window forward. Example : 
    Fold 1: Train on Day 1-260 -> Validate after gap 
    Fold 2: Train on Day 1-320 -> Validate after gap 

* Validation Gap: Gap between training data and validation data. Prevents look-ahead bias, because features like lag variables could leak future info if validation starts immediately after training. Example : 60-day validation gap implies 60 days between last row of training data and first row to validation data.


### Questions
1. Variance in statistics vs variance in machine learning? 
2. If bias is for a single value, how do we apply it to multiple data points?
3. How does stratified kfold work for imbalanced dataset?
4. Why does creation of a Q–Q plot in Excel need an adjustment by 0.5? (to make the distribution symmetrical)
5. In xgboost, why cant we use only the best iteration instead of using all the iterations from 0 till the best iteration (in the case of early stopping)?
6. Can tree based methods pick relations such as a*b, a/b,a+b? Do we need to do feature engineering with models like xgboost?

### References
1. https://datasciencewithchris.com/transform-the-target-variable/
2. https://www.linkedin.com/posts/astrosica_we-made-it-92-recall-without-smote-resampling-activity-7368651053564719109-_a9f?utm_source=share&utm_medium=member_desktop&rcm=ACoAACH01PIBBVfmHJKmclThlgIPLNnlISIVIAA
3. https://www.reddit.com/r/learnmachinelearning/comments/1n0x1kp/advice_for_becoming_a_top_tier_mle/
4. https://www.youtube.com/watch?v=IgIN9-4-kSg
5. https://www.kaggle.com/code/nareshbhat/outlier-the-silent-killer
6. https://www.kaggle.com/code/cdeotte/analyze-original-dataset-from-kaggle-playgrounds#Playground-E5-S9:-Predicting-the-Beats-per-Minute-of-Songs
7. https://stats.stackexchange.com/questions/463870/eval-set-in-xgboost-and-validation-data
8. https://www.reddit.com/r/datascience/comments/1054dl3/why_hasnt_automl_been_more_widely_adopted_by/
9. https://www.kaggle.com/code/adilshamim8/predicting-the-beats-per-minute-of-songs-101
10. https://datascience.stackexchange.com/questions/17710/is-feature-engineering-still-useful-when-using-xgboost
11. https://www.youtube.com/playlist?list=PLKmQjl_R9bYd32uHImJxQSFZU5LPuXfQe (Time Series - Egor Howell)
12. https://www.kaggle.com/competitions/store-sales-time-series-forecasting/code?competitionId=29781&sortBy=voteCount&excludeNonAccessedDatasources=true
13. https://stats.stackexchange.com/questions/28166/how-does-acf-pacf-identify-the-order-of-mo-and-ar-terms 

### To Explore
1. https://medium.com/@anagha.srivasa/nvidia-t4-x2-v-s-p100-gpu-when-to-choose-which-one-87cf1c55f386
