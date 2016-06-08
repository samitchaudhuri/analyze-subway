## Analyzing the NYC Subway Dataset ##

Today we will investigate the relation between weather and subway ridership in New York City in May 2011. All data and code is available in the [git repository \[1\]][1].

The weather data, downloaded from [\[weatherData\]][&weatherData], includes information regarding temperature, daily precipitation, barometric pressure, and wind speed, etc.

The subway data is a sample of the MTA New York City Subway dataset that includes hourly entries and exits at the turnstiles of different subway stations.

### Wrangling the Data ###

### Exploratory Data Analysis ###

The following code produces two histograms of hourly entries in rainy days are non-rainy days 

```python
# plot a historgram for hourly entries when it is raining and not
# raining - use ggplot
from ggplot import *
imagename = "entries_hist.png" 
plot = ggplot(aes(x='ENTRIESn_hourly', fill='rain', color='rain',
                  legend=True), data=turnstile_weather) +\
    geom_histogram(binwidth=300) +\
    xlim(low=0, high=6000) +\
    ggtitle('Histogram of ENTRIESn_hourly') +\
    xlab('Entriesn_hourly') +\
    ylab('Frequency')
ggsave(imagename, plot, path='plots', width=6, height=4,
       bbox_inches='tight')
```

<img class="displayed" src="plots/entries_hist.png" width="600px" height="auto">>

The x-axis has been truncated at 6000 to leave out outliers in the
long tail. There are a lot more samples with no rain than rain. We
suspect that more people enter and ride subways on rainy days. To test
this hypothesis, we need to perform a statistical significance
test. Note that the neither of the rainy or non-rainy samples are
normally distributed. Therefore the commonly used Welch’s T test does
not apply here.

### Statistical Inference ###

Now that we have observed evidence in the sample data set that
ridership in rainy days is larger than that in non-rainy days, we are
ready to perform a statistical test and to report the confidence level
in the evidence.

The histograms of hourly ridership in rainy and non-rainy days show
that the data is not normally distributed. So a parametric test is not
appropriate on such data, and we used a non-parametric test such as
Mann Whitney U test [\[udacityMH\]][&udacityMH] . Additionally we
chose a confidence interval of 95%; i.e. we reject the null hypothesis
if the p-value is less than 0.05.

The distributions of the two populations, hourly riderships in rainy
and hourly riderships in non-rainy days, are unknown. So instead of
testing for means or distributions, we use Mann-Whitney U test to test
if we draw randomly from each population, whether one draw is likely
to generate a higher value than the other. Stated mathematically,
given a random draw x from population X of hourly ridership in reainy
days, and a random draw y from population Y of hourly ridership in
non-rainy days we formulate the two-tailed hypotheses as follows:

* The alternative hypothesis states: Each hourly ridership in rainy days is more likely to be differnt than each houry ridersip in non-rainy days; i.e.

<center>H<sub>1</sub>: P (x > y) [math]\neq[/math] 0.5</center>

* The corresponding null hypothesis states: Each hourly ridership in rainy days has equal chance of being greater or smaller than each houry ridersip in non-rainy days; i.e.

<center>H<sub>0</sub>: P (x > y) = 0.5</center>

According to the null hypothesis, the observed differences in sample
mean response times during rainy and non-rainy days are the result of
pure chance, and cannot be generalized to the entire population.

Here we apply statistical inference to generalize the effect observed
in the samples to the entire population, because data were randomly
sampled from a large population.

In particular, we want to establish that the difference between mean
hourly ridership in rainy and non-rainy days is greater than 0. We do
so by rejecting the null hypothesis: the observed differences in mean
hourly ridership of the samples in rainy and non-rainy days are the
result of pure chance, and cannot be generalized to the entire
population.

The histograms of hourly ridership on rainy and non-rainy days show
that the data is not normally distributed. So t-test is not
appropriate on such data, and we used a non-parametric test such as
Mann Whitney U test [\[udacityMH\]][&udacityMH] . Additionally we
chose a confidence interval of 95%; i.e. we reject the null hypothesis
if the p-value is less than 0.05.

We use the Python package scipy.stats.mannwhitneyu
[\[scipyMH\]][&scipyMH] to compute the t and p-values. Data under
incongruent and congruent conditions are used as sample 1 and sample 2
respectively.

```python
# Compute Mann Whitney U test (two samples one-tailed test with 95%
# confidence interval). You would reject the null hypothesis of a 
# greater-than test when p/2 < alpha and t > 0, and of a less-than 
# test when p/2 < alpha and t < 0.
with_rain = turnstile_weather[turnstile_weather.rain == 1]['ENTRIESn_hourly']
without_rain = turnstile_weather[turnstile_weather.rain == 0]['ENTRIESn_hourly']
print "rainy day mean    = ", np.mean(with_rain)
print "non-rainy day mean = ", np.mean(without_rain)
U, p = scipy.stats.mannwhitneyu(with_rain, without_rain)
print "Mann Whitney U = ", U
print "one-tailed p-value: P > U = ", p
```

The test results are listed below. Note that while our alternative
hypothesis is two-tailed, the python package only returns a one-tailed
p-value; see next paragraph on how to post-process the results:

```
rainy day mean    =  1105.4464
non-rainy day mean =  1090.2788
Mann Whitney U = 8.02071
one-tailed p-value: P > U =  0.02499
```

Since the U distribution is symmetric, the two-tailed p-value for the
alternative hypothesis, P > |U|, is twice the one tailed p-value
computed above [\[tailTests\]][&tailTests] [\[tailed2\][&tailed2] [\[paired\]][&paired].

```python
alpha = 0.5   # significance level
# one-tailed test, two-tailed hypothesis
if (2*p) < alpha:
    print "Reject null hypotheis because P > |U| = {0} < {1}".format(2*p, alpha)
else:
    print "Cannot reject null hypotheis becuase P > |U| = {0} >= {1}".format(2*p, alpha)
```

This derived two-tailed p-value, 4.99%, is less than the alpha value
of 0.05. Therefore, the null hypothesis can be rejected in favor of
the alternaitve hypothesis, as shown in the output below:

```
Reject null hypotheis because P > |U| = 0.000548213914249 < 0.5%
```

The Mann-Whitney U test produces a one-tail p-value of
0.02499. Although the observed mean ridership on rainy days is larger
than that on non-rainy days, we use a two-tailed test to account for
the cases where ridership on rainy days can either be higher or lower
than that on non-rainy days. The one-tail p-value can be doubled to
get a two-tail p-value of 4.99% [\[tailTests\][&tailTests] [\[tailed2\][&tailed2]. This implies a 4.99% chance of
seeing a U value as extreme as in our test, if the subway ridership
was the same on rainy and non-rainy days. In other words, for a
p-critical value of 5%, there is a statistically significant
difference between the hourly number of riders on rainy and non-rainy
days.

### Linear Regression ###

### Visualization ###

### Do People Ride the Subway More when it is Raining ? ###

Our analyses of the May 2011 data show that more people ride the subway when it is raining.

Our exploratory data analyses indicated that more people might ride
the subway on rainy days. When we compare the density plots or hourly
entries on rainy and non-rainy days both on weekends and weekdays, we
see that the weekday density plot of rainy days (color blue) have
wider right tail.

![](plots/riders_density.png?raw=true)

The above plot was created with the following code. Note that the legends
do not work in the current version of ggplot.

```python
plot = ggplot(turnstile_weather, aes(x='ENTRIESn_hourly', color='rain')) + \
       geom_density() + \
       xlim(0, 3000) +\
       facet_wrap('weekday') +\
       labs(x='Entriesn_hourly', y='Count of Records')
```

According to the sample data set, average ridership on the NYC subway
increases from 1090 entries per hour on non-rainy days to 1105 entries
per hour on rainy days. As mentioned above, we we validated this
observation with a statistical test. The Mann-Whitney U test produces
a one-tail p-value of 0.02499, which leads to a two-tail p-value of
4.99%. In other words, for a p-critical value of 5%, there is a
statistically significant increase in the hourly number of riders on
rainy days compared to non-rainy days.

Next step is to predict the number of hourly entries linear regression
models. Unfortunately the Gradient Descent model produced a negative
co-efficient for the 'rain' variable which seems to imply that fewer
people ride the subway when it is rainig. This is the opposite of what
we expected. We suspect that the calculation of hourly entries in the
original dataset is not reliable (more on this in the 'Reflection'
section). The improved data set, on the other hand, counts the hourly
entries in a more consistent manner. When the regression model is
trained on the improved data, it produces a positive coefficient for
the 'rain' variable.

### Reflection on Shortcomings of Data and Analysis ###

#### Shortcomings of Data ####

We have used statistical techinques to analyse data and build
regression models. For these techniques to work, the dataset needs to
represent a random sample of the population. In this section we will
examine how "random" the sample really is.


The originial MTA data set contains time-series records of
cumulative number of entries and exits at each turnstile. From
		this data we calculated hourly entries and exits by taking the
		difference between the successive records. This is valid only
		if the time series data is colleced at fixed intervals in a
		periodic fashion. Unfortunately this is not the case, unit R003 only
		logs data every 4 hours starting from midnight. Another unit,
		R550, logs data many times every hour.

```python
unitName = 'R003'
date = '2011-05-01'
hours = df[(df.UNIT==unitName) & (df.DATEn==date)].TIMEn
print "Unit ", unitName, " has logged entries on ", date, " at hours:"
print hours.values
```

The following histogram clearly shows how the total number of
records vary from hour to hour. Due to this uneven number of
records per hour, our calculation of entries per hour is not
acccurate, and may lead to unreliable conclusions and predictions.

```python
rph_plot = ggplot(aes(x='hour'), data=df) +\
           geom_histogram() +\
		   ggtitle('Histogram of Hour') + \
		   xlab('Hour or Entry') + ylab('Number of Records')
```

<img class="displayed" src="plots/hrhist.png" width="600px" height="auto">


This problem has been addressed in the improved data set [\[combinedData\]][&combinedData].
Indeed, it contains an even number of records for every 4-hour
period. 

<img class="displayed" src="plots/hrhist_improved.png" width="600px" height="auto">

#### Shortcomings of Analysis ####

Although the MTA data records entry and exit data for each station
several times a day, the weather data is recorded once a day. Thus on
a given day, we expect the hourly variations in ridership independent
of whether it is rainy or not.

Since the Mann-Whitney test confirmend our expectation that more
people ride the subway on rainy days, we expect the theta coefficient
of the 'rain' varaible to be positive in the regression
model. Although, this coefficient is negative in the model creative in
the model trained on the original data, it is positive in the improved
data. This confirms that the data should be adequately cleaned up
before using to to build predictive models.

A careful examination of the residuals can tell us if our choice of
the regression model is appropriate. Residuals are a form of error,
and basically we expect the errors to be normally and independently
distributed with a mean of 0 and some constant variance [5]. Here is a
histogram of the residuals.

<img class="displayed" src="plots/gdes_residuals.png" width="600px" height="auto">

```python
# Test residuals for normal distribution
residuals = outcomes - predictions
(zscore, pvalue) = scipy.stats.normaltest(residuals)
legLabel = 'mean = {0}, pvalue = {1}'.format(residuals.mean(), pvalue)

# Plot a histogram of residuals
plt.figure()
residuals.hist(bins=20, label=legLabel)
plt.xlabel(prefix+" Residuals")
plt.ylabel("Frequency")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
fancybox=True, shadow=True)
```

Although the histogram looks a good fit for Gaussian distribution
(symmetrical with peak in the middle), the normality test produces a
p-value of 0.0. In other words, the residuals deviate from a normal
distribution in a statistically significant manner [6]. This failure
in the normality test needs further investigation: we may either
examine if this caused by a few outliers, or swtich to non-parametric
tests.

We could have worked a bit more to improve the performance of the
model by a combination of the following two approaches.
		
* Make the model more complex by either adding more features
or by using polynomial combinations of features. However, a complex model
 takes longer to train and to use.
 
* Make the data more complex and train the model with a larger data
set. However, processing a large data set often requires a distributed
processing framework such as MapReduce.

Note that these two approaches are complimentary. If the model is too
simple, it underfits the data (high bias) and its performance cannot
be improved by feeding more data. If the model is too complex, it
overfits the data (high variance), and its performance can be improved
by tuning it with more data. </p>

#### Shortcomings of Statistical Test ####

The conclusions drawn based on Mann-Whitney U test results can be made
stronger with additional descriptive statistics such as interquartile
range [\[13\]][13].


### References

[&gitRepo]: https://github.com/samitchaudhuri/analyze-subway "Analyze New York City Subway Ridership"
[&weatherData]: https://www.dropbox.com/s/7sf0yqc9ykpq3w8/weather_underground.csv "Underground Weather Data"
[&combinedData]: https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv "Combined MTA and underground weather data."
[&scipyMH]: http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html "sipy.stats.mannwhitneyu"
[&udacityMH]: https://storage.googleapis.com/supplemental_media/udacityu/649959144/MannWhitneyUTest.pdf "Understanding the Mann-Whitney U Test"
[&tailTests]: http://www.ats.ucla.edu/stat/mult_pkg/faq/general/tail_tests.htm "What are the differences between one-tailed and two-tailed tests ?"
[&tailed12]: http://graphpad.com/guides/prism/6/statistics/index.htm?one-tail_vs__two-tail_p_values.htm "One-tail vs. two-tail P values, GraphPad Software"
[&paired]: https://en.wikipedia.org/wiki/Student%27s_t-test#Unpaired_and_paired_two-sample_t-tests "Unpaired and paired two-sample t-tests"
[5]: http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm "Are the model residuals well behaved?, NIST Engineering Statistics Handbook"
[6]: http://www.graphpad.com/guides/prism/6/statistics/index.htm?stat_interpreting_results_normality.htm "Interpreting results: Normality tests, GraphPad Software"
[7]: http://en.wikipedia.org/wiki/Ordinary_least_squares "Ordinary least 
squares, Wikipedia"
[8]: http://statsmodels.sourceforge.net/devel/generated/statsmodels.regression.linear_model.OLS.html "Statsmodels regression OLS"
[9]: http://en.wikipedia.org/w/index.php?title=Linear_least_squares_(mathematics) "Linear least squares (mathematics), Wikipedia"
[10]: http://en.wikipedia.org/wiki/Polynomial_regression "Polynomial regression, Wikipedia"
[11]: http://people.duke.edu/~rnau/rsquared.htm#punchline "What's
  a good value for R-squared ?"

[\[gitRepo\] Analyze New York City Subway Ridership][&gitRepo]

[\[weatherData\] Underground Weather Data] [&weatherData]

[\[combinedData\] Combined MTA and underground weather data.] [&combinedData]

[\[scipyMH\] scipy.stats.mannwhitneyu] [&scipyMH]

[\[udacityMH\] Understanding the Mann-Whitney U Test] [&udacityMH]

[\[tailTests\] scipy.stats.mannwhitneyu] [&tailTests]

[\[tailed12\] One-tail vs. two-tail P values, GraphPad Software] [&tailed12]

[\[paired\] Unpaired and paired two-sample t-tests] [&paired]
