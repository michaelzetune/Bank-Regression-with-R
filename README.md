# Bank Marketing Regression with R

## This project was created for an upper-division statistics course at UT. We used data from UCI's [dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) on their Machine Learning Repository.

## The R script in this repo is replicated below in a more readable form:


## 1. Pull csv from Michael's GitHub and view

```R
   marketing <- read.csv("https://raw.githubusercontent.com/MichaelZetune/Bank-Regression-with-R/master/Bank%20Data%20-%20bank-additional.csv")
   View(marketing)
```

## 2. Analyze variables:

#### We can keep the following variables as is for now:
- __age__ (numeric)
- __job__ : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
- __marital__ : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
- __education__ (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
- __default__: has credit in default? (categorical: 'no','yes','unknown')
- __housing__: has housing loan? (categorical: 'no','yes','unknown')
- __loan__: has personal loan? (categorical: 'no','yes','unknown')
related with the last contact of the current campaign:
- __contact__: contact communication type (categorical: 'cellular','telephone')
- __month__: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
- __day_of_week__: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')

#### Other attributes:
- __campaign__: number of contacts performed during this campaign and for this client (numeric, includes last contact)
- __previous__: number of contacts performed before this campaign and for this client (numeric)
- __poutcome__: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
 social and economic context attributes
- __emp.var.rate__: employment variation rate - quarterly indicator (numeric)
- __cons.price.idx__: consumer price index - monthly indicator (numeric)
- __cons.conf.idx__: consumer confidence index - monthly indicator (numeric)
- __euribor3m__: euribor 3 month rate - daily indicator (numeric)
- __nr.employed__: number of employees - quarterly indicator (numeric)


#### We need to do some cleaning for these:

- __duration__: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

#### We should discard duration:
```R
marketing$duration <- NULL
```

- __pdays__: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)

#### Since 999 is an arbitrary dummy varaible, we should make it NA:
```R
marketing$pdays[marketing$pdays == 999] <- NA
```

#### Since the vast majority of the data points are NA now, we will discard this column:
```R
marketing$pdays <- NULL
```

- __make.account__ - has the client subscribed a term deposit? (binary: 'yes','no')

#### Let's make this a clearer dummy variable to suit it for logistic regression. We'll change the name of the column slightly and delete the original column to make this happen:

```R
marketing$made.account[marketing$make.account == 'yes'] <- 1
marketing$made.account[marketing$make.account == 'no'] <- 0
marketing$make.account <- NULL
```

#### Now we will create the initial model, using all of the variables:

```R
model1 <- glm(made.account ~ ., data=marketing, family='binomial')
summary(model1)
```

## 3. Check Assumptions:

### Assumption 1: The dependent variable should be binary
- This is true because made.account is now 0 or 1

### Assumption 2: Observations should be independent of each other
- Since customers are unrelated individuals in our data, this is true

### Assumption 3: No multicollinearity among independent variables
```R
library(car)
vif(model1) # returns error, fix is below
```

### VIF returns an error with aliased coefficients, so we need to find where the perfect multicollinearity is:
```R
alias(model1)
```

### We find that loan is "unknown" *if and only if* housing if "unknown". So we should eliminate rows from the dataset where housing is unknown
```R
marketing <- subset(marketing, marketing$housing != 'unknown')
model1 <- glm(made.account ~ ., data=marketing, family='binomial')
alias(model1)
```

### Now use a correlation matrix to verify minimal multicollinearity in general:
```R
marketing.numeric.bool <- unlist(lapply(marketing, is.numeric))
marketing.numeric <- marketing[ , marketing.numeric.bool]
cor(marketing.numeric)
```

### Assumption 4: Large data set
- The marketing data set has over 4000 rows, which is a very large data set for this problem


## 3. Now that assumptions are verified, create a backward, forward, and 'both' model and hold on to them to assess performance later
```R
null <- glm(made.account ~ 1, data=marketing, family='binomial')
full <- glm(made.account ~ ., data=marketing, family='binomial')

backward.model <- step(full, scope=list(lower=null, upper=full), direction='backward')
summary(backward.model)
AIC(backward.model)

forward.model <- step(null, scope = list(lower=null, upper=full), direction = 'forward')
summary(forward.model)
AIC(forward.model)

both.model <- step(null, scope=list(lower=null, upper=full), direction='both')
summary(both.model)
AIC(both.model)
```
## 4. Exhaustive Search for Best Attributes

### Now we will try an exhaustive search of all variables in the dataset. A warning was returned in regards to lienar dependencies, so we had to exclude the 'housing' and 'loan' columns from the analysis
```R
install.packages("leaps")
library(leaps)
```

### This next command will take some time to run, but it's retreiving the best subset while considering up to 20 variables in the dataset.
```R
regsubsets.output <- regsubsets(made.account ~ age + job + marital + education + default + month + day_of_week + campaign + previous + poutcome + emp.var.rate + cons.price.idx + euribor3m + nr.employed, data=marketing, nvmax=20)
```

### Use outmat variable to see the best subset of variables for 1,2 ... 8 variables
```R
best.subset.summary <- summary(regsubsets.output)
best.subset.summary$outmat
```

### Use Adjusted R^2 to find the best model from regsubsets
```R
best.subset.overall <- which.max(best.subset.summary$adjr2)
best.subset.overall
```

### The regsubsets function suggests using 19 of the variables. These variables create the logistic regression. Many of these variables are dummy variables, so the final model is actually looks smaller:
```R
subset.model <- glm(made.account ~ age + job + month + campaign + previous + poutcome + emp.var.rate + euribor3m + nr.employed, data=marketing, family='binomial')
summary(subset.model)
```

## 5. Analyze Models
### We now have four models to consider: forward.model, backward.model, both.model, and subset.model. Compare AIC and R^2:
```R
# AIC Comparison
AIC(forward.model) # 2230.536
AIC(backward.model) # 2224.604
AIC(both.model) #2230.536
AIC(subset.model) #2256.057

# Pseudo R^2 Comparison
# forward.model
1-(2196.5/2783.7) # = .2109
# backward.model
1-(2188.6/2783.7) # = .2138
# both.model
1-(2196.5/2783.7) # = .2109
# subset.model
1-(2198.1/2783.7) # = .2104
```

### Additionally, predictive accuracy is listed below. They are approximately the same across models.
```R
# Naive model
sum(marketing$made.account == 0)/nrow(marketing) #0.8899

# forward.model
predicted.frwd <- (predict(forward.model, type = 'response') >= 0.5)
actual.frwd <- (marketing$made.account == 1)
sum(predicted.frwd == actual.frwd) / nrow(marketing) #0.9041

# backward.model
predicted.bwrd <- (predict(backward.model, type = 'response') >= 0.5)
actual.bwrd <- (marketing$made.account == 1)
sum(predicted.bwrd == actual.bwrd) / nrow(marketing) #0.9033

# both.model
predicted.both <- (predict(both.model, type = 'response') >= 0.5)
actual.both <- (marketing$made.account == 1)
sum(predicted.both == actual.both) / nrow(marketing) #0.9041

# subset.model
predicted.sub <- (predict(subset.model, type = 'response') >= 0.5)
actual.sub <- (marketing$made.account == 1)
sum(predicted.sub == actual.sub) / nrow(marketing) #0.9021
```

## 6. Conlusion
### Since the pseudo R^2s and predictive accuracy are about the same, we use AIC to judge. The backward.model clearly has is the best at modeling our bank marketing information. Cheers!
```R
summary(backward.model)
```
