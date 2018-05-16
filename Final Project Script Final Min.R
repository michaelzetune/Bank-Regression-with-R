# Pull csv
marketing <- read.csv("https://raw.githubusercontent.com/MichaelZetune/Bank-Regression-with-R/master/Bank%20Data%20-%20bank-additional.csv")
View(marketing)

marketing$duration <- NULL

# Since the vast majority of the data points are NA now, we will discard this column:
marketing$pdays <- NULL

#   21 - make.account - has the client subscribed a term deposit? (binary: 'yes','no')
# Let's make this a clearer dummy variable to suit it for logistic regression. We'll change the name of the column slightly and delete the original column to make this happen:
marketing$made.account[marketing$make.account == 'yes'] <- 1
marketing$made.account[marketing$make.account == 'no'] <- 0
marketing$make.account <- NULL


model1 <- glm(made.account ~ ., data=marketing, family='binomial')
summary(model1)

# Assumptions:

# Assumption 1: The dependent variable should be binary
# This is true because made.account is now 0 or 1

# Assumption 2: Observations should be independent of each other
# Since customers are unrelated individuals in our data, this is true

# Assumption 3: No multicollinearity among independent variables
library(car)
vif(model1) # returns error, fix is below

# VIF returns an error with aliased coefficients, so we need to find where the perfect multicollinearity is:
alias(model1)

# We find that loan is "unknown" if and only if housing if "unknown". So we should eliminate rows from the dataset where housing is unknown
marketing <- subset(marketing, marketing$housing != 'unknown')
model1 <- glm(made.account ~ ., data=marketing, family='binomial')
alias(model1)

# Now use a correlation matrix to verify minimal multicollinearity in general:
marketing.numeric.bool <- unlist(lapply(marketing, is.numeric))
marketing.numeric <- marketing[ , marketing.numeric.bool]
cor(marketing.numeric)

# Assumption 4: Large data set
# The marketing data set has over 4000 rows, which is a very large data set for this problem


# Now that assumptions are done, create a backward, forward, and 'both' model and hold on to them to assess performance later
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

# Now we will try an exhaustive search of all variables in the dataset. A warning was returned in regards to lienar dependencies, so we had to exclude the 'housing' and 'loan' columns from the analysis
install.packages("leaps")
library(leaps)

#This next command will take a minute or so to run (at least on my computer), but it's retreiving the best subset while considering up to 20 variables in the dataset.
regsubsets.output <- regsubsets(made.account ~ age + job + marital + education + default + month + day_of_week + campaign + previous + poutcome + emp.var.rate + cons.price.idx + euribor3m + nr.employed, data=marketing, nvmax=20)

# Use outmat variable to see the best subset of variables for 1,2 ... 8 variables
best.subset.summary <- summary(regsubsets.output)
best.subset.summary$outmat

# Use Adjusted R^2 to find the best model from regsubsets
best.subset.overall <- which.max(best.subset.summary$adjr2)
best.subset.overall

# The regsubsets function suggests using 19 of the variables. These variables create the logistic regression. Many of these variables are dummy variables, so the final model is actually fewer:
subset.model <- glm(made.account ~ age + job + month + campaign + previous + poutcome + emp.var.rate + euribor3m + nr.employed, data=marketing, family='binomial')
summary(subset.model)

# We now have four models to consider: forward.model, backward.model, both.model, and subset.model. Compare AIC and R^2:
AIC(forward.model) # 2230.536
AIC(backward.model) # 2224.604
AIC(both.model) #2230.536
AIC(subset.model) #2256.057

# forward.model pseudo R^2
1-(2196.5/2783.7) # = .2109
# backward.model pseudo R^2
1-(2188.6/2783.7) # = .2138
# both.model pseudo R^2
1-(2196.5/2783.7) # = .2109
# subset.model pseudo R^2
1-(2198.1/2783.7) # = .2104

# Additionally, predictive accuracy is listed below. They are approximately the same across models.

#Naive model
sum(marketing$made.account == 0)/nrow(marketing) #0.8899

#forward.model
predicted.frwd <- (predict(forward.model, type = 'response') >= 0.5)
actual.frwd <- (marketing$made.account == 1)
sum(predicted.frwd == actual.frwd) / nrow(marketing) #0.9041

#backward.model
predicted.bwrd <- (predict(backward.model, type = 'response') >= 0.5)
actual.bwrd <- (marketing$made.account == 1)
sum(predicted.bwrd == actual.bwrd) / nrow(marketing) #0.9033

#both.model
predicted.both <- (predict(both.model, type = 'response') >= 0.5)
actual.both <- (marketing$made.account == 1)
sum(predicted.both == actual.both) / nrow(marketing) #0.9041

#subset.model
predicted.sub <- (predict(subset.model, type = 'response') >= 0.5)
actual.sub <- (marketing$made.account == 1)
sum(predicted.sub == actual.sub) / nrow(marketing) #0.9021


# Since the pseudo R^2s and predictive accuracy are about the same, we use AIC to judge. The backward.model clearly has is the best at modeling our bank marketing information. Cheers!
summary(backward.model)
