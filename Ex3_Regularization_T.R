#Ex.3: Regularization

set.seed(2019)

#Libraries -----------------------------------------------------------
library(glmnet) #Ridge and Lasso regression

#Load data-------------------------------------------------------------
setwd("/Users/cuauhtemoc/Documents/Machine Learning Course")
xtrain <- read.csv('data/Tut4CookieXTraining.csv',header = TRUE, sep = ',')
ytrain <- read.csv('data/Tut4CookieYTraining.csv',header = FALSE,sep = ',')
xtest <- read.csv('data/Tut4CookieXTtest.csv',header = TRUE,sep = ',')
ytest <- read.csv('data/Tut4CookieYTest.csv',header = FALSE,sep = ',')

#Transform to classification problem 
avg<- mean(ytrain$V1)
ytrain$V1[ytrain$V1<avg] <-0
ytrain$V1[ytrain$V1>avg] <-1
ytest$V1[ytest$V1<avg] <-0
ytest$V1[ytest$V1>avg] <-1

#Step 1) Check factor variables
ytest <- as.factor(ytest$V1)
ytrain <- as.factor(ytrain$V1)

#Case p>>n!!! Needs regularisation techniques!
p=ncol(xtrain)   #700 covariates!!!!!!
n=nrow(xtrain)   #40 samples   

################################################################################
#3.1 Preparation for glmnet package

#Ridge regression with GLMNET
library(glmnet)

#Necessary transformation for GLMNET 
xtrain = as.matrix(xtrain)
xtest = as.matrix(xtest)

#Lambdas to try
lambda.grid = 10^seq( -5, 2,length=100)

################################################################################
#3.2 Ridge Logistic Regression

#To do: Implement Ridge Logistic Regression for the cookies dataset

#Use cv.glmnet to choose best lambda, alpha=0 -> ridge, type.measure = class
ridge.cv = cv.glmnet(xtrain, ytrain, alpha=0,family="binomial",type.measure = "class", nfolds = 10,lambda = lambda.grid) 

#plot the coefficients as a function of lambda plot(ridge.cv)
plot(ridge.cv)

#Predict on the test data set
ridge.predict.test = predict(ridge.cv,newx = xtest, s = 'lambda.min', exact=TRUE,type='class')
#convert prediction to factor
ridge.predict.test = as.factor(ridge.predict.test )

#Confusion matrix
ridge.matrix<-table(ridge.predict.test,ytest)

#Accuracy
ridge.acc<- sum(diag(ridge.matrix))/sum(ridge.matrix)
ridge.acc


################################################################################
#3.3 Lasso Logistic Regression

#To do: Implement Lasso Logistic Regression for the cookies dataset

#Use cv.glmnet to choose best lambda, alpha=1 -> lasso, type.measure = class
lasso.cv = cv.glmnet(xtrain, ytrain, alpha=1,type.measure = "class",family='binomial', nfolds = 10,lambda = lambda.grid) #lasso a=1

#plot the coefficients as a function of lambda plot(ridge.cv)
plot(lasso.cv)

#Predict on the test data set
lasso.predict.test = predict(lasso.cv,newx = xtest, s = 'lambda.min', exact=TRUE,type='class')
#convert prediction to factor
lasso.predict.test = as.factor(lasso.predict.test )

#Confusion matrix
lasso.matrix<-table(lasso.predict.test,ytest)

#Accuracy
lasso.acc<- sum(diag(lasso.matrix))/sum(lasso.matrix)
lasso.acc

################################################################################
#3.4 Calculate accuracy without regularization and compare with Lasso and Ridge
#Tip: predict on ridge.cv and choose s=0

#Predict on the test data set
noreg.predict = predict(ridge.cv,newx = xtest, s = 0, exact=TRUE,type='class')
#convert prediction to factor
noreg.predict.test = as.factor(noreg.predict )

#Confusion matrix
noreg.matrix<-table(noreg.predict.test,ytest)

#Accuracy
noreg.acc<- sum(diag(noreg.matrix))/sum(noreg.matrix)
noreg.acc

#Show results in  barplot
barplot(c(lasso.acc, ridge.acc,noreg.acc),main="MSE on test set", ylab="Accuracy",names.arg=c("Lasso", "Ridge",'NoReg'), ylim = c(0,1))




 


