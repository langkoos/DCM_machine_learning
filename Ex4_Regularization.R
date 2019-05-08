#Ex.4: NIR spectroscopy for cookie analysis
set.seed(2019)
#1. Import Data
setwd("/Users/cuauhtemoc/Documents/Machine Learning Course")
xtrain <- read.csv('data/Tut4CookieXTraining.csv',header = TRUE, sep = ',')
ytrain <- read.csv('data/Tut4CookieYTraining.csv',header = FALSE,sep = ',')
xtest <- read.csv('data/Tut4CookieXTtest.csv',header = TRUE,sep = ',')
ytest <- read.csv('data/Tut4CookieYTest.csv',header = FALSE,sep = ',')

#Case p>>n!!! Needs regularisation techniques!
p=ncol(xtrain)   #700 covariates!!!!!!
n=nrow(xtrain)   #40 samples   
n==nrow(ytrain)  #Same amount of samples

#Ridge regression with GLMNET
library(glmnet)

#Necessary transformation for GLMNET 
xtrain = as.matrix(xtrain)
ytrain = as.matrix(ytrain)
xtest = as.matrix(xtest)
ytest = as.matrix(ytest)

lambda.grid = 10^seq( -5, 2,length=100)


############################
#RIDGE
ridge.cv = cv.glmnet(xtrain, ytrain, alpha=0,type.measure = "mse", nfolds = 10,lambda = lambda.grid) #ridge a=0

#plot the coefficients as a function of lambda plot(ridge.cv)
plot(ridge.cv)

#make some prediction
ridge.predict.test = predict(ridge.cv,newx = xtest, s = "lambda.min", exact=TRUE)

#MSE
MSE_ridge = mean((ytest - ridge.predict.test)**2)

############################
#LASSO
lasso.cv = cv.glmnet(xtrain, ytrain, alpha=1,type.measure = "mse", nfolds = 10,lambda = lambda.grid) #lasso a=1

#plot the coefficients as a function of lambda plot(ridge.cv)
plot(lasso.cv)

#make some prediction
lasso.predict.test = predict(lasso.cv,newx = xtest, s = "lambda.min", exact=TRUE)

#MSE
MSE_lasso = mean((ytest - lasso.predict.test)**2)

###############################
#Test performance (MSE) of the two approaches
#create barplot
barplot(c(MSE_lasso, MSE_ridge),main="MSE on test set", ylab="MSE",names.arg=c("Lasso", "Ridge"), ylim = c(0,0.5))

###############################
#To do: include MSE without regularization
noreg.pred = predict(ridge.cv,newx = xtest, s = 0, exact=TRUE)
MSE_noreg = mean((ytest - noreg.pred)**2)

barplot(c(MSE_lasso, MSE_ridge,MSE_noreg),main="MSE on test set", ylab="MSE",names.arg=c("LASSO", "Ridge","No Reg"), ylim = c(0,2))

