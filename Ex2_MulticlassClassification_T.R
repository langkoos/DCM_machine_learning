#Ex.2: PostCar World Mode Choice

#Libraries -----------------------------------------------------------
library(nnet) #Softmax Regression
library(dplyr)

#Load data-------------------------------------------------------------
#Set working directory
setwd("/Users/cuauhtemoc/Documents/Machine Learning Course")
#Load data
postcar <- read.csv("data/mc_archive.csv",header = TRUE,sep = ';') 

#Omit unnecessary columns
todelete<-c('person_id','studycode','av_cp','av_pt')
postcar<- postcar[,-which(names(postcar) %in% todelete)]

#Strategy -------------------------------------------------------------
#Step 1) Check factor variables
#Step 2) Check continuous variables
#Step 3) Train/test set
#Step 4) Build the model
#Step 5) Assess the performance of the model

#########################################################################
#Ex. 2.1: Softmax Regression 

#Step 1) Check factor variables ----------------------------------------
# Convert categorical columns to factors
factorlabels <- c('choice','purpose','av_bf','av_v','av_cs','cost',
                  'time','acc','risk','trns','freq')
postcar[,factorlabels]<-lapply(postcar[,factorlabels],factor)

#Step 2) Check continuous variables-------------------------------------
continuous <-select_if(postcar, is.numeric)
summary(continuous)
#Standarize numeric columns
postcar <- postcar %>% mutate_if(is.numeric, funs(as.numeric(scale(.))))

#Step 3) Train/test set-------------------------------------------------
N <-nrow(postcar) #total number of rows
percentage_training <- 0.7 #percentage of training samples
train_index <-sample(N,size=round(percentage_training*N))
test_index  <- (1:nrow(postcar))[-train_index]

#Step 4) Build the model-------------------------------------------------
#Softmax Regression (aka Multinomial Logistic Regression)

#Declare reference level for Softmax Regression
postcar$choice <- relevel(postcar$choice, ref='1')

#Use multinom to fit model on train data
mlr <- multinom(choice~ ., data=postcar[train_index,])

#Step 5) Assess the performance of the model------------------------------

#Predict on test data
mlr.predict<-predict(mlr,postcar[test_index,])

#Confusion matrix
mlr.matrix<-table(mlr.predict,postcar[test_index,'choice'])

#Calculate accuracy
mlr.acc<- sum(diag(mlr.matrix))/sum(mlr.matrix)
mlr.acc

#########################################################################
#Ex. 2.2: Regularization (Softmax Regression)
#Note: tudents would have implement before lasso and ridge, the following they can do from scratch
#Implement Lasso Softmax Regression on the PostCar World dataset
library(glmnet)

#Separate data into xtrain,ytrain,xtest,ytest. x should be data.matrix
xtrain<-data.matrix(select(postcar,-choice)[train_index,])
xtest<-data.matrix(select(postcar,-choice)[test_index,])
ytrain<-postcar[train_index,'choice']
ytest<-postcar[test_index,'choice']

#Set Lambdas
lambda.grid = 10^seq( -6, 2,length=100)


#Lasso Softmax Regression (i.e. L1 norm) ---------------------------------

#ridge a=0, lasso a=1, type.measure = "mse" for regression, "class" for classification
lasso.cv = cv.glmnet(xtrain, ytrain, alpha=1,type.measure = "class",family='multinomial', nfolds = 10,lambda = lambda.grid) #lasso a=1

#plot the coefficients as a function of lambda plot(ridge.cv)
plot(lasso.cv)

#Prediction
lasso.predict.test = predict(lasso.cv,newx = xtest, s ='lambda.min', exact=TRUE,type='class')
lasso.predict.test = as.factor(lasso.predict.test )

#Confusion matrix
lasso.matrix<-table(lasso.predict.test,ytest)

#Accuracy
lasso.acc<- sum(diag(lasso.matrix))/sum(lasso.matrix)
lasso.acc

#Did the accuracy improve?
#What can you see from the value of the coefficients?
coef(lasso.cv,s='lambda.min')

#########################################################################
#Ex. 2.3: Decision Tree

library(tree)

#Step 4) Build the model-------------------------------------------------

#4.1  Grow a tree

dtree = tree(choice ~ ., data=postcar[train_index,],
                       control = tree.control(nobs = length(train_index),mindev = 0.001))
plot(dtree)
text(dtree, cex=0.75)

#4.2 Prune the Tree
#Let us see if the trees needs pruning
#We use the argument "FUN = prune.misclass" because we are doing classification
cv.pruning = cv.tree(dtree, FUN = prune.misclass)

#Plot size of tree vs. performance (from cross validation)
plot(cv.pruning$size, cv.pruning$dev, pch=20, col="red", type="b",
     main="Cross validation for finding the optimal size",
     xlab="size of tree", ylab="performance")

#Let us extract the optimal size
best.size = cv.pruning$size[which(cv.pruning$dev == min(cv.pruning$dev))]
cat(" Optimal size =", best.size, "nn")

#and let us prune the tree accordingly
dtree.pruned = prune.misclass(dtree, best=best.size[1])

#Plot pruned tree
plot(dtree.pruned)
text(dtree.pruned, cex=0.75)

#Step 5) Model Validation -------------------------------------------------

#Predict
dtree.pruned.pred= predict(dtree.pruned , postcar[test_index,],type='class')
dtree.pred= predict(dtree , postcar[test_index,],type='class')

#Confusion matrix
dtree.pruned.matrix<-table(dtree.pruned.pred,postcar[test_index,'choice'])
dtree.matrix<-table(dtree.pred,postcar[test_index,'choice'])

#Accuracy
dtree.pruned.acc<- sum(diag(dtree.pruned.matrix))/sum(dtree.pruned.matrix)
dtree.acc<-sum(diag(dtree.matrix))/sum(dtree.matrix)

dtree.pruned.acc
dtree.acc


#########################################################################
#Ex. 2.4: Random Forest

library(randomForest)

#Step 4) Build the model
rf <- randomForest(choice ~ ., postcar[train_index,],ntree=800,mtry=6)

#Step 5) Validate the model

#Predict
rf.predict<-predict(rf,postcar[test_index,], type='response')

#Confusion matrix
rf.matrix<-table(rf.predict,postcar[test_index,'choice'])

#Accuracy
rf.acc<- sum(diag(rf.matrix))/sum(rf.matrix)
rf.acc

#Variable Importance
importance(rf)
varImpPlot(rf)

#Compare Accuracies, which model performs better?
plot(c(mlr.acc,lasso.acc,dtree.acc,rf.acc))

