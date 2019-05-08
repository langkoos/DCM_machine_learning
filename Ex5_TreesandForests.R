#Ex.5: Swiss metro

#Load data
setwd("/Users/cuauhtemoc/Documents/Machine Learning Course") #set working directory
swissmetro <- read.table("data/swissmetro.dat",header = TRUE) 
#omit unnecessary columns
swissmetro<- swissmetro[,-which(names(swissmetro) %in% c('GROUP','SURVEY','SP','ID','SM_AV','TRAIN_AV'))]

#Steps
#Step 1) Check factor variables
#Step 2) Check continuous variables
#Step 3) Train/test set
#Step 4) Build the model
#Step 5) Assess the performance of the model

#Step 1) Check factor variables
factorlabels <- c('CHOICE','PURPOSE','FIRST','TICKET','WHO','LUGGAGE','AGE','MALE',
                  'INCOME','GA','ORIGIN','DEST','CAR_AV','SM_SEATS')
swissmetro[,factorlabels]<-lapply(swissmetro[,factorlabels],factor)

#Step 2) Check continuous variables
continuous <-select_if(swissmetro, is.numeric)
summary(continuous)
#Standarize numeric columns
swissmetro <- swissmetro %>% mutate_if(is.numeric, funs(as.numeric(scale(.))))

#Step 3) Train/test set
N <-nrow(swissmetro) #total number of rows
percentage_training <- 0.7 #percentage of training samples
train_index <-sample(N,size=round(percentage_training*N))
test_index  <- (1:nrow(swissmetro))[-train_index]

#Step 4) Build the model (Multinomial Logistic Regression)
library(nnet) 
#reference level
swissmetro$CHOICE <- relevel(swissmetro$CHOICE, ref=1)
#Train model
mlr <- multinom(CHOICE~ ., data=swissmetro[train_index,])
#summary(mlr)

#Step 5) Assess the performance of the model
mlr.predict<-predict(mlr,swissmetro[test_index,])
mlr.matrix<-table(mlr.predict,swissmetro[test_index,'CHOICE'])
mlr.acc<- sum(diag(mlr.matrix))/sum(mlr.matrix)
mlr.acc

################
#5.2 Decision Tree
library(tree)

#Step 4) Build the model
#fit a tree
tree.swissmetro = tree(CHOICE ~ ., data=swissmetro[train_index,],
                       control = tree.control(nobs = length(train_index),mindev = 0.005))
plot(tree.swissmetro)
text(tree.swissmetro, cex=0.75)

#Prune Tree
#let us see if the trees needs pruning
#we use the argument "FUN = prune.misclass" because we are doing classification
cv.pruning = cv.tree(tree.swissmetro, FUN = prune.misclass)
plot(cv.pruning$size, cv.pruning$dev, pch=20, col="red", type="b",
     main="Cross validation for finding the optimal size",
     xlab="size of tree", ylab="performance")


#let us extract the optimal size
best.size = cv.pruning$size[which(cv.pruning$dev == min(cv.pruning$dev))]
cat(" Optimal size =", best.size, "nn")

#and let us prune the tree accordingly
tree.swissmetro.pruned = prune.misclass(tree.swissmetro, best=rev(best.size)[1])
plot(tree.swissmetro.pruned)
text(tree.swissmetro.pruned, cex=0.75)

#Step 5) Assess the performance of the model
tree.swissmetro.pred = predict(tree.swissmetro.pruned , swissmetro[test_index,],type='class')

#Confusion matrix
tree.matrix<-table(tree.swissmetro.pred,swissmetro[test_index,'CHOICE'])

#Accuracy
tree.acc<- sum(diag(tree.matrix))/sum(tree.matrix)
tree.acc

################
#5.3 Bagging Decision Trees
library(randomForest)

#Step 4) Build the model
bagging <- randomForest(CHOICE ~ ., swissmetro[train_index,],ntree=100,mtry=21)

#Step 5) Assess the performance of the model
bagging.predict<-predict(bagging,swissmetro[test_index,], type='response')

#Confusion matrix
bagging.matrix<-table(bagging.predict,swissmetro[test_index,'CHOICE'])

#Accuracy
bagging.acc<- sum(diag(bagging.matrix))/sum(bagging.matrix)
bagging.acc

################
#5.4 Random Forest

#Step 4) Build the model
rf <- randomForest(CHOICE ~ ., swissmetro[train_index,],ntree=800,mtry=5)

#Step 5) Assess the performance of the model
rf.predict<-predict(rf,swissmetro[test_index,], type='response')

#Confusion matrix
rf.matrix<-table(rf.predict,swissmetro[test_index,'CHOICE'])

#Accuracy
rf.acc<- sum(diag(rf.matrix))/sum(rf.matrix)
rf.acc

