#Op2: Swiss metro 

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
#Step 5) Validate the model/Generalization

#Step 1) Check factor variables: convert them as factors
factorlabels <- c('CHOICE','PURPOSE','FIRST','TICKET','WHO','LUGGAGE','AGE','MALE',
                  'INCOME','GA','ORIGIN','DEST','CAR_AV','SM_SEATS')
swissmetro[,factorlabels]<-lapply(swissmetro[,factorlabels],factor)

#Step 2) Check continuous variables: scale them
continuous <-select_if(swissmetro, is.numeric)
summary(continuous)
#Standarize numeric columns
swissmetro <- swissmetro %>% mutate_if(is.numeric, funs(as.numeric(scale(.))))

#Step 3) Train/test set
N <-nrow(swissmetro) #total number of rows
percentage_training <- 0.7 #percentage of training samples
train_index <-sample(N,size=round(percentage_training*N))
test_index  <- (1:nrow(swissmetro))[-train_index]

################################################################################
#3.1 Train a Softmax Regression (aka Multinomial Logistic Regression)

#Step 4) Build the model 

library(nnet) 
#reference level
swissmetro$CHOICE <- relevel(swissmetro$CHOICE, ref=1)
#Train model
mlr <- multinom(CHOICE~ ., data=swissmetro[train_index,])
#summary(mlr)

#Step 5) Validate the model/Generalization: report accuracy on test data
mlr.predict<-predict(mlr,swissmetro[test_index,])
mlr.matrix<-table(mlr.predict,swissmetro[test_index,'CHOICE'])
mlr.acc<- sum(diag(mlr.matrix))/sum(mlr.matrix)
mlr.acc

################################################################################
#3.2 Train a Decision Tree

library(tree)

#Step 4) Build the model: grow a tree

dtree = tree(CHOICE ~ ., data=swissmetro[train_index,],
                       control = tree.control(nobs = length(train_index),mindev = 0.001))
plot(dtree)
text(dtree, cex=0.75)

#Prune Tree
#let us see if the trees needs pruning
#we use the argument "FUN = prune.misclass" because we are doing classification
cv.pruning = cv.tree(dtree, FUN = prune.misclass)

#Plot Tree size Cross Validation
plot(cv.pruning$size, cv.pruning$dev, pch=20, col="red", type="b",
     main="Cross validation for finding the optimal size",
     xlab="size of tree", ylab="performance")


#let us extract the optimal size
best.size = cv.pruning$size[which(cv.pruning$dev == min(cv.pruning$dev))]
cat(" Optimal size =", best.size, "nn")

#and let us prune the tree accordingly
dtree.pruned = prune.misclass(dtree, best=rev(best.size)[1])
plot(dtree.pruned)
text(dtree.pruned, cex=0.75)

#Step 5) Assess the performance of the model
dtree.pruned.pred = predict(dtree.pruned , swissmetro[test_index,],type='class')
dtree.pred = predict(dtree , swissmetro[test_index,],type='class')

#Confusion matrix
dtree.pruned.matrix<-table(dtree.pruned.pred,swissmetro[test_index,'CHOICE'])
dtree.matrix<-table(dtree.pred,swissmetro[test_index,'CHOICE'])

#Accuracy
dtree.pruned.acc<- sum(diag(dtree.pruned.matrix))/sum(dtree.pruned.matrix)
dtree.pruned.acc

dtree.acc<- sum(diag(dtree.matrix))/sum(dtree.matrix)
dtree.acc

################################################################################
#3.3 Train a Random Forest

library(randomForest)

#Step 4) Build the model
rf <- randomForest(CHOICE ~ ., swissmetro[train_index,],ntree=800,mtry=5)

#Step 5) Assess the performance of the model
rf.predict<-predict(rf,swissmetro[test_index,], type='response')

#Confusion matrix
rf.matrix<-table(rf.predict,swissmetro[test_index,'CHOICE'])

#Accuracy
rf.acc<- sum(diag(rf.matrix))/sum(rf.matrix)
rf.acc

#Plot variable importance
varImpPlot(rf)


################
#5.5 Gradient Boosting

library(gbm)

boosting <- gbm(CHOICE~., data = swissmetro[train_index,],distribution = 'multinomial',n.trees = 1200,interaction.depth = 4,cv.folds = 5, shrinkage = 0.005)
gbm.perf(boosting, plot.it = TRUE)
boost.optimal<- gbm.perf(boosting, plot.it = FALSE)

gb.predict <- predict(boosting,newdata = swissmetro[test_index,],type = 'response',n.trees = boost.optimal)
summary(boosting)
gb.df <-data.frame(gb.predict)
gb.predict2<-as.factor(apply(gb.df, 1, which.max))
#Confusion matrix
gb.matrix<-table(gb.predict2,swissmetro[test_index,'CHOICE'])

#Accuracy
gb.acc<- sum(diag(gb.matrix))/sum(gb.matrix)
gb.acc



###
##data input prep
X_train = select(swissmetro,-CHOICE)[train_index,]
X_test = select(swissmetro,-CHOICE)[test_index,]
X_train = data.matrix(X_train)
X_test = data.matrix(X_test)
y_train = swissmetro[train_index,'CHOICE']
y_test = swissmetro[test_index, 'CHOICE']
y_train <- to_categorical(as.numeric(y_train)-1,num_classes = 4)
y_test <- to_categorical(as.numeric(y_test)-1, 4)

##model
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 48, activation = 'relu', input_shape = c(21)) %>% 
  #layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 36, activation = 'relu') %>%
  #layer_dense(units = 32, activation = 'relu') %>%
  #layer_dropout(rate = 0.3) %>%
  layer_dense(units = 24, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 4, activation = 'softmax')

summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history <- model %>% fit(
  X_train, y_train, 
  epochs = 200, batch_size = 128, 
  validation_split = 0.2
)
#require(devtools)
#install_version("keras", version = "2.2.2", repos = "http://cran.us.r-project.org")
## Test data (we need the one-hot version)
evaluate(model, X_test, y_test, verbose = 0)


