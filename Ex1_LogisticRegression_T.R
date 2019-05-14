#Ex.1: Logistic Regression

set.seed(2019) #Set random seed for reproducibility 
library(dplyr)
###################################################################################
#Load data
setwd("/Users/cuauhtemoc/Documents/Machine Learning Course")  #set working directory
gendata<-read.csv('data/gendata.csv')   #read file

#Create feature matrix X with column of 1s.
X.mat<-as.matrix(cbind(rep(1,nrow(gendata)),gendata[,1:2]))

#Extract y and convert to (-1,1) 
gendata[,3][gendata[,3]==0] <- -1
y<- gendata[,3]

#Plot dataset
plot(X.mat[,2],X.mat[,3],col=as.factor(y))

##################################################################
#Ex.1.1: Logistic Regression: Gradient Descent

#Instructions:  Implement gradient descent for logistic regression and 
#               compare results against glm package 

#initialize betas
betas<-c(1,1,1)


#To do: Define Loss function
loss<-function(beta,X,y){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
  #Write code here!
  sum(log(1+exp(-y*(X%*%beta))))
  
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
}

#Evaluate loss function. (~210.7)
loss(betas,X.mat,y)

#To do: Define Gradient function
grad <- function(beta, X, y){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
  #Write code here!
  -t(X)%*%((y)/(1+exp(y*(X%*%beta))))
  
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
  }
#Test Gradient function (42.2, 99.3, -51.12)
grad(betas,X.mat,y)


#To do Define Gradient Descent
logGradDescent=function(X,y,beta_init,learning_rate,n_iterations){
  beta=beta_init
  loss_history=rep(0,n_iterations)
  for(k in 1:n_iterations){
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #Write beta update equation here!
    beta=beta-learning_rate * grad(beta,X,y)
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    loss_history[k]=loss(beta,X,y)
  }
  return(list(beta=beta,loss_history=loss_history))
}

#Test Gradient Descent
beta_init<-rnorm(3) #Initialize betas
learning_rate<-0.01  #Input learning rate
n_iterations<-300    #Number of iterations

results <- logGradDescent(X.mat,y,beta_init,learning_rate,n_iterations)

#Show Coefficients (Betas)
results$beta

#Plot Loss/Cost through gradient descent iterations
plot(results$loss_history,pch=20,type='b',main = 'Gradient Descent for Logistic Regression')

#Question: Does the loss function converges?

#Now compare results against r package for linear models (i.e. glm)

#y needs to be reformat as factor
gendata$Y<-as.factor(gendata$Y)

#Logistic regression in R
lr <- glm(Y~.,data=gendata,family = binomial)

#Show the coefficients
lr$coefficients

#Compare r coefficient results with our results

##################################################################
#Ex.1.2: Logistic Regression: Decision boundary

#To do: Plot decision boundaries

#Our results
m<- -results$beta[2]/results$beta[3]
b<- -results$beta[1]/results$beta[3]
plot(X.mat[,2],X.mat[,3],col=as.factor(y))
abline(b,m,col='blue')

#R package results
m2<- -lr$coefficients[2]/lr$coefficients[3]
b2<- -lr$coefficients[1]/lr$coefficients[3]
abline(b2,m2,col='green')

##################################################################
#Ex.1.3: Logistic Regression: Validation/Generalization

#To do: Separate data into train (70%) and test (30%)

#Total number of samples
N <-nrow(gendata)
#Define percentage of training samples
percentage_training <- 0.7
#Use sample function to get the indexes for training
train_index <-sample(N,size=round(percentage_training*N))
#Define indexes for testing
test_index  <- (1:nrow(gendata))[-train_index]

#To do: Train logistic regression on train data and report
#       accuracy for test data.

#Fit model with training data
lr <- glm(Y~.,data=gendata[train_index,],family = binomial)

#Predict on test data: use predict function with type='response' to get probabilities
lr.predprob<- predict(lr,newdata = gendata[test_index,],type = 'response')

#Extract class from predicted probabilities
lr.pred <- ifelse(lr.predprob>0.5,'1','-1')

#Construct confusion matrix
lr.matrix <- table(lr.pred,gendata[test_index,'Y'])

#Compute accuracy
lr.acc <- sum(diag(lr.matrix))/sum(lr.matrix)
lr.acc

#Question:  If you resample the train and test datasets, 
#           does the accuracy change? Why?

##################################################################
#Ex.1.3: Logistic Regression: Cross Validation

#To do: Implement Monte Carlo Cross Validation to report a robust
#       accuracy metric

nexp<-100 #number of Monte Carlo experiments
acc_list <- rep(0,nexp) #vector to store accuracy results for each experiment

#Loop
for(i in 1:nexp){
  #resample train and test index
  train_index <-sample(N,size=round(percentage_training*N))
  test_index  <- (1:nrow(gendata))[-train_index]
  
  #fit model on training data
  lr <- glm(Y~.,data=gendata[train_index,],family = binomial)
  
  #predict on test data
  lr.predprob<- predict(lr,newdata = gendata[test_index,],type = 'response')
  lr.pred <- ifelse(lr.predprob>0.5,'1','-1')
  
  #Construct confusion matrix
  lr.matrix <- table(lr.pred,gendata[test_index,'Y'])
  
  #Compute accuracy
  lr.acc <- sum(diag(lr.matrix))/sum(lr.matrix)
  
  #Store accuracy
  acc_list[i]<-lr.acc
}

#Report accuracy average
mean(acc_list)

#Plot distribution of accuracy
boxplot(acc_list)


##################################################################
#Ex.1.4: Feature Engineering: Basis Functions

#To do: Explore the effects of a polynomial basis function
#       on the decision boundary 

maxdegree<-15  #Maximum degree of polynomial
gendata_ext<-gendata #Create new data frame to store extended design matrix X

#Define a and b for plotting
a <- seq(-4, 4, length.out = 100)
b <- seq(-4, 4, length.out = 100)
#g<- meshgrid(a,b)
betas_hist<-matrix(0,maxdegree, 2*maxdegree+1)
for(k in 2:maxdegree){
  gendata_ext<-cbind(gendata_ext,gendata_ext$X1^k,gendata_ext$X2^k)
  last<-length(gendata_ext)
  names(gendata_ext)[(last-1):last]<-c(paste('X1^',k),paste('X2^',k))
  lr_ext <- glm(Y~.,data=gendata_ext,family = binomial)
  betas<-lr_ext$coefficients #Save list of betas
  betas[is.na(betas)]<-0 #na to 0
  f1 <- function(x, x2){as.numeric(betas[1]+sum(sapply(seq_along(1:k), function(k) betas[2*k]*x^k +betas[2*k+1]*x2^k)))}
  vf1<-Vectorize(f1)
  z<-outer(a,b,vf1)
  plot(X.mat[,2],X.mat[,3],col=as.factor(y))
  contour(a, b, z, levels = 0, add = TRUE, drawlabels = FALSE,lwd=2,col = 'blue')
  #Store betas
  betas_hist[k,]<-c(betas,rep(0,(2*(maxdegree-k))))
}


#To do: Extract the value of the coefficients for every model
#       What happens to the value of the coefficients?

betas_hist<-data.frame(betas_hist)

##################################################################
#Ex.1.5: Choose the best polynomial degree with MC Cross Validation


#1 Experiment ---------------------------------------
N <-nrow(gendata) #total number of rows
percentage_training <- 0.7 #percentage of training samples
train_index <-sample(N,size=round(percentage_training*N)) #sample train index
test_index  <- (1:nrow(gendata))[-train_index]

maxdegree<-15 #Maximum polynomial degree to explore
acc_history=rep(0,maxdegree) #Vector to store accuracies
gendata_ext<-gendata # Copy gendata into gendata_ext

#Loop for polynomial degree exploration
for(k in 2:maxdegree){
  gendata_ext<-cbind(gendata_ext,gendata_ext$X1^k,gendata_ext$X2^k)
  last<-length(gendata_ext)
  names(gendata_ext)[(last-1):last]<-c(paste('X1^',k),paste('X2^',k))
  lr_ext <- glm(Y~.,data=gendata_ext[train_index,],family = binomial)
  pred_prob<-predict(lr_ext,newdata = gendata_ext[test_index,],type = 'response')
  pred <- ifelse(pred_prob>0.5,'1','-1')
  lrext.matrix<-table(pred,gendata[test_index,'Y'])
  lrext.acc<- sum(diag(lrext.matrix))/sum(lrext.matrix)
  acc_history[k]<-lrext.acc
}

plot(acc_history[2:maxdegree]) #Plot accuracies
which.max(acc_history[1:maxdegree]) #Which degree had the highest accuracy

#MC Cross Validation ---------------------------------------
maxdegree=15 #max degree of polynomial exploration
nexp=100 #number of Monte Carlo experiments
N <-nrow(gendata) #total number of rows
percentage_training <- 0.7 #percentage of training samples
accs=matrix( rep( 0, nexp*maxdegree), nrow = nexp) #define matrix to store accuracies

for(n in 1:nexp){
  acc_history=rep(0,maxdegree) #vector to store acc
  gendata_ext<-gendata
  #Separate data into training and testing
  train_index <-sample(N,size=round(percentage_training*N)) #sample train index
  test_index  <- (1:nrow(gendata))[-train_index]
  
  for(k in 1:maxdegree){
    gendata_ext<-cbind(gendata_ext,gendata_ext$X1^k,gendata_ext$X2^k)
    last<-length(gendata_ext)
    names(gendata_ext)[(last-1):last]<-c(paste('X1^',k),paste('X2^',k))
    lr_ext <- glm(Y~.,data=gendata_ext[train_index,],family = binomial)
    pred_prob<-predict(lr_ext,newdata = gendata_ext[test_index,],type = 'response')
    pred <- ifelse(pred_prob>0.5,'1','-1')
    lrext.matrix<-table(pred,gendata[test_index,'Y'])
    lrext.acc<- sum(diag(lrext.matrix))/sum(lrext.matrix)
    acc_history[k]<-lrext.acc
    accs[n,k]<-acc_history[k]
  }
}

#Extract average accuracy by polynomial degree
accs.means=rep(0,maxdegree)
for(i in 1:maxdegree){
  accs.means[i]<-mean(accs[,i])
}

#Plot average accuracy by polynomial degree
plot(accs.means[1:maxdegree])

#Which polynomial degree has the highest accuracy?
which.max(accs.means[1:maxdegree])
max(accs.means[1:maxdegree])

#Boxplots
boxplot(accs[,1],accs[,2],accs[,3],accs[,13],accs[,15])

#Question: Which polynomial order generalizes better?
