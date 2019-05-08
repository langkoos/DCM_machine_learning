#Ex.1: Olympic Marathon Times (Men)

#libraries
library(dplyr) 
library(matlib)
#set working directory
setwd("/Users/cuauhtemoc/Documents/Machine Learning Course") 
#read file
olympic <-read.csv('data/olympicMarathonTimes.csv',header = FALSE)
#name columns
names(olympic)<-c('year','times')
#plot
plot(olympic[c('year','times')])

#setup X.matrix
X0<-rep(1,nrow(olympic))
X1<-olympic['year']
X.mat <-data.matrix(cbind(X0,X1))
#y (response vector)
y <- data.matrix(olympic['times'])

#Remember:
#Linear Model y=BX
#Objective Function -> argmin((y-XB)^2)
#Closed form solution-> B_hat = (X.T X)^-1 X.T y 

#To do: Estimate linear regression coefficients (B)
B <- solve(t(X.mat) %*% X.mat) %*% t(X.mat) %*%y
B0 <- B[1]
B1 <- B[2]

#To do: Plot data and fitted line
plot(olympic[c('year','times')])
abline(B0, B1,col='red')

#To do: Compare result against built package
lr <- lm(times~. ,data = olympic)
coeff<-coefficients(lr)
abline(coeff[1], coeff[2],col='blue')


##################################################################
#Ex.1.2: Generalization / Model Validation

#Separate data into training and testing
N <-nrow(olympic) #total number of rows
percentage_training <- 0.8 #percentage of training samples
train_index <-sample(N,size=round(percentage_training*N)) #sample train index
test_index  <- (1:nrow(olympic))[-train_index]

#Fitting training data
lr <- lm(times~. ,data = olympic[train_index,])
coeff<-coefficients(lr)
plot(olympic[c('year','times')])
abline(coeff[1], coeff[2],col='brown')

#Predicting
lr.pred <- predict(lr,newdata = olympic[test_index,])

#Measure mean squared error (MSE) of testing data
lr.mse<-mean((olympic[test_index,'times'] - lr.pred)^2)
lr.mse

#Use Cross Validation to report a robust MSE
nexp<-100 #number of Monte Carlo experiments
mse_list <- rep(0,nexp)
for(i in 1:nexp){
  #resample train and test index
  train_index <-sample(N,size=round(percentage_training*N)) #sample train index
  test_index  <- (1:nrow(olympic))[-train_index]
  #fit model on training data
  lr <- lm(times~. ,data = olympic[train_index,])
  #predict on test data
  lr.pred <- predict(lr,newdata = olympic[test_index,])
  #calculate MSE
  mse_list[i]<-mean((olympic[test_index,'times'] - lr.pred)^2)
}

mean(mse_list)
boxplot(mse_list)

##################################################################
#Ex.1.3: Basis Functions: Polynomial Regression
#Implement a polynomial regression from degree=1 to degree 15 and plot fitted line

maxdegree=15 #max degree of polynomial exploration
olympic.pol<-olympic[c('year','times')] #copy dataset to olympic.pol

#Standarize y 
olympic.pol$times<-(olympic$times - mean(olympic$times))/sqrt(var(olympic$times))
#Normalize x in the range -1 to 1
center<-(max(olympic$year)+min(olympic$year))/2
span<-max(olympic$year)-min(olympic$year)
olympic.pol$year<- 2*(olympic$year - center)/span

#Create x for plotting results
xplot<- seq(from=-1, to=1, by=0.001)
xplot<-as.data.frame(x=xplot) #to data.frame
names(xplot)<-'year'

for(k in 2:maxdegree){
  olympic.pol<-cbind(olympic.pol,olympic.pol[,'year']^k) #add polynomial column
  names(olympic.pol)[names(olympic.pol) == last(names(olympic.pol))] <- paste('exp',toString(k)) #rename new column
  lr <- lm(times~. ,data = olympic.pol)#Fit model
  xplot <- cbind(xplot,xplot[,'year']^k) #expand xplot to calculate yplot
  names(xplot)<-names(select(olympic.pol,-times)) #name xplot same as olympic.pol
  yplot<-predict(lr,newdata = xplot) #predict yplot
  #Plot
  plot(olympic.pol[c('year','times')])
  lines(xplot[,1],yplot,col='red')
}

##################################################################
#Ex.1.4: Basis Functions: MSE on test data
#To do: Implement polynomial regression on training data and calculate MSE on test data

maxdegree=15 #max degree of polynomial exploration
olympic.pol<-olympic[c('year','times')] #copy dataset to olympic.pol

#Standarize y 
olympic.pol$times<-(olympic$times - mean(olympic$times))/sqrt(var(olympic$times))
#Normalize x in the range -1 to 1
center<-(max(olympic$year)+min(olympic$year))/2
span<-max(olympic$year)-min(olympic$year)
olympic.pol$year<- 2*(olympic$year - center)/span

mse_history=rep(0,maxdegree) #vector to store mse
for(k in 2:maxdegree){
  olympic.pol<-cbind(olympic.pol,olympic.pol[,'year']^k)
  names(olympic.pol)[names(olympic.pol) == last(names(olympic.pol))] <- paste('exp',toString(k))
  lr <- lm(times~. ,data = olympic.pol[train_index,])
  y_hat<-predict(lr,newdata = olympic.pol[test_index,])
  #MSE
  mse_history[k]<- mean((y_hat - olympic.pol[test_index,'times'])^2)
}

mean(mse_history[2:maxdegree])
boxplot(mse_history[2:maxdegree])

##################################################################
#Ex.1.5: Basis Functions: Cross Validation 
#To do: Use Cross Validation to calculate average MSE of polynomial regression

maxdegree=15 #max degree of polynomial exploration
nexp=10 #number of Monte Carlo experiments
mses=matrix( rep( 0, nexp*maxdegree), nrow = nexp)
for(n in 1:nexp){
  olympic.pol<-olympic[c('year','times')]
  olympic.pol$times<-(olympic$times - mean(olympic$times))/sqrt(var(olympic$times))
  center<-(max(olympic$year)+min(olympic$year))/2
  span<-max(olympic$year)-min(olympic$year)
  olympic.pol$year<- 2*(olympic$year - center)/span
  mse_history=rep(0,maxdegree)
  #Resample train and test index
  train_index <-sample(N,size=round(percentage_training*N))
  test_index  <- (1:nrow(olympic))[-train_index]
  for(k in 1:maxdegree){
    olympic.pol<-cbind(olympic.pol,olympic.pol[,'year']^k)
    names(olympic.pol)[names(olympic.pol) == last(names(olympic.pol))] <- paste('exp',toString(k))
    lr <- lm(times~. ,data = olympic.pol[train_index,])
    y_hat<-predict(lr,newdata = olympic.pol[test_index,])
    #MSE
    mse_history[k]<-mean((y_hat - olympic.pol[test_index,'times'])^2)
    mses[n,k]<-mse_history[k]
  }
}

#Extract average of MSE by polynomial degree
mses.means=rep(0,maxdegree)
for(i in 1:maxdegree){
mses.means[i]<-mean(mses[,i])
}
plot(mses.means[1:maxdegree])
which.min(mses.means[1:maxdegree])
min(mses.means[1:maxdegree])
#boxplot(mses[,2],mses[,3],mses[,4],mses[,5],mses[,6],mses[,7],mses[,8],mses[9,],mses[,10])
boxplot(mses[,1],mses[,2],mses[,3])
