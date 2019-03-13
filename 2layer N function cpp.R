library(data.table)
library(tidyr)
library(progress)
library(Rcpp)
sourceCpp("C:/Users/yrhot/Desktop/perceptron/perceptron/test.cpp")


#####################################

# lr = learning rate / H = num of hidden node / K = num of output / least.loss / seed / label
train_nn <- function(dat,label,lr,max.iter,H,K,least.loss,seed=1,batch_size){     
  
  # browser()
  
  # setting  
  batch_idx <- sample(1:nrow(dat),batch_size)
  
  X <- dat[batch_idx,]
  t <- label[batch_idx,]
  N <- batch_size
  C <- ncol(X)

  set.seed(seed)
  W1 <- matrix(rnorm(C*H,0,0.1),nrow = C, ncol=H)  
  b1 <- matrix(0, nrow = 1, ncol = H)
  W2 <- matrix(rnorm(H*K,0,0.1),nrow = H, ncol=K)  
  b2 <- matrix(0, nrow = 1, ncol = K)
  
  summary_nn <- c()
  
  pb <- progress_bar$new(total = max.iter)
  
  # train
  idx <- 0
  loss <- 10000
  
  while( idx != max.iter & loss > least.loss ){

    idx <- idx + 1
    
    # forward
    H1 <- affine(X,W1,b1)
    a_H1 <- ReLU(H1)
    
    score <- affine(a_H1,W2,b2)
    a_score <- ReLU(score)
    
    prob <- softmax(a_score)
    loss <- cross_entropy(prob,t,N)   # loss
    
    
    # find accuracy
    temp <- ( apply(prob,1,which.max) == apply(t,1,which.max) ) %>% sum()
    accuracy <- temp / N
    
    summary_nn <- append(summary_nn,c(loss, accuracy))
    
    # backward
    da_score <- d_softmax_loss(prob,t)
    dscore <- d_ReLU(score,da_score)
    
    dW2 <- d_affine_W(a_H1,dscore)
    db2 <- colSums(dscore)
    da_H1 <- d_affine_X(W2,dscore)
    dH1 <- d_ReLU(H1,da_H1)
    
    
    dW1 <- d_affine_W(X,dH1)
    db1 <- colSums(dH1)
    dX <- d_affine_X(W1,dH1)
    
    # update
    W1 <- W1 - lr*dW1
    b1 <- b1 - lr*db1
    
    W2 <- W2 - lr*dW2
    b2 <- b2 - lr*db2
    
    # batch reset
    batch_idx <- sample(1:nrow(dat),batch_size)
    X <- dat[batch_idx,]
    t <- label[batch_idx,]
    
    pb$tick()
  }
  
  # save result
  summary_nn <- matrix(summary_nn, ncol = 2,byrow = T)
  colnames(summary_nn) <- c('loss', 'accuracy')
  
  result <- list(W1 = W1,b1 = b1,W2 = W2,b2 = b2,result = summary_nn, iter = idx)
  
  return(result)
}
#####################################
# predict function
predict_nn <- function(model, dat){
  H1 <- affine(dat,model$W1,model$b1) %>% ReLU()
  score <- affine(H1,model$W2,model$b2) %>% ReLU()
  
  prob <- softmax(score)
  
  Y <- apply(prob, 1, which.max)-1 # idx가 0이 아닌 1에서 시작하므로 -1
  
  return(Y)
}
#####################################
# read data
dat <- fread('C:/Users/yrhot/Downloads/mnist-in-csv/mnist_train.csv')
dim(dat)
class(dat)

dat <-  as.matrix(dat, nrow = 60000, ncol = 785)
head(dat)

dat_X <- dat[,-1]/max(dat[,-1])
dat_t <- dat[,1]

# one hot coding
temp <- matrix(0, nrow = length(dat_t), ncol = 10)
colnames(temp) <- 0:9
for(i in 1:length(dat_t)){
  temp[i,(dat_t[i]+1)] <- 1
}
dat_t <- temp

# training set
test_idx <- sample(1:60000, 10000)

train_x <- dat_X[-test_idx,]
train_t <- dat_t[-test_idx,]

test_x <- dat_X[test_idx,]
test_t <- dat_t[test_idx,]

# train
result <- train_nn(dat = train_x,
                   label = train_t,
                   lr = 0.00001,
                   max.iter = 10000,
                   H = 100,
                   K = 10,
                   least.loss = 0.1,
                   seed = 40,
                   batch_size = 200)


plot(result$result[,2], pch = 20, type = 'l')

# predict
label_nn <- predict_nn(result, test_x) + 1  # label이 one hot coding 이므로 다시 +1
((label_nn == apply(test_t,1,which.max)) %>% sum())/10000

# train
result2 <- train_nn(dat = train_x,
                   label = train_t,
                   lr = 0.02,
                   max.iter = 3000,
                   H = 80,
                   K = 10,
                   least.loss = 0.1,
                   seed = 40,
                   batch_size = 1000)

plot(result2$result[,2], pch = 20, type = 'l')

# predict
label_nn2 <- predict_nn(result2, test_x) + 1  # label이 one hot coding 이므로 다시 +1
((label_nn2 == apply(test_t,1,which.max)) %>% sum())/10000
