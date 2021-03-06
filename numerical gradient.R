library(microbenchmark)

A <- matrix(rnorm(100000),ncol = 100,nrow=1000)
B <- matrix(rnorm(100000),ncol = 100,nrow=1000)


system.time(A%*%B)
system.time(mmult(A,B))
system.time(matmult(A,B))
system.time(eigenMapMatMult(A,B))
all.equal(A%*%B,eigenMapMatMult(A,B))

microbenchmark(d_ReLU(A,B),
               d_ReLU2(A,B))
all.equal(d_ReLU(A,B),
          d_ReLU2(A,B))

d_ReLU <- function(X,dout){
  Y <- ifelse(test = X > 0, yes = dout, no = 0)
  return(Y)
}

d_ReLU2 <- function(X,dout){
  Y <- dout
  idx <- X <= 0
  Y[idx] <- 0
  return(Y)
}

dat = train_x;
label = train_t;
lr = 0.000005;
max.iter = 10000;
H = 100;
K = 10;
least.loss = 0.1;
seed = 40;
batch_size = 2000


# numeric_gradient 함수 정의 
numeric_gradient <- function(f,X){
  # browser()
  dimm <- dim(X)
  pb <- progress_bar$new(total = (dimm[1]*dimm[2]) )
  grad <- matrix(0,nrow=dimm[1],ncol = dimm[2])
  
  h<- 7
  
  for(i in 1:dimm[1]){
    for(j in 1:dimm[2]){
      # f(x+h) 
      temp <- X[i,j]
      X[i,j] <- temp + h
      fxh1 <- f(X)
      
      # f(x-h)
      X[i,j] <- temp - h
      fxh2 <- f(X)
      
      grad[i,j] <- (fxh1 -fxh2)/(2*h)
      
      X[i,j] <- temp
      
      pb$tick()
    }
  }
  
  return(grad)
}

# loss 함수 정의 
loss <- function(X){
  H1 <- affine(X,W1,b1)
  a_H1 <- sigmoid(H1)
  
  score <- affine(a_H1,W2,b2)
  a_score <- sigmoid(score)
  
  prob <- softmax(a_score)
  loss <- cross_entropy(prob,t,10)
  
  return(loss)
}

# # 값 초기화 for test
# X <- matrix(rnorm(6,1,0.5),nrow=2)
# 
# W1 <- matrix(rnorm(6,1,0.5),nrow = 3, ncol=2)  
# b1 <- matrix(0, nrow = 1, ncol = 2)
# W2 <- matrix(rnorm(10,1,0.5),nrow = 2, ncol=10) 
# b2 <- matrix(0, nrow = 1, ncol = 10)
# 
# t <- matrix(c(1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0), nrow = 2, ncol=10)
# 
# numeric_gradient(forward,X)

class(iris)
dat_X <- as.matrix(iris[,-5],ncol=4)
dat_t <- iris[,5] %>% as.factor() %>% as.numeric()
temp <- matrix(0, nrow = length(dat_t), ncol = 3)
colnames(temp) <- c('setosa','versicolor','virginica')
for(i in 1:length(dat_t)){
  temp[i,dat_t[i]] <- 1
}
dat_t <- temp
result <- train_nn(dat = dat_X,
                   label = dat_t,
                   lr = 0.01,
                   max.iter = 10000,
                   H = 10,
                   K = 3,
                   least.loss = 0.1,
                   seed = 40,
                   batch_size = 50)
num_grad <- numeric_gradient(loss,X)
abs(num_grad-dX) %>% mean()

