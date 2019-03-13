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

X <- matrix(rnorm(6,1,0.5),nrow=2)

W1 <- matrix(rnorm(6,1,0.5),nrow = 3, ncol=2)  
b1 <- matrix(0, nrow = 1, ncol = 2)
W2 <- matrix(rnorm(10,1,0.5),nrow = 2, ncol=10) 
b2 <- matrix(0, nrow = 1, ncol = 10)

t <- matrix(c(1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0), nrow = 2, ncol=10)

dimm <- dim(X)

grad <- matrix(0,nrow=dimm[1],ncol = dimm[2])

h<-1e-4

for(i in 1:dimm[1]){
  for(j in 1:dimm[2]){
    # f(x+h) 
    temp <- X[i,j]
    X[i,j] <- temp + h
    fxh1 <- forward(X)
    
    # f(x-h)
    X[i,j] <- temp - h
    fxh2 <- forward(X)
    
    grad[i,j] <- (fxh1 -fxh2)/(2*h)
    
    X[i,j] <- temp
  }
}


forward <- function(X){
  H1 <- affine(X,W1,b1)
  a_H1 <- ReLU(H1)
  
  score <- affine(a_H1,W2,b2)
  a_score <- ReLU(score)
  
  prob <- softmax(a_score)
  loss <- cross_entropy(prob,t,10)
  
  return(loss)
}
forward(X)
