#####################################
# activation function
# ReLU function
ReLU <- function(X){
  temp <- X
  idx <- temp < 0
  temp[idx] <- 0
  return(temp)
}
d_ReLU <- function(X,dout){
  Y <- dout
  idx <- X <= 0
  Y[idx] <- 0
  return(Y)
}

# sigmoid function
sigmoid <- function(X){
  Y <- 1/(1+exp(-X))
  
  return( Y )
}
# sigmoid function backward
d_sigmoid <- function(Y,dout){
  dX <- dout * Y * (1-Y)
  return(dX)
}
#####################################
# softmax function
softmax <- function(X){
  c <- apply(X,1,max)
  exp_X <- exp(X-c)
  Y <- exp_X/apply(exp_X,1,sum)
  
  return( Y )
} 
# cross entropy
cross_entropy <- function(X,t,N){
  delta <- 1e-7
  Y <- -sum(t*log(X +delta)) / N
  
  return( Y )
}
# softmax and loss function backward
d_softmax_loss <- function(Y,t,dout=1){
  dX <- Y - t
}
#####################################
# affine
affine <- function(X,W,b){
  Y <- sweep(eigenMapMatMult(X,W),2,b)
  
  return( Y )
}
# affine backward
d_affine_X <- function(W,dout){
  return(eigenMapMatMult(dout,t(W)))
}
d_affine_W <- function(X,dout){
  return(eigenMapMatMult(t(X),dout))
}
#####################################