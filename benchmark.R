library(microbenchmark)
library(data.table)
library(tidyr)
library(progress)
library(Rcpp)
library(RcppArmadillo)
sourceCpp("C:/Users/yrhot/Desktop/cpp/test.cpp")
sourceCpp("C:/Users/yrhot/Desktop/cpp/test1.cpp")
#####################################
A <- matrix( rnorm(100000), ncol = 100)
B <- matrix( rnorm(100000), nrow = 100)
system.time(A%*%B)
system.time(armaMatMult(A,B))
system.time(eigenMatMult(A,B))
system.time(eigenMapMatMult(A,B))
system.time(StrassenAlgorithm(A,B))

microbenchmark("a"=A%*%B,
          "b"=armaMatMult(A,B),
          "c"=eigenMatMult(A,B),
          "d"=eigenMapMatMult(A,B),
          "e"=.mm(A,B)
          )
all.equal(A%*%B,eigenMapMatMult(A,B))
