# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 23:30:07 2017

@author: Efe
"""
#GEC for MAP estimation of graph matching

import numpy as np
import matplotlib.pyplot as plt

def d(Q):
    return(Q);
    #return(np.ones((Q.shape))*np.mean(Q) );

def gec(A, B, aux, tol):
    N = A.shape[0];
    sA, UA = np.linalg.eigh(A)
    sB, UB = np.linalg.eigh(B)
    UA2 = UA*UA;
    UBT2 = (UB.T)*(UB.T);
    sAsB =  sA[:,None]*sB[None,:];   
        
    Jn = np.ones((N,N))/N;
    num_facnodes = 3;    
    gammap = [];
    for i in range(num_facnodes):
        gammap.append(Jn*N);
    rp = [];
    for i in range(num_facnodes):
        rp.append(Jn);        
        
    gammam = gammap[:];    
    rm = rp[:];
    x = rp[:];
    change = float("inf");
    iter = 0;   
    eta = [None]*num_facnodes;
    while change>tol:
        xold = x[:]; 
        
        #Factor node 1
        scale = gammap[0]/(gammap[0]+2*sAsB );
        x[0] = UA.dot(UA.T.dot(rp[0]).dot(UB)*scale ).dot(UB.T);
        Q = UA2.dot(scale).dot(UBT2)/gammap[0];             
        eta[0] = 1/d(Q);
        gammam[0] = eta[0] - gammap[0];
        print(gammam[0]>0)
        rm[0] = eta[0]*x[0]/(num_facnodes-1) - gammap[0]*rp[0];        
        
        #factor node 2
        x[1] = Jn + rp[1] - np.dot(Jn,rp[1])- np.dot(rp[1],Jn)+np.sum(rp[1])*Jn/N;        
        Q = (N-2+1/N)*Jn/gammap[1];       
        eta[1] = 1/d(Q);        
        gammam[1] = eta[1] - gammap[1];
        rm[1] = eta[1]*x[1]/(num_facnodes-1) - gammap[1]*rp[1];  

        #factor node 3 
        x[2] = (rp[2] + np.absolute(rp[2]))/2;
        Q = (x[2]>0)/gammap[2] + 1e-3*(1-x[2]>0);        
        eta[2] = 1/d(Q);
        gammam[2] = eta[2] - gammap[2];
        rm[2] = eta[2]*x[2]/(num_facnodes-1) - gammap[2]*rp[2];        
        
        #variable node
        eta0 = sum(gammam)/(num_facnodes-1);                
        gammap = eta0 - gammam;
        rmsum = sum(rm);
        for i in range(num_facnodes):
            rp[i] = (rmsum - rm[i])/gammap[i];           
       
        change = 0;
        for i in range(len(x)):
            change = change + np.linalg.norm(x[i]-xold[i], 'fro');            
        iter = iter + 1; 
        print (change)
        print(iter)
        
    return (1-np.trace(np.dot(x[0].T, aux))/N );

def main():
    Nvec = np.linspace(10, 10, 1);
    experiments = 1;
    p = 0.5;
    rvec = np.linspace(0, 0, 1);
    iterations = [len(Nvec), len(rvec), experiments];
    gec_better_sols = np.zeros(iterations);
    gec_error = np.zeros(iterations);
        
    for ind in range(np.prod(iterations)):
        [Nind, rind,expind] = np.unravel_index(ind, iterations);
        N = int(Nvec[Nind]);
        r = rvec[rind];
        print([N,r,expind]);
        
        X1 = np.zeros((N,N));
        X2 = np.zeros((N,N));
        
        p11 = 1-r;
        p01 = r;
        
        #correlated ER
        #p11 = r+1 -r/p;
        
        for i in range(N):
            for j in range(i+1,N):
                if np.random.rand() < p :
                    X1[i,j] = 1;
                    if np.random.rand() < p11 :
                        X2[i,j] = 1;
                else:
                    if np.random.rand() < p01 :
                        X2[i,j] = 1;
        X1 = X1 + X1.T;
        X2 = X2 + X2.T;
        
        perm = np.random.permutation(N); 
        aux = np.zeros((N,N));
        for i in range(N):
            aux[i, perm[i]] = 1;
        X2 = np.dot(aux.T, np.dot(X2, aux));
        tol = 1;
        
        error = gec(X1, X2, aux, tol);
        gec_error[Nind, rind,expind] = error;
        
    print(gec_error);    
    gec_better_sols = np.sum(gec_better_sols, axis = 2);
    #plt.imshow(rvec, Nvec, np.sum(gec_error, axis = 2)/(experiments - gec_better_sols));
    
if __name__ == "__main__" :
    main()
