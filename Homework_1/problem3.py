###
# Introduction to Data Science Homework Assignment # 1
# Student: Alan Fernandez, aefernandez@wpi.edu
# Date: 08/29/18
# Course: DS501, Introduction to Data Science (Grad Level)
# Worcester Polytechnic Institute (WPI), Worcester, MA
###

import numpy as np
#-------------------------------------------------------------------------
'''
    Problem 3: PageRank algorithm (version 1) 
    In this problem, we implement a simplified version of the pagerank algorithm, which doesn't consider about sink node problem or sink region problem.
    You could test the correctness of your code by typing `nosetests -v test3.py` in the terminal.
'''

#--------------------------
def compute_P(A):
    '''
        compute the transition matrix P from addjacency matrix A. P[j][i] represents the probability of moving from node i to node j.
        Input: 
                A: adjacency matrix, a (n by n) numpy matrix of binary values. If there is a link from node i to node j, A[j][i] =1. Otherwise A[j][i]=0 if there is no link.
        Output: 
                P: transition matrix, a (n by n) numpy matrix of float values.  P[j][i] represents the probability of moving from node i to node j.
    The values in each column of matrix P should sum to 1.
    '''
    #########################################
    ## INSER YOUR CODE HERE

    # sum of each column of A
    column_sum = A.sum(axis = 0)

    # create a diagonal matrix
    D = np.diag(column_sum.getA1())

    # normalize each column of A

    # Invert the diagonal matrix to divide the adjacency matrix by the sum of the columns.
    D = np.linalg.inv(D)

    # Multiply the matrices to execute the division. This returns the transition matrix.
    P = A * D

    #########################################
    return P


#--------------------------
def random_walk_one_step(P, x_i):
    '''
        compute the result of one step random walk.
        Input:
                P: transition matrix, a (n by n) numpy matrix of float values.  P[j][i] represents the probability of moving from node i to node j.
                x_i: pagerank scores before the i-th step of random walk. a numpy vector of shape (n by 1).
        Output:
                x_i_plus_1: pagerank scores after the i-th step of random walk. a numpy vector of shape (n by 1).
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    x_i_plus_1 = np.dot(P, x_i)


    #########################################
    return x_i_plus_1


#--------------------------
#--------------------------
def random_walk(P, x_0, max_steps=10000):
    '''
        compute the result of multiple-step random walk. The random walk should stop if the score vector x no longer change (converge) after one step of random walk, or the number of iteration reached max_steps.
        Input:
                P: transition matrix, a (n by n) numpy matrix of float values.  P[j][i] represents the probability of moving from node i to node j.
                x_0: the initial pagerank scores. a numpy vector of shape (n by 1).
                max_steps: the maximium number of random walk steps. an integer value.
        Output:
                x: the final pagerank scores after multiple steps of random walk. a numpy vector of shape (n by 1).
                n_steps: the number of steps actually used (for example, if the vector x no longer changes after 3 steps of random walk, return the value 3.
        Hint: you could use np.allclose(x, previous_x) function to determine when to stop the random walk iterations.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    x_old = x_0
    n_steps = 0
    while n_steps < max_steps:
        n_steps += 1
        x_new = random_walk_one_step(P, x_old)
        if np.allclose(x_new, x_old):
            break
        x_old = x_new

    #########################################

    return x_old, n_steps


#--------------------------
def pagerank_v1(A):
    ''' 
        A simplified version of PageRank algorithm.
        Given an adjacency matrix A, compute the pagerank score of all the nodes in the network. 
        Here we ignore the issues of sink nodes and sink regions in the network.
        Input: 
                A: adjacency matrix, a numpy matrix of binary values. If there is a link from node i to node j, A[j][i] =1. Otherwise A[j][i]=0 if there is no link.
        Output: 
                x: the ranking scores, a numpy vector of float values, such as np.array([[.3], [.5], [.7]])
    '''

    # compute the transition matrix from adjacency matrix
    P = compute_P(A)

    # initialize the score vector with all one values
    num_nodes, _ = A.shape # get the number of nodes (n)
    x_0 =  np.ones((num_nodes,1)) # create an all-one vector of shape (n by 1)
    
    # random walk
    x, n_steps = random_walk(P, x_0)

    return x

