###
# Introduction to Data Science Homework Assignment # 1
# Student: Alan Fernandez, aefernandez@wpi.edu
# Date: 08/29/18
# Course: DS501, Introduction to Data Science (Grad Level)
# Worcester Polytechnic Institute (WPI), Worcester, MA
###

import numpy as np

from problem3 import random_walk

#-------------------------------------------------------------------------
'''
    Problem 4: Solving sink-node problem in PageRank
    In this problem, we implement the pagerank algorithm which can solve the sink node problem.
    You could test the correctness of your code by typing `nosetests test4.py` in the terminal.
'''

#--------------------------
def compute_S(A):
    '''
        compute the transition matrix S from addjacency matrix A, which solves sink node problem by filling the all-zero columns in A.
        S[j][i] represents the probability of moving from node i to node j.
        If node i is a sink node, S[j][i] = 1/n.
        Input: 
                A: adjacency matrix, a (n by n) numpy matrix of binary values. If there is a link from node i to node j, A[j][i] =1. Otherwise A[j][i]=0 if there is no link.
        Output: 
                S: transition matrix, a (n by n) numpy matrix of float values.  S[j][i] represents the probability of moving from node i to node j.
    The values in each column of matrix S should sum to 1.
    '''
#########################################
## INSERT YOUR CODE HERE
    #print(A)
    # sum of each column of A
    column_sum = A.sum(axis = 0)

    # create a diagonal matrix
    sum_vector = column_sum.getA1()
    D = np.diag(np.where(sum_vector == 0, 1, sum_vector))
    #print(D)
    # normalize each column of A

    # Invert the diagonal matrix to divide the adjacency matrix by the sum of the columns.
    D = np.linalg.inv(D)

    # Multiply the matrices to execute the division. This returns the transition matrix.
    S = A * D

    # Determine the number of elements in each column
    n = S.shape[0]

    # Iterate through each column, determine if its a sink node, and replace with all equal probabilities.
    for index, column in enumerate(S):
        if ~S[:,index].any(axis=0).any():
            S[:,index] = 1/n

    return S
#########################################

#--------------------------
def pagerank_v2(A):
    ''' 
        A simplified version of PageRank algorithm, which solves the sink node problem.
        Given an adjacency matrix A, compute the pagerank score of all the nodes in the network. 
        Input: 
                A: adjacency matrix, a numpy matrix of binary values. If there is a link from node i to node j, A[j][i] =1. Otherwise A[j][i]=0 if there is no link.
        Output: 
                x: the ranking scores, a numpy vector of float values, such as np.array([[.3], [.5], [.7]])
    '''

    # Initialize the score vector with all one values
    num_nodes, _ = A.shape 
    x_0 =  np.asmatrix(np.ones((num_nodes,1))) 

    # compute the transition matrix from adjacency matrix
    S = compute_S(A)
    # random walk
    x, n_steps = random_walk(S,x_0)

    return x

