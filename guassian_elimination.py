# The following code is an implementation of Guassian Elimination for Solving a system of linear equations
# For detailed explanation. Please refer to this video by Linear Algebra O.G Glibert Strang; https://www.youtube.com/watch?v=QVKj3LADCnA&list=PLE7DDD91010BC51F8&index=2
import numpy as np

class GaussianElimination:
    @staticmethod
    def __forward_elimination(A,b):
        m,n = A.shape
        augmented_A = np.c_[A,b]
        j = 0
        for i in range(m - 1):
            pivot = augmented_A[i][j]
            if pivot == 0:
                found = False
                for k in range(i+1,m):
                    if augmented_A[k][j] != 0:
                        temp = augmented_A[i].copy()
                        augmented_A[i] = augmented_A[k].copy()
                        augmented_A[k] = temp.copy()
                        found = True
                        break 
                if found == False:
                    raise Exception("The Matrix A is singular. There is no unique Solution")
                else:
                    pivot = augmented_A[i][j]
            for k in range(i+1,m):
                target = augmented_A[k][j]
                multiplier = target / pivot
                augmented_A[k] = augmented_A[k] - multiplier * augmented_A[i]
            j += 1

        #new_A is a triangular matrix
        new_A = augmented_A[:,0:n]

        new_b = augmented_A[:,-1]
        return new_A, new_b
    
    @staticmethod
    def __backward_substitution(new_A, new_b):
        m,n = new_A.shape
        x = [None]*3
        for i in range(m-1,-1,-1):
            s = 0
            for k in range(m -1 , i, -1):
                s += new_A[i][k] * x[k]
            x[i] = ( new_b[i] - s ) / new_A[i][i]
        return x 
                 
    @staticmethod
    def solve(A,b):
        A = np.array(A)
        b = np.array(b)
        m,n = A.shape
        if m != n:
            raise Exception("Matrix A should be square.")
        if m != len(b):
            raise Exception("Number of unknown variables should be equal to the length of b")
        new_A, new_b = GaussianElimination.__forward_elimination(A,b)
        x = GaussianElimination.__backward_substitution(new_A, new_b)

        return x
   

if __name__ == "__main__":
    # find x, y and z for the system of linear equations
    # x + y + z = 3
    # 2x + 2y + 5z = 5
    # 4x + 6y + 8z = 8
    
    A = np.array([[1,1,1],[2,2,5],[4,6,8]])
    b = np.array([[3],[5],[8]])
    x = GaussianElimination.solve(A,b)
    print(x)


