import numpy as np
def objective(y, u, v, l):
    error_sum = (1/2)* np.linalg.norm((y-np.matmul(u, v.transpose())), ord=2)
    reg_sum = (l/2)*np.linalg.norm(u, ord=2)*np.linalg.norm(v, ord=2)
    print("Error sum is: "+str(error_sum)+", regularization sum is: "+reg_sum)
    return error_sum + reg_sum



u = np.matrix([6, 0, 3, 6]).transpose()
v = np.matrix([4, 2, 1]).transpose()

l=1
k=1

x0 = np.matmul(u, v.transpose())

y = np.matrix([[5,12,7], [0,2,0], [4,6,3], [24,3,6]])

error_term = (1/2)*np.power(y-x0, 2)
total_error = np.sum(error_term)

reg_term = (1/2)*(np.matmul(u.transpose(),u) + np.matmul(v.transpose(),v))

print("error term total: "+str(error_term)+", reg term total: "+str(reg_term))