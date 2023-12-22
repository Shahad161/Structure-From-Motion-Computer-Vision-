from matplotlib.pylab import eig
import numpy as np
from step1 import get_projection_matrix
from step2 import get_matched_points

def skew_symmetric(x):
    return np.array([[0, -1, x[1]], [1, 0, -x[0]], [-x[1], x[0], 0]])

def linear_triangulate(P1, P2, x1, x2):

    A = np.vstack((np.dot(skew_symmetric(x1), P1), np.dot(skew_symmetric(x2), P2)))

    # Compute A^TA
    ATA = np.dot(A.T, A)

    # Compute the eigenvalues and eigenvectors of A^TA
    eigenvalues, eigenvectors = np.linalg.eig(ATA)

    w,v= eig(np.matmul(np.transpose(A), A))
    # index of minumum eigenvalue
    index_min_eigenvalue = list(w).index(w.min())

# get eigenvector that has minimum eigenvalue
    X = v[:,index_min_eigenvalue]

    # Find the eigenvector corresponding to the smallest eigenvalue
    # min_eigenvalue_index = np.argmin(eigenvalues)
    # X = eigenvectors[:, min_eigenvalue_index]

    # Dehomogenize (if the last component is not 1, divide all by the last component)
    X = X / X[3]
    return X[:3]




if __name__ == "__main__":

    points_image1 = [(1856, 1969), (2093, 1880), (2359, 1632)]
    points_image2 = [(1710, 2034), (2041, 1934), (2503, 1691)]

    _, _, _, P1, P2 = get_projection_matrix()

    # Triangulate the 3D point
    for point1, point2 in zip(points_image1, points_image2):
        X = linear_triangulate(P1, P2, point1, point2)
        print("The 3D location of the point is:", X)

