import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib
                images.append(img)
    return images

def find_projection_matrix(points_3D, points_2D):
    num_points = len(points_3D)
    A = np.zeros((2 * num_points, 12))

    for i in range(num_points):
        X, Y, Z = points_3D[i]
        x, y = points_2D[i]
        A[2*i] = [0, 0, 0, 0, -X, -Y, -Z, -1, y*X, y*Y, y*Z, y]
        A[2*i+1] = [X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x]

    # Compute ATA
    ATA = np.dot(A.T, A)

    # Compute eigenvalues and eigenvectors of ATA
    eigenvalues, eigenvectors = np.linalg.eig(ATA)

    # Find the eigenvector corresponding to the smallest eigenvalue
    min_eigenvalue_index = np.argmin(eigenvalues)
    P = eigenvectors[:, min_eigenvalue_index].reshape(3, 4)
    # print(P)
    return P

def get_projection_matrix():
# 2D points for Image 1
    points_2D_image1 = np.array([
            [1856, 1969], [2093, 1880], [2359, 1632], [2489, 1478], [2613, 1437],
            [1897, 2200], [2229, 1880], [2483, 1667], [2530, 1602], [2625, 1567],
            [1992, 2371], [2152, 2217], [2211, 2176], [2465, 1874], [2572, 1727],
            [2625, 1715], [1880, 2667], [2104, 2407], [2353, 2129], [2465, 1999],
            [2542, 1851], [2613, 1869], [2654, 1336]
    ])

        # 2D points for Image 2
    points_2D_image2 = np.array([
            [1710, 2034], [2041, 1934], [2503, 1691], [2745, 1520], [2982, 1466],
            [1758, 2241],[2272, 1963], [2716, 1721], [2810, 1644], [2994, 1608],
            [1876, 2413], [2136, 2265], [2225, 2229], [2674, 1934], [2875, 1786],
            [2952, 1780], [1746, 2679], [2053, 2454], [2467, 2170], [2668, 2058],
            [2798, 1916], [2946, 1886], [3053, 1366]
    ])

    points_3D = np.array([
            [8.5, 0.3, 11],[7.5, 4, 11],[8, 10.75, 11],[8.5, 15, 11],[7.5, 20.4, 11],[6, 0.6, 11],
            [6, 7, 11],[6, 15.25, 11],[6.25, 17.25, 11],[5.25, 21.7, 11], [3.4, 2, 11],[3.2, 5.5, 11],
            [2.2, 6.9, 11],[3, 14.8, 11],[3.5, 19.2, 11],[2.8, 21.5, 11],[0.8, 0.3, 11],[1.3, 4.4, 11],
            [1, 11.25, 11],[0.3, 14.75, 11],[2, 17.75, 11],[0.6, 22, 11],[8.5, 22.5, 11],
    ])

    P1 = find_projection_matrix(points_3D, points_2D_image1)
    P2 = find_projection_matrix(points_3D, points_2D_image2)

    return points_2D_image1, points_2D_image2, points_3D , P1, P2

def project_points(points_3D, projection_matrix):
    # Initialize the list for projected 2D points
    projected_2D_points = []

    # Iterate over each 3D point
    for point in points_3D:
        # Convert to homogeneous coordinates (add 1 as the fourth dimension)
        homogeneous_3D = [point[0], point[1], point[2], 1]

        # Project the point using the projection matrix
        projected_homogeneous_2D = [0, 0, 0]
        for i in range(3):  # Iterate over rows of the projection matrix
            for j in range(4):  # Iterate over columns of the projection matrix
                projected_homogeneous_2D[i] += projection_matrix[i][j] * homogeneous_3D[j]

        # Convert back to 2D by normalizing the x and y coordinates
        x_2D = projected_homogeneous_2D[0] / projected_homogeneous_2D[2]
        y_2D = projected_homogeneous_2D[1] / projected_homogeneous_2D[2]

        # Add the 2D point to the list
        projected_2D_points.append([x_2D, y_2D])
        # print(projected_2D_points)
    return projected_2D_points

def display_image_with_points(image, original_points_2D, projected_points_2D, title, subplot):
    plt.subplot(subplot)
    plt.imshow(image)

    # Original 2D points (red 'X')
    orig_x_coords = [point[0] for point in original_points_2D]
    orig_y_coords = [point[1] for point in original_points_2D]
    plt.scatter(orig_x_coords, orig_y_coords, marker="x", color="red", s=100)

    # Projected 2D points (green 'O')
    proj_x_coords = [point[0] for point in projected_points_2D]
    proj_y_coords = [point[1] for point in projected_points_2D]
    plt.scatter(proj_x_coords, proj_y_coords, marker="o", color="green", s=20)

    plt.title(title)
    plt.axis('off')


def main():
    global images, axs
    folder_path = "images"
    images = load_images_from_folder(folder_path)

    if len(images) < 2:
        print("The folder must contain at least two images.")
        return

    # 2D points for Image 1
    points_2D_image1, points_2D_image2, points_3D, P1, P2 = get_projection_matrix()

    # Project points onto each image
    points_image1 = project_points(points_3D, P1)
    points_image2 = project_points(points_3D, P2)

    # Display both original and projected points on each image
    plt.figure(figsize=(12, 6))
    display_image_with_points(images[0], points_2D_image1, points_image1, "Image 1", 121)
    display_image_with_points(images[1], points_2D_image2, points_image2, "Image 2", 122)
    plt.show()

if __name__ == "__main__":
    main()