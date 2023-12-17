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

# Global variables
current_image_index = 0  # Start with the first image
matched_points = []      # Store pairs of matched points

# Function to display both images side by side
def display_images_side_by_side(images):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(images[0])
    axs[0].set_title('Image 1')
    axs[0].axis('off')

    axs[1].imshow(images[1])
    axs[1].set_title('Image 2')
    axs[1].axis('off')

    return fig, axs

# Event handler for mouse clicks
def onclick(event):
    global current_image_index, matched_points, axs
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)

        # Append point and switch to the other image
        if current_image_index == 0:
            matched_points.append([(x, y)])
            current_image_index = 1
        elif current_image_index == 1:
            matched_points[-1].append((x, y))
            current_image_index = 0

        # Redraw points on both images
        for i, ax in enumerate(axs):
            ax.clear()
            ax.imshow(images[i])
            ax.set_title(f'Image {i + 1}')
            ax.axis('off')
            for point_pair in matched_points:
                if i == 0 and len(point_pair) > 0:
                    ax.plot(point_pair[0][0], point_pair[0][1], 'ro')  # Red dot for first image
                elif i == 1 and len(point_pair) == 2:
                    ax.plot(point_pair[1][0], point_pair[1][1], 'ro')  # Red dot for second image
        plt.draw()


def match_features(img1, img2, points1, points2):
    if len(points1) != len(points2):
        print("The number of points in both images should be the same.")
        return None

    # Create a blank image that fits both images
    height = max(img1.shape[0], img2.shape[0])
    width = img1.shape[1] + img2.shape[1]
    composite_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Place the first image on the left
    composite_image[:img1.shape[0], :img1.shape[1], :] = img1

    # Place the second image on the right
    composite_image[:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1], :] = img2

    plt.imshow(composite_image)
    ax = plt.gca()

    # Draw lines between matched points
    for (x1, y1), (x2, y2) in zip(points1, points2):
        ax.annotate("", xy=(x1, y1), xycoords='data',
                    xytext=(x2 + img1.shape[1], y2), textcoords='data',
                    arrowprops=dict(arrowstyle="-", lw=1, color="red"))

    plt.axis("off")
    plt.show()
    matched_pairs = list(zip(points1, points2))
    return matched_pairs

def get_focal_lengths_from_exif(image_path):

    img = Image.open(image_path)
    exif_data = img._getexif()
    focal_length = None

    if exif_data:
        for tag, value in exif_data.items():
            if TAGS.get(tag) == "FocalLength":
                if isinstance(value, tuple):
                        # Handle IFDRational object
                    focal_length = float(value[0]) / float(value[1])
                else:
                        # Directly use the value if it's not a tuple
                    focal_length = value
                break

    return float(focal_length)

def get_image_center(image):
    height, width = image.shape[:2]
    center_x = width / 2
    center_y = height / 2
    return center_x, center_y


def get_camera_matrix(focal_length, center):
    return np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)


def estimate_motion(matched_pairs, camera_matrix):
    # Unpack the points from the pairs
    points1 = np.float32([pair[0] for pair in matched_pairs])
    points2 = np.float32([pair[1] for pair in matched_pairs])

    E, mask = cv2.findEssentialMat(points1, points2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, points1, points2, camera_matrix)

    return R, t


def main():
    global images, axs
    folder_path = "images"
    images = load_images_from_folder(folder_path)

    if len(images) < 2:
        print("The folder must contain at least two images.")
        return

    fig, axs = display_images_side_by_side(images)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Extract points for each image from matched_points
    # Print matched points
    points_image1 = [pt[0] for pt in matched_points if len(pt) == 2]
    points_image2 = [pt[1] for pt in matched_points if len(pt) == 2]
    print("Matched points:", matched_points)

    # Match features and visualize them
    # Print the matched pairs
    matched_pairs = match_features(images[0], images[1], points_image1, points_image2)
    if matched_pairs:
        for pair in matched_pairs:
            print(f"Point in Image 1: {pair[0]}, Point in Image 2: {pair[1]}")


    #obtain focal lengths
    focal_length = get_focal_lengths_from_exif('images/image0.jpg')
    if focal_length:
        print(f"Focal Length of image : {focal_length}mm")
    else:
        print(f"Focal length not found in EXIF data for image.")

    #obtain image centers
    center_x, center_y = get_image_center(images[0])
    print(f"Image center: ({center_x}, {center_y})")

    camera_matrix = get_camera_matrix(focal_length, (center_x, center_y))
    print("Camera Matrix:", camera_matrix)

    R, t = estimate_motion(matched_pairs, camera_matrix)
    print("R:", R)
    print("t:", t)

    # Projection matrix for the first camera (Assuming it's at the origin)
    # np.eye(3) The parameter 3 means it's a 3x3 identity matrix.
    # np.zeros((3, 1) This creates a 3x1 column matrix (or vector) filled with zeros.
    P1 = camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))

    # Projection matrix for the second camera
    P2 = camera_matrix @ np.hstack((R, t))

    # Now, P1 and P2 can be used for further steps like triangulation
    print("Projection Matrix P1:\n", P1)
    print("Projection Matrix P2:\n", P2)


if __name__ == "__main__":
    main()