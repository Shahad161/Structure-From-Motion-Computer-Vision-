import matplotlib.pyplot as plt
import cv2
import os

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


def estimating_motion(points1, points2, images):
    focal_length = 1.0  # Example value, use actual focal length

    for img in images:
        height, width = img.shape[:2]
        principal_point = (width / 2, height / 2)

    # Compute the Essential Matrix
    E, mask = cv2.findEssentialMat(points1, points2, focal=focal_length, pp=principal_point, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    print("E", E)
    print("mask", mask)

    # Recover rotation and translation from the Essential Matrix
    _, R, t, _ = cv2.recoverPose(E, points1, points2)


def main():
    global images, axs
    folder_path = "images"  # Replace with your folder path
    images = load_images_from_folder(folder_path)

    if len(images) < 2:
        print("The folder must contain at least two images.")
        return

    fig, axs = display_images_side_by_side(images)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show() 

    # Extract points for each image from matched_points
    points_image1 = [pt[0] for pt in matched_points if len(pt) == 2]
    points_image2 = [pt[1] for pt in matched_points if len(pt) == 2]

    # Print matched points
    print("Matched points:", matched_points)

    # Match features and visualize them
    matched_pairs = match_features(images[0], images[1], points_image1, points_image2)

    # Optionally, print the matched pairs
    if matched_pairs:
        for pair in matched_pairs:
            print(f"Point in Image 1: {pair[0]}, Point in Image 2: {pair[1]}")

    estimating_motion(points_image1, points_image2, images)

if __name__ == "__main__":
    main()