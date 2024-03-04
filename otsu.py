from typing import cast
import numpy as np
import cv2


def otsu(image):
    histogram = np.zeros(256)

    for i in range(256):
        histogram[i] = np.sum(image == i)

    # Normalize the histogram to obtain the probability of each intensity level
    histogram = histogram / (image.shape[0] * image.shape[1])

    intra_class_variances = np.zeros(256)

    for thresh in range(256):
        # Divide the histogram into two classes using the threshold
        background = histogram[:thresh]
        foreground = histogram[thresh:]

        # Calculate the mean of each class using the intensity levels and their probabilities
        background_mean = np.sum(np.arange(thresh) * background)
        foreground_mean = np.sum(np.arange(thresh, 256) * foreground)

        sum_background = np.sum(background)
        sum_foreground = np.sum(foreground)

        if sum_background == 0:
            intra_class_variances[thresh] = np.inf
            continue

        if sum_foreground == 0:
            intra_class_variances[thresh] = np.inf
            continue

        # Normalize the means
        background_mean = background_mean / sum_background
        foreground_mean = foreground_mean / sum_foreground

        # Calculate the variance of each class
        background_variance = np.sum(
            ((np.arange(thresh) - background_mean) ** 2) * background
        )
        foreground_variance = np.sum(
            ((np.arange(thresh, 256) - foreground_mean) ** 2) * foreground
        )

        # Calculate the intra-class variance
        intra_class_variance = background_variance + foreground_variance
        intra_class_variances[thresh] = intra_class_variance

    # Find the threshold that minimizes the intra-class variance
    optimal_threshold = np.argmin(intra_class_variances)

    return cast(int, optimal_threshold)


def main():
    img = cv2.imread("test_img.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    threshold = otsu(img)

    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    opencv_threshold, _ = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    print(f"Optimal threshold: {threshold}")
    print(f"OpenCV's threshold: {opencv_threshold}")

    cv2.imshow("Original Image", img)
    cv2.imshow("Binary Image", binary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
