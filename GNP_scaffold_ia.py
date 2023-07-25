# Import Needed Packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
import statistics


def image_preprocessing(path, background):
    """ 
    Inputs:
        Path == the location of the image being run
        Background == the color of background. Only accepts "Black" or "White"
    
    Outputs:
        image == the original image
        binary_image == binary image of the original that has been de-noised, blurred and threshold.
    """

    # Initializing Variables
    image = cv2.imread(path)

    # Processing the Images
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    dst = cv2.fastNlMeansDenoising(blur, None, 25, 25, 30)
    thres = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Inverting the threshold image and making it binary
    binary_image = np.copy(thres)

    if background == "Black":
        binary_image[thres == 0] = 0
        binary_image[thres == 255] = 1
        
    if background == "White":
        binary_image[thres == 0] = 1
        binary_image[thres == 255] = 0
        
    return image, binary_image


def single_image(path, background):
    """
    Inputs:
        Path == the location of the image being run
        Background == the color of background. Only accepts "Black" or "White"

    Returns:
        org_image == A grayscale of the original image
        scaffold == A binary image of the scaffold that has been de-noised, blurred and threshold
        cropped == A shrunk version of the "scaffold" image
    """
    
    org_image, scaffold = image_preprocessing(path, background)

    # Cropping the image to the scaffold
    # Determining where the scaffold is
    results = np.array(np.where(scaffold == 1, scaffold, 0))

    # Find the non-zero min-max coords
    pts = np.argwhere(results == 1)

    y1, x1 = pts.min(axis=0)
    y2, x2 = pts.max(axis=0)

    # Crop the region
    cropped = results[y1:y2, x1:x2]

    return org_image, scaffold, cropped


def find_pore_centroid(path, background):
    """
    This function finds the centroids of the 7th pore within the scaffold.

    Inputs:
        Path == the location of the image being run
        Background == the color of background. Only accepts "Black" or "White"

    Returns:
        (cX, cY) == The centroid of the 7th accepted pore based on dimension criteria.
                    This pore tends to be in the second row of the scaffold.
    """
    # Initializing Variables
    connectivity = 4
    org_image, scaffold = image_preprocessing(path, background)
    (num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(scaffold, connectivity, cv2.CV_32S)

    # Finding a second row pore
    pore_num = []
    for i in range(0, num_labels):
        if i == 0:
            continue

        # extract the pore's statistics
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # ensure the width, height, and area are not too small (Removing labels that may not be pores)
        keep_width = w > 100
        keep_height = h > 100
        keep_area = 1000 < area < 300000  # The upper range needed to be changed depending on the time point

        # ensure the connected component we are examining passes all three criteria
        if all((keep_width, keep_height, keep_area)):
            # Adding num_labels to a list of accepted pores
            pore_num.append(i)

    # Using the 7th accepted pore for midpoint measurements
    (cX, cY) = centroids[pore_num[6]]

    return cX, cY


def continuous_line(line):
    """
    This function determines the length of continuous numbers to determine fiber length. This function returns a list
    of fiber lengths that meet a set criteria too. (Doesn't accept 'fibers' that are less than 80 pixels)

    Inputs:
        line == a list of positions where the scaffold is from the image where the fiber diameter will be measured along

    Returns:
        distance == a list of lengths of each fiber within the scaffold that are longer than 80 pixels
    """
    fiber_pts = np.argwhere(line == 0)  # Getting the indexes of where white or a fiber is
    distance = []
    holder_list = []

    # Going through the midline and summing consecutive numbers
    for index in range(len(fiber_pts)):

        # initializing the list at the beginning of the row
        if len(holder_list) == 0:
            holder_list.append(fiber_pts[index])
            continue

        # After the list has a index in it
        if len(holder_list) != 0:
            last_value = holder_list[-1]

            # Is it consecutive with our current list? (Is it in the same fiber)
            if (fiber_pts[index]) == (last_value + 1):
                holder_list.append(fiber_pts[index])  # Yes it is, add to the fiber's diameter

                # Are we close to the end of the photo?
                if index == (len(fiber_pts) - 1):
                    diameter = len(holder_list)  # Yes, let's finish the last fiber
                    distance.append(diameter)
                    holder_list = []
            else:
                diameter = len(holder_list)  # No it isn't, we have reached the end of a fiber
                distance.append(diameter)
                holder_list = [fiber_pts[index]]

    # Combining small "fiber" measures. This may be due to holes within the fibers
    distance = np.array(distance)
    remove_index = []
    for fibers in range(len(distance)):
        if distance[fibers] < 80:  # Setting a diameter critique to 80 pixels
            remove_index.append(fibers)
            if fibers == range(len(distance))[-1]:
                distance[fibers - 1] = distance[fibers] + distance[fibers - 1]

            else:
                distance[fibers + 1] = distance[fibers] + distance[fibers + 1]

    distance = np.delete(distance, remove_index)

    return distance


def fiber_diameter(path, scale, background):
    """
    This function determined the fibers diameter at the midpoint of the image
    Input:
        Path == the location of the image being run
        Scale == numerical value that converts pixel to a physical length
        Background == the color of background. Only accepts "Black" or "White"

    Output: List of the fiber diameters at the midpoint of the image
    """
    org_image, scaffold, cropped = single_image(path, background)

    # Determining the mid point of the image
    height = scaffold.shape[0]
    midline = scaffold[round(height / 2), :]
    distance = continuous_line(midline)
    lengths = np.round(distance / scale, decimals=5)
    average_length = statistics.mean(lengths)

    return lengths, average_length


def fiber_diameter_v2(path, scale, background):
    """
    This function determined the fibers diameter within the image at the center of the 7th accepted pore

    Input:
        Path == the location of the image being run
        Scale == numerical value that converts pixel to a physical length
        Background == the color of background. Only accepts "Black" or "White"

    Output:
        vertical_lengths ==  List of the vertical fiber diameters at the centroid of the 7th accepted pore
        vertical_average_length == Average vertical fiber diameter
        horizontal_lengths == List of the horizontal fiber diameters at the centroid of the 7th accepted pore
        horizontal_average_length == Average horizontal fiber diameter
        cX, cY == The centroid of the 7th accepted pore
    """
    # Initializing variables
    org_image, scaffold, cropped = single_image(path, background)
    (cX, cY) = find_pore_centroid(path, background)  # Centroid coordinates for the 7th accepted pore

    # Calculating the fiber diameter at the vertical line
    vertical_line = scaffold[round(cX), :]
    vertical_distance = continuous_line(vertical_line)

    vertical_lengths = np.round(vertical_distance / scale, decimals=5)
    vertical_average_length = statistics.mean(vertical_lengths)

    # Calculating the fiber diameter at the horizontal line
    horizontal_line = scaffold[:, round(cY)]
    horizontal_distance = continuous_line(horizontal_line)

    horizontal_lengths = np.round(horizontal_distance / scale, decimals=5)
    horizontal_average_length = statistics.mean(horizontal_lengths)

    return vertical_lengths, vertical_average_length, horizontal_lengths, horizontal_average_length, cX, cY


def surface_area(image, scale):
    """
    This function reports the area of the image that is the scaffold

    Input
        Image == A binary image that has been ran through the single image process method above of the scaffold
        Scale ==  An integer that represents the amount of pixels in a mm

    Output
        SA_mm == This integer represents the amount of area the fibers/scaffold takes up within the image in mm^2
    """
    # Inputs/ outputs
    fiber_pts = np.argwhere(image == 1)
    sa_pixel = len(fiber_pts)
    sa_mm = np.round(sa_pixel / np.square(scale), decimals=5)

    return sa_mm


def pore_area(path, scale, background):
    """
    This function uses the connected component method in cv2 to isolate the pores within an inverted scaffold image
    to obtain the average pore area and overall pore area within the scaffold on a black background.

    Inputs:
        Path == The location of the image being run
        Scale == An integer representing the number of pixels in a mm
        Background == The color of background. Only accepts "Black" or "White"

    Output:
        Mask == An image representing the connected components that met the given size requirement
        Average_pore_size_mm2 == An integer that represents the average individual pore area
        Total_pore_area_mm2 == An integer that represents the porous area within the scaffold in mm^2
    """
    # Initializing Variables
    connectivity = 4  # Connectivity is either 4 or 8
    org_image, scaffold = image_preprocessing(path, background)
    (num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(scaffold, connectivity, cv2.CV_32S)

    # initialize an output mask to store all characters
    mask = np.zeros(scaffold.shape, dtype="uint8")

    pore_count = 0
    total_pore = 0
    for i in range(1, num_labels):
        # extract the connected component statistics for the current
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # ensure the width, height, and area are all neither too small
        keep_width = w > 100
        keep_height = h > 100
        keep_area = 1000 < area < 300000

        # ensure the connected component we are examining passes all
        # three tests
        if all((keep_width, keep_height, keep_area)):
            pore_count += 1
            total_pore += area
            component_mask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, component_mask)
    average_pore_size_pixels = total_pore / pore_count
    average_pore_size_mm2 = average_pore_size_pixels / np.square(scale)

    total_pore_area_pixels = total_pore
    total_pore_area_mm2 = total_pore_area_pixels / np.square(scale)

    return mask, average_pore_size_mm2, total_pore_area_mm2, pore_count


def create_figure(org_image, cropped, pore_mask, cx, cy):
    """ This function returns a figure that combined all the images of each step in process
    Inputs:
        org_image == A grayscale of the original image
        cropped == A shrunk version of the "scaffold" image
        pore_mask == An image representing the connected components that met the given size requirement
        cx, cy == The centroid of the 7th accepted pore based on dimension criteria. This pore tends to be in the
                    second row of the scaffold.
    Outputs:
    A 2 x 2 photo grid that shows the original image (quad 1), the cropped binary image of the scaffold with a red line
    through the midpoint (quad 2), the cropped binary image of the scaffold with a red line through the 7th accepted
    pore centroid (quad 3) and an binary image of the accepted pores.
    """

    # Creating a Figure
    fig = plt.figure(figsize=(10, 10))

    # Original image of the scaffold
    fig.add_subplot(2, 2, 1)
    plt.imshow(org_image, cmap='gray')
    plt.title('Original Image', fontsize=20)

    # Scaffold with lines through image midpoint
    fig.add_subplot(2, 2, 2)
    cropped_width = cropped.shape[1]
    x = [round(cropped_width / 2), round(cropped_width / 2)]
    y = [0, cropped_width]
    plt.plot(y, x, color="red", linewidth=3)
    plt.imshow(cropped, cmap='gray')
    plt.title('Cropped Scaffold', fontsize=20)

    # Scaffold with lines through the centroid of the 7th accepted pore
    fig.add_subplot(2, 2, 3)
    plt.imshow(pore_mask, cmap='gray')
    plt.title('Accepted Pores', fontsize=20)

    copy_image = cropped.copy()
    width = copy_image.shape[1]
    height = copy_image.shape[0]
    cv2.line(copy_image, (round(cx), 0), (round(cx), height), (0, 255, 0), 25)
    cv2.line(copy_image, (0, round(cy)), (round(width), round(cy)), (0, 255, 0), 25)

    # Showing Pore
    fig.add_subplot(2, 2, 4)
    plt.imshow(copy_image, cmap='gray')
    plt.title('Fiber Diameter Measured Here', fontsize=20)

    plt.show()
