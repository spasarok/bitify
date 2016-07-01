import cv2
import numpy as np
import scipy as sp

"""
# NES color palette
palette_nes = np.array([
    [124,124,124],
    [0,0,252],
    [0,0,188],
    [68,40,188],
    [148,0,132],
    [168,0,32],
    [168,16,0],
    [136,20,0],
    [80,48,0],
    [0,120,0],
    [0,104,0],
    [0,88,0],
    [0,64,88],
    [0,0,0]])
"""

# Compare color difference based on rgb values
def rgb_diff(pixel_1, pixel_2, threshold):

    if(threshold < 0):
        return np.sum(np.absolute(pixel_1 - pixel_2))

    def channel_order(pixel):
        max_1 = np.argmax(pixel)
        max_3 = np.argmin(pixel)
        if(max_1 + max_3 == 3):
            max_2 = 3
        elif(max_1 + max_3 == 4):
            max_2 = 2
        else:
            max_2 = 1
        max_val = (max_1 * 100) + (max_2 * 10) + max_3
        return max_val

    def channel_diff(pixel):
        max_1 = np.amax(pixel)
        max_3 = np.amin(pixel)
        if(max_1 + max_3 == 3):
            max_2 = 3
        elif(max_1 + max_3 == 4):
            max_2 = 2
        else:
            max_2 = 1
        max_diff = np.absolute(max_1 - max_2)
        return max_diff

    diff = np.sum(np.absolute(pixel_1 - pixel_2))

    if(channel_order(pixel_1) == channel_order(pixel_2) and channel_diff(pixel_1) >= threshold):
        return diff
    else:
        return diff * 10

# Compare color difference based on hsv values
def hsv_diff(pixel_1, pixel_2, threshold):
    ranges = np.zeros((3))
    ranges[0] = 179.0
    ranges[1] = 255.0
    ranges[2] = 255.0
    diff = np.sum(np.absolute(pixel_1 - pixel_2) * 1.0 / ranges)
    return diff * 10

# Get palette from an image of distinct colors 
# (see provided palettes)
def get_palette(image):
    shape = image.shape
    rows = shape[0]
    cols = shape[1]
    colors = []
    colors.append(image[0][0])

    for row in range(rows):
        for col in range(cols):
            color = image[row][col]
            for val in range(len(colors)):
                if(np.all(colors[val] == color)):
                    continue
                else:
                    colors.append(color)
                    break

    colors = np.asarray(colors)
    num_colors = np.size(colors) / 3
    out = np.zeros((1, num_colors, 3))
    for val in range(num_colors):
        out[0][val] = colors[val]

    return out

# Converts image to specified color palette using the specified
# color_compare function
def palettize(image, palette, color_compare, threshold):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_palette = cv2.cvtColor(palette, cv2.COLOR_BGR2HSV)

    shape = image.shape;
    rows = shape[0]
    cols = shape[1]
    rgb = shape[2]

    pal_shape = palette.shape
    pal_cols = pal_shape[1]

    out = np.zeros((rows, cols, rgb))

    for row in range(rows):
        for col in range(cols):

            pixel = np.zeros(rgb)

            if(color_compare == "hsv_diff"):
                for channel in range(rgb):
                    pixel[channel] = hsv_image.item(row, col, channel)

                closest = color_compare(pixel, hsv_palette[0][0], threshold)
                index = 0

                for color in range(1, pal_cols):
                    diff = color_compare(pixel, hsv_palette[0][color], threshold)
                    if(diff < closest):
                        closest = diff
                        index = color

                out[row, col] = palette[0][index]

            else:
                for channel in range(rgb):
                    pixel[channel] = image.item(row, col, channel)

                closest = color_compare(pixel, palette[0][0], threshold)
                index = 0

                for color in range(1, pal_cols):
                    diff = color_compare(pixel, palette[0][color], threshold)
                    if(diff < closest):
                        closest = diff
                        index = color

            out[row, col] = palette[0][index]

    return out

"""
# Can't remember what this does
def convert(image, palette):

    shape = image.shape
    rows = shape[0]
    cols = shape[1]
    channels = shape[2]

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_palette = cv2.cvtColor(palette, cv2.COLOR_BGR2HSV)

    out = np.zeros((rows, cols, channels))

    for row in range(rows):
        for col in range(cols):

            hue = hsv_image.item(row, col, 0)
            closest = np.absolute(hue - hsv_palette.item(0, 0, 0))
            index = 0

            for pixel in range(1, np.size(hsv_palette)/3):
                diff = np.absolute(hue - hsv_palette[0][pixel][0])
                if(diff < closest):
                    closest = diff
                    index = pixel

            b, g, r = palette[0][index]
            out[row, col, 0] = b
            out[row, col, 1] = g
            out[row, col, 2] = r

    return out
"""

# Helper to get allocate bits across color channels
def get_bits(bits):
    bits_per_channel = bits / 3.0
    bits = np.zeros(3, dtype=np.uint8)

    lower = np.floor(bits_per_channel)
    upper = np.ceil(bits_per_channel)
    remainder = bits_per_channel - lower

    if(remainder < 0.3):
        bits[2] = lower
        bits[1] = lower
        bits[0] = lower
    elif(remainder < 0.6):
        bits[2] = upper
        bits[1] = lower
        bits[0] = lower
    else:
        bits[2] = upper
        bits[1] = upper
        bits[0] = lower

    return bits

# Converts image to color channels of specified bit size
def bitify(image, bits, pixel_size):
    shape = image.shape
    rows = shape[0]
    cols = shape[1]
    channels = shape[2]
    bit_array = get_bits(bits)
    out = np.zeros((rows, cols, channels))

    for row in range(rows):
        for col in range(cols):
            for channel in range(channels):
                colors = (2 ** bit_array[channel] - 1) * 1.0
                val_256 = image.item(row, col, channel)
                val_norm = np.round(val_256 / 255.0 * colors) / colors

                out[row, col, channel] = np.round(val_norm * 255)

    if(pixel_size > 1):
        out = pixelate(out, pixel_size)

    return out

# Makes an image blocky and pixelated
def pixelate(image, pixel_size):
    shape = image.shape
    rows = shape[0]
    cols = shape[1]
    channels = shape[2]
    side_length = int(np.sqrt(pixel_size))

    small = image[0:rows:side_length][:,0:cols:side_length]
    shape = small.shape
    rows = shape[0]
    cols = shape[1]

    mergeRows = np.zeros([rows * side_length, cols, channels])
    mergeRows[0::side_length] = small
    mergeCols = np.zeros([rows * side_length, cols * side_length, channels])
    mergeCols[:,0::side_length] = mergeRows

    out = mergeCols
    shape = out.shape
    rows = shape[0]
    cols = shape[1]

    for row in xrange(0, rows, side_length):
        for col in xrange(0, cols, side_length):
            rgb = out[row][col]
            for off_row in range(0, side_length):
                for off_col in range(0, side_length):
                    out[row + off_row][col + off_col] = rgb

    return out

# Slightly offsets rgb values to look like an old tv image
def color_fringe(image, bb, gg, rr):
    b, g, r = cv2.split(image)
    b += bb
    g += gg
    r += rr
    return cv2.merge((b,g,r))

# Adds retro scan lines
def scanlines(image, pixel_size, difference):
    shape = image.shape
    rows = shape[0]
    cols = shape[1]
    channels = shape[2]
    out = image
    side_length = int(np.sqrt(pixel_size))

    for row in xrange(rows):
        if(row % side_length != 0):
            continue
        for col in range(cols):
            for channel in range(channels):
                out[row][col][channel] = image.item(row, col, channel) + difference

    return out

### Example usage

#palette_nes = cv2.imread("palettes/palette_nes.png")
#palette_ntsc = cv2.imread("palettes/palette_atari_ntsc.png")
#palette_pal = cv2.imread("palettes/palette_atari_pal.png")
#palette_smb2 = cv2.imread('palettes/palette_smb2.png')
#parrot = cv2.imread("input/parrot.jpg")
#cv2.imwrite('palette_nes_hsv_fringe.jpg', color_fringe(palletize(parrot, palette_nes, hsv_diff, 4), 10, 30, -10))
#cv2.imwrite('parrot_color_frined.jpg', color_fringe(convert(parrot, 6, 1), 5, 15, -5))
#cv2.imwrite('parrot_pixel_16.jpg', pixelate(parrot, 16))
#cv2.imwrite('parrot_scanlines.jpg', scanlines(bitify(parrot, 6, 4), 16, 50))
#cv2.imwrite('out.jpg', color_fringe(scanlines(bitify(parrot, 6, 4), 16, 30), 30, 15, -10))




