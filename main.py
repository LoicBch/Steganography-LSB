import cv2
import numpy as np
def to_binary_array(image):
    array_bits = []
    hauteur, largeur = image.shape[:2]
    for y in range(hauteur):
        for x in range(largeur):
            b, g, r = image[y, x]

            valeur_binaire_b = bin(b)[2:].zfill(8)
            valeur_binaire_g = bin(g)[2:].zfill(8)
            valeur_binaire_r = bin(r)[2:].zfill(8)

            array_bits.extend(map(int, valeur_binaire_b))
            array_bits.extend(map(int, valeur_binaire_g))
            array_bits.extend(map(int, valeur_binaire_r))

    array_bits = np.array(array_bits)
    return array_bits


def lsb_img(image_host, image_hidden):
    hidden_binary_array = to_binary_array(image_hidden)
    host_binary_array = to_binary_array(image_host)

    for index in range(7, hidden_binary_array.size*8, 8):
        host_binary_array[index] = hidden_binary_array[index // 8]

    host_array = np.array(host_binary_array, dtype=np.uint8)
    host_octets = [host_array[i:i + 8] for i in range(0, len(host_array), 8)]
    host_ints = [int(''.join(map(str, octet)), 2) for octet in host_octets]
    host_np_array = np.array(host_ints, dtype=np.uint8)

    print(hidden_binary_array[:100])
    print(host_binary_array[:50])

    height, width = image_host.shape[0], image_host.shape[1]
    image = np.zeros((height, width, 3), np.uint8)
    index = 0
    for y in range(height):
        for x in range(width):
            b = host_np_array[index]
            g = host_np_array[index + 1]
            r = host_np_array[index + 2]
            image[y, x] = [b, g, r]
            index = index + 3

    print(to_binary_array(image)[:100])

    return image


def show_hidden_image(image, size, limit):
    lsb_array = []
    image_binary_array = to_binary_array(image)

    # Inserer un stop pour detecter la fin de detection
    for index in range(7, limit, 8):
        lsb_array.append(image_binary_array[index])

    print("----> ")
    print(image_binary_array[:100])
    print(lsb_array[:100])

    print(len(lsb_array))
    hidden_array = np.array(lsb_array, dtype=np.uint8)
    host_octets = [hidden_array[i:i + 8] for i in range(0, len(hidden_array), 8)]
    hidden_ints = [int(''.join(map(str, octet)), 2) for octet in host_octets]
    hidden_np_array = np.array(hidden_ints, dtype=np.uint8)

    # Inserer un stop pour detecter la fin de detection
    height, width = size, size
    image = np.zeros((height, width, 3), np.uint8)
    index = 0
    for y in range(height):
        for x in range(width):
            b = hidden_np_array[index]
            g = hidden_np_array[index + 1]
            r = hidden_np_array[index + 2]
            image[y, x] = [b, g, r]
            index = index + 3

    return image


img = cv2.imread("host.png")
img_hidden = cv2.imread("secret.png")
img = lsb_img(img, img_hidden)
img = show_hidden_image(img, 64, 786432)
cv2.imshow("Encoded image", img)
cv2.waitKey(0)
