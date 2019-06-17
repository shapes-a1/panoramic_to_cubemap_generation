# ---------------------------------------------------------------------------------------------
# This file converts a given 2:1 Equirectangular Image to CubeMap Faces
# We follow the Inverse Transformation Procedure
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# Necessary Imports
import math
import argparse
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------------------------
# Configuration Parameters

# Dictionary for CUBEMAP FACES
CUBEMAP_FACES = {'back': 0, 'left': 1, 'front': 2, 'right': 3, 'top' : 4, 'bottom' : 5 }

# Dimensions of the input image
INPUT_WIDTH = 2048
INPUT_HEIGHT = 1024

# ---------------------------------------------------------------------------------------------
# Necessary Methods
# -------------------------------------------------------
# -------------------------------------------------------
def output_image_to_xyz(i, j, cube_face_id, edge_length):
    """
    Get x,y,z coordinates from the output image pixel coordinates (i,j) taking into account different faces of the cube map
    :param i: Image row Pixel Coordinate
    :param j: Image column Pixel Coordinate
    :param cube_face_id: The face Id of the Cube Map
    :param edge_length: The size of the face
    :return: Tuple (x,y,z)
    """

    a = 2.0 * float(i) / edge_length
    b = 2.0 * float(j) / edge_length

    if cube_face_id == CUBEMAP_FACES['back']:
        (x,y,z) = (-1.0, 1.0 - a, 1.0 - b)
    elif cube_face_id == CUBEMAP_FACES['left']:
        (x,y,z) = (a - 1.0, -1.0, 1.0 - b)
    elif cube_face_id == CUBEMAP_FACES['front']:
        (x,y,z) = (1.0, a - 1.0, 1.0 - b)
    elif cube_face_id == CUBEMAP_FACES['right']:
        (x,y,z) = (1.0 - a, 1.0, 1.0 - b)
    elif cube_face_id == CUBEMAP_FACES['top']:
        (x,y,z) = (b - 1.0, a - 1.0, 1.0)
    elif cube_face_id == CUBEMAP_FACES['bottom']:
        (x,y,z) = (1.0 - b, a - 1.0, -1.0)

    return (x, y, z)

# -------------------------------------------------------
def generate_cubemap_face(input_image, output_image, cube_face_id):
    """
    Generate a face of the Cube Map from the Input Image
    :param input_image: Input Image
    :param output_image: Output Image Handle as the blank image
    :param cube_face_id: Face ID to construct
    :return:Implicit in the Output Image
    """

    input_size = input_image.size
    output_size = output_image.size
    input_pixels = input_image.load()
    output_pixels = output_image.load()
    edge_length = output_size[0]

    for x_out in xrange(edge_length):
        for y_out in xrange(edge_length):
            (x,y,z) = output_image_to_xyz(x_out, y_out, cube_face_id, edge_length)
            theta = math.atan2(y,x)
            r = math.hypot(x,y)
            phi = math.atan2(z,r)

            # Source Image Coordinates
            uf = 0.5 * input_size[0] * (theta + math.pi) / math.pi
            vf = 0.5 * input_size[0] * (math.pi/2 - phi) / math.pi

            # Bilinear Interpolation amongst the four surrounding pixels
            ui = math.floor(uf)  # bottom left
            vi = math.floor(vf)
            u2 = ui+1            # top right
            v2 = vi+1

            mu = uf-ui
            nu = vf-vi

            # Pixel values of four corners
            A = input_pixels[ui % input_size[0], np.clip(vi, 0, input_size[1]-1)]
            B = input_pixels[u2 % input_size[0], np.clip(vi, 0, input_size[1]-1)]
            C = input_pixels[ui % input_size[0], np.clip(v2, 0, input_size[1]-1)]
            D = input_pixels[u2 % input_size[0], np.clip(v2, 0, input_size[1]-1)]

            # interpolate
            (r,g,b) = (
              A[0]*(1-mu)*(1-nu) + B[0]*(mu)*(1-nu) + C[0]*(1-mu)*nu+D[0]*mu*nu,
              A[1]*(1-mu)*(1-nu) + B[1]*(mu)*(1-nu) + C[1]*(1-mu)*nu+D[1]*mu*nu,
              A[2]*(1-mu)*(1-nu) + B[2]*(mu)*(1-nu) + C[2]*(1-mu)*nu+D[2]*mu*nu )

            output_pixels[x_out, y_out] = (int(round(r)), int(round(g)), int(round(b)))

# -------------------------------------------------------
def generate_cubemap_outputs(input_image_path):
    """
    Generate the cubemap faces output from the input image path
    (Directly save the output in the same path with suffixes for cubemap faces)
    :param input_image_path: Input Image
    :return:Implicit in the Output Image Saving Process
    """

    print ('---------------------------------------------------------------------------')
    print ('Generating Cubemap Faces for the image ------------ ' + input_image_path)

    # Read the input image
    input_image = Image.open(input_image_path)
    input_image = input_image.resize((int(INPUT_WIDTH), int(INPUT_HEIGHT)))
    input_size = input_image.size

    # Define the edge length - This will be the width and height of each cubemap face
    edge_length = input_size[0] / 4

    # Generate cubemap for the left face
    out_image = Image.new("RGB", (edge_length, edge_length), "black")
    generate_cubemap_face(input_image, out_image, CUBEMAP_FACES['left'])
    out_image.save(input_image_path[0:-4] + '_cubemap_left.png')
    print ('LEFT face done')

    # Generate cubemap for the front face
    out_image = Image.new("RGB", (edge_length, edge_length), "black")
    generate_cubemap_face(input_image, out_image, CUBEMAP_FACES['front'])
    out_image.save(input_image_path[0:-4] + '_cubemap_front.png')
    print ('FRONT face done')

    # Generate cubemap for the right face
    out_image = Image.new("RGB", (edge_length, edge_length), "black")
    generate_cubemap_face(input_image, out_image, CUBEMAP_FACES['right'])
    out_image.save(input_image_path[0:-4] + '_cubemap_right.png')
    print ('RIGHT face done')

    # Generate cubemap for the back face
    out_image = Image.new("RGB", (edge_length, edge_length), "black")
    generate_cubemap_face(input_image, out_image, CUBEMAP_FACES['back'])
    out_image.save(input_image_path[0:-4] + '_cubemap_back.png')
    print ('BACK face done')

    # Generate cubemap for the top face
    out_image = Image.new("RGB", (edge_length, edge_length), "black")
    generate_cubemap_face(input_image, out_image, CUBEMAP_FACES['top'])
    out_image.save(input_image_path[0:-4] + '_cubemap_top.png')
    print ('TOP face done')

    # Generate cubemap for the bottom face
    out_image = Image.new("RGB", (edge_length, edge_length), "black")
    generate_cubemap_face(input_image, out_image, CUBEMAP_FACES['bottom'])
    out_image.save(input_image_path[0:-4] + '_cubemap_bottom.png')
    print ('BOTTOM face done')

    print ('---------------------------------------------------------------------------')

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help = "Enter the path of the input photosphere image (Resizes to INPUT_WIDTH x INPUT_HEIGHT for speed)",
                        type=str, default='', required=True)

    args = parser.parse_args()
    input_image_path = args.input
    generate_cubemap_outputs(input_image_path)

if __name__ == "__main__":
    main()

