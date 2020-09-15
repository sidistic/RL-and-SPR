import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image
from tqdm import tqdm

def readImage(filename):
    a = plt.imread(filename)
    image_np = np.array(a)
    # print (image_np.shape[1])
    image_r = image_np[:,:,0]
    image_g = image_np[:,:,1]
    image_b = image_np[:,:,2]
    return image_r, image_g, image_b

def readImage_new(image_np):
    # print (image_np.shape[1])
    image_r = image_np[:,:,0]
    image_g = image_np[:,:,1]
    image_b = image_np[:,:,2]
    return image_r, image_g, image_b

def getEigen(imageStream):
    centralisedImage = (imageStream - np.mean(imageStream.T, axis = 1)).T
    eigenValues, eigenVectors = np.linalg.eigh(np.cov(centralisedImage))
    return eigenValues, eigenVectors, centralisedImage

def getTopIndex(eigenValues, eigenVectors, numComponents):
    idx = np.argsort(eigenValues)
    idx = idx[::-1]
    eigenVectors = eigenVectors[:,idx]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, range(numComponents)]
    return eigenValues, eigenVectors

def getImage(reconstructedEigenValues, reconstructedEigenVectors, centralisedImage, imageStream):
    score = np.dot(reconstructedEigenVectors.T, centralisedImage)
    reconstructedImage = np.dot(reconstructedEigenVectors, score).T + np.mean(imageStream, axis = 0) # SOME NORMALIZATION CAN BE USED TO MAKE IMAGE QUALITY BETTER
    recon_img_mat = np.uint8(np.absolute(reconstructedImage))
    return recon_img_mat

def oneStream(imageStream, numComponents):
    eigenValues, eigenVectors, centralisedImage = getEigen(imageStream)
    reconstructedEigenValues, reconstructedEigenVectors = getTopIndex(eigenValues, eigenVectors, numComponents)
    tempImage = getImage(reconstructedEigenValues, reconstructedEigenVectors, centralisedImage, imageStream)
    return tempImage

def fullProcess(image_np, numComponents):
    image_r, image_g, image_b = readImage_new(image_np)
    image_r_reconstructed, image_g_reconstructed, image_b_reconstructed = oneStream(image_r, numComponents), oneStream(image_g,numComponents), oneStream(image_b,numComponents) # RECONSTRUCTING R,G,B COMPONENTS SEPARATELY
    reconstructed_color_img = np.dstack((image_r_reconstructed, image_g_reconstructed, image_b_reconstructed)) # COMBINING R.G,B COMPONENTS TO PRODUCE COLOR IMAGE
    # reconstructed_color_img = Image.fromarray(reconstructed_color_img)
    return reconstructed_color_img

    
def main():
    image_r, image_g, image_b = readImage('1.jpg')
    images = []
    listIndex = [200, 500]
    for i in listIndex:
        image_r_reconstructed, image_g_reconstructed, image_b_reconstructed = oneStream(image_r, i), oneStream(image_g,i), oneStream(image_b,i) # RECONSTRUCTING R,G,B COMPONENTS SEPARATELY
        reconstructed_color_img = np.dstack((image_r_reconstructed, image_g_reconstructed, image_b_reconstructed)) # COMBINING R.G,B COMPONENTS TO PRODUCE COLOR IMAGE
        reconstructed_color_img = Image.fromarray(reconstructed_color_img)
        reconstructed_color_img.save(str(i)+'-animal.jpg')
    # images[0].save('animal.gif',
    #            save_all=True, append_images=images[1:], optimize=True, duration=40, loop=0)

def trial2(image_np, numComponents):
    image_r, image_g, image_b = readImage_new(image_np)

    u, s, v = np.linalg.svd(image_r)
    u_ = u[:, range(numComponents)]
    v_ = v[range(numComponents), :]
    s_ = s[range(numComponents)]
    s_ = np.diag(s_)
    # print (u_.shape, v_.shape, s_.shape)

    temp_r = (u_ @ s_) @ v_


    u, s, v = np.linalg.svd(image_g)
    u_ = u[:, range(numComponents)]
    v_ = v[range(numComponents), :]
    s_ = s[range(numComponents)]
    s_ = np.diag(s_)
    # print (u_.shape, v_.shape, s_.shape)

    temp_g = (u_ @ s_) @ v_


    u, s, v = np.linalg.svd(image_b)
    u_ = u[:, range(numComponents)]
    v_ = v[range(numComponents), :]
    s_ = s[range(numComponents)]
    s_ = np.diag(s_)
    # print (u_.shape, v_.shape, s_.shape)

    temp_b = (u_ @ s_) @ v_

    temp = (abs(np.dstack((temp_r, temp_g, temp_b))))
    temp = np.uint8(temp)
    # temp_img = Image.fromarray(temp)
    return temp
   
main()