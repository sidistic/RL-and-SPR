import cv2
from imagePCA import fullProcess, trial2

numberFrames = 120
numberComponents = 200
FPS = 24
width = 1920
height = 1080

fourcc = cv2.VideoWriter_fourcc(*'MP42')
video = cv2.VideoWriter('compressedCars_200.avi', fourcc, float(FPS), (width, height))

# Function to extract frames 
def compressVideo(path): 
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1
    while success and count < numberFrames:
        count += 1
        print (count, flush=True)
        success, image = vidObj.read()
        # print (image.shape, type(image))
        new_image = trial2(image, numberComponents)
        # print (type(new_image))
        video.write(new_image)
    video.release()

  
if __name__ == '__main__': 
    compressVideo('test_cars.mp4') 