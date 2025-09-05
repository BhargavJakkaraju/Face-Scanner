import cv2

algorimith = '/Users/bhargavjakkaraju/Face-Scanner/haarcascade_frontalface_default.xml'

haar_cascade = cv2.CascadeClassifier(algorimith)

file_name = '/Users/bhargavjakkaraju/Face-Scanner/face-image.jpeg'
img = cv2.imread(file_name, 0)
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#detect face
faces = haar_cascade.detectMultiScale (
    gray_img, scaleFactor = 1.05, minNeighbor = 5, minSize=(100,100)
)

i = 0
for x, y, w, h in faces:
    cropped_image = img[y: y + h, x: x + w]
    target_file_name = 'stored-faces/' + str(i) + '.jpg'
    cv2.imwrite(
        target_file_name,
        cropped_image,
    )
    i = i + 2

