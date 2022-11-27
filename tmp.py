import cv2
from PIL import Image
from torchvision import transforms

cv2_img = cv2.imread('Project-StylizedNeRF/tmp.png')
print('Before cv2 resize: ', cv2_img.shape)
cv2_new_img = cv2.resize(cv2_img, (8, 8))
print('After cv2 resize: ', cv2_new_img.shape)


resize = transforms.Resize(8)
toTensor = transforms.PILToTensor()
pil_img = Image.open('Project-StylizedNeRF/tmp.png').convert('RGB')
print('Before pil transform: ', toTensor(pil_img).shape)
pil_new_img = toTensor(resize(pil_img))
print('After pil transform: ', pil_new_img.shape)


