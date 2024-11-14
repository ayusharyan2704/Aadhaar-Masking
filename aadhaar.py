import cv2
import pytesseract
import re
from scipy import ndimage
import matplotlib.pyplot as plt

def rotate(image, center=None, scale=1.0):
    angle = int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(image)).group(0))
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    rotated = ndimage.rotate(image, float(angle) * -1)
    return rotated


def preprocessing(image):
    w, h = image.shape[0], image.shape[1]
    if w < h:
        image = rotate(image)
    resized_image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    grey_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.medianBlur(grey_image, 3)
    thres_image = cv2.adaptiveThreshold(blur_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 13, 7)
    return thres_image, resized_image

def aadhar_mask_and_ocr(thres_image, resized_image):
    d = pytesseract.image_to_data(thres_image, output_type=pytesseract.Output.DICT)
    number_pattern = r"(?<!\d)\d{4}(?!\d)"
    n_boxes = len(d['text'])
    c = 0
    temp = []
    UID = []
    
    for i in range(n_boxes):
        if int(d['conf'][i]) > 20:  
            if re.match(number_pattern, d['text'][i]):
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                if c < 2:
                    resized_image = cv2.rectangle(resized_image, (x, y), (x + w, y + h), (255, 255, 255), -1)
                    temp.append(d['text'][i])
                    c += 1
                elif c == 2:
                    UID = temp + [d['text'][i]]
                    c += 1

    if UID:
        final_image = cv2.resize(resized_image, None, fx=0.33, fy=0.33)
        return final_image, UID
    else:
        return resized_image, UID  

image_path = 'image3.jpeg'  
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()

thres_image, resized_image = preprocessing(image)
masked_image, UID = aadhar_mask_and_ocr(thres_image, resized_image)


try:
    cv2.imshow('Masked Image we get:', masked_image)

    output_path = 'masked_' + image_path
    cv2.imwrite(output_path, masked_image)
    print(f'Masked image saved as {output_path}')


    if UID:
        print(f'UID : {" ".join(UID)}')
    else:
        print('No UID detected.')

    print('Press q over output window to close.')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception as e:

    print(f'OpenCV display failed: {e}. Displaying with Matplotlib instead.')
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    plt.title('Masked Image')
    plt.axis('off')  
    plt.show()
