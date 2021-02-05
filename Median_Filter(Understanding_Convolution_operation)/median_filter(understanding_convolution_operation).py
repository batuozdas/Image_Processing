import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
# Reading images with different formats.
img_original = cv2.imread('resimler/for_work/wesley_sneijder.png',1)
img_original_rgb = cv2.cvtColor(img_original,cv2.COLOR_BGR2RGB)
noisy_img_bgr = cv2.imread('resimler/for_work/wesley_sneijder_noisy.png',1)
noisy_img_hsv = cv2.cvtColor(noisy_img_bgr,cv2.COLOR_BGR2HSV)
noisy_img_rgb = cv2.cvtColor(noisy_img_bgr,cv2.COLOR_BGR2RGB)
# We apply median filter to v channel of our noisy img.
h,s,v = cv2.split(noisy_img_hsv)

# We create our function. There will be 2 values. arr: gray img (v channel of hsv image), ksize = kernel size
def median_filter(arr,ksize):
    row = arr.shape[0]
    col = arr.shape[1]
    ksize_values = np.arange(3,np.min((row,col)),2) # We create ksize_values to use later as a warning.
    index = np.where(ksize_values == ksize) # We create index value to use later as a padding.
    if ksize not in ksize_values: # If ksize not in ksize_values (ksize!= 3,5,7,9,11,...);
        print('ksize must be these values:{}'.format(ksize_values))
    else:
        arr_pad = np.pad(arr,pad_width=index[0]+1,mode='edge') # Padding. If ksize = 3, pad_width = 1. If ksize = 5, pad_width = 2, If ksize = 7, pad_width = 3,...
        median_values = [] # We create empty list called median_values to fill later with median values.

        # Convolution Operation: We create a (ksize x ksize) kernel matris. Then this kernel matris moves in our arr value. In every move, we will take median value of
        # that kernel matris. And we will add these median values to median_values list. After the loop is over, we will reshape median_values list with the shape of
        # arr (v channel of noisy hsv image) shape.
        for i in range(row):
            for j in range(col):
                k_size_array = arr_pad[i:i+ksize,j:j+ksize]
                median_value = np.median(k_size_array)
                median_values.append(median_value)
        return_array = np.array(median_values).reshape(arr.shape)
        return_array = np.uint8(return_array)
        return return_array
cv2_start_time = time.time()
cv2_median_v = cv2.medianBlur(v,ksize=5) # We apply medianBlur to our v channel of hsv noisy image.
cv2_close_time = time.time()
func_start_time = time.time()
func_median_v = median_filter(v,ksize=5) # We apply our function called median_filter to our v channel of hsv noisy image.
func_close_time = time.time()

# We merge channels.
cv2_median_img_hsv = cv2.merge((h,s,cv2_median_v))
func_median_img_hsv = cv2.merge((h,s,func_median_v))

# HSV to RGB
cv2_median_img_rgb = cv2.cvtColor(cv2_median_img_hsv,cv2.COLOR_HSV2RGB)
func_median_img_rgb = cv2.cvtColor(func_median_img_hsv,cv2.COLOR_HSV2RGB)

# Results
print("Median filter takes {} seconds with Opencv Library.".format(cv2_close_time-cv2_start_time))
print("Median filter takes {} seconds with function we created.".format(func_close_time-func_start_time))
fig,((ax1,ax2,ax3,ax4)) = plt.subplots(ncols=4,figsize=(16,9))
ax1.imshow(img_original_rgb)
ax1.set_title('Original Image')
ax2.imshow(noisy_img_rgb)
ax2.set_title('Noisy Image')
ax3.imshow(cv2_median_img_rgb)
ax3.set_title('Opencv Median Filter')
ax4.imshow(func_median_img_rgb)
ax4.set_title('Function Median Filter')
plt.tight_layout()
plt.show()








