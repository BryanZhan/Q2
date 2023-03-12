import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img1 = cv.imread('Fig0427_a.bmp',0)
f1 = np.fft.fft2(img1)
fshift1 = np.fft.fftshift(f1)
phase_spectrum1 = np.angle(f1)
magnitude_spectrum1 = 20*np.log(np.abs(fshift1))
plt.imshow(img1, cmap = 'gray')
plt.title('Input Image')
plt.show()

plt.title('Magnitude Spectrum')
plt.imshow(magnitude_spectrum1, cmap = 'gray')
plt.show()

cv.imwrite("Q2b.jpg",np.uint8(phase_spectrum1))
plt.imshow(phase_spectrum1, cmap = 'gray')
plt.title('(b) phase angle')
plt.show()

img2 = cv.imread('Fig0424_a.bmp',0)
img2=cv.resize(img2, (512,512))
f2 = np.fft.fft2(img2)
fshift2 = np.fft.fftshift(f2)
phase_spectrum2 = np.angle(f2)
magnitude_spectrum2 = 20*np.log(np.abs(fshift2))

plt.imshow(img2, cmap = 'gray')
plt.title('Input Image')
plt.show()

plt.imshow(magnitude_spectrum2, cmap = 'gray')
plt.title('Magnitude Spectrum')
plt.show()

plt.imshow(phase_spectrum2, cmap = 'gray')
plt.title('phase')
plt.show()

combined = np.exp(1j*np.angle(f1))
imgCombined = np.real(np.fft.ifft2(combined))
plt.title('(c) Woman reconstructed using only the phase')
plt.imshow(imgCombined, cmap='gray')
plt.show()
plt.imsave("Q2c.jpg",imgCombined,cmap='gray')

imgCombined = np.real(np.fft.ifft2(np.abs(f1)))
plt.title('(d) Woman reconstructed using only the sspectrum')
plt.imshow(imgCombined, cmap='gray')
plt.show()
plt.imsave("Q2d.jpg",imgCombined,cmap='gray')

combined = np.multiply(np.abs(f2), np.exp(1j*np.angle(f1)))
imgCombined = np.real(np.fft.ifft2(combined))
plt.title('(e)')
#print('(e) reconstruction using the phase angle corresponding to rhe woman and the spectrum corresponding to the rectangle in Fig0424')
plt.imshow(imgCombined, cmap='gray')
plt.show()
plt.imsave("Q2e.jpg",imgCombined,cmap='gray')

combined = np.multiply(np.abs(f1), np.exp(1j*np.angle(f2)))
imgCombined = np.real(np.fft.ifft2(combined))
plt.title('(f)')
#print('(f) Reconstruction using the phase of the rectangle and the spectrum of the woman')
plt.imshow(imgCombined, cmap='gray')
plt.show()
plt.imsave("Q2f.jpg",imgCombined,cmap='gray')