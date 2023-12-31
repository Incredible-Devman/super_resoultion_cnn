[TensorFlow] Super-Resolution CNN 
=====

TensorFlow implementation of 'Image Super-Resolution using Deep Convolutional Network'.
## Architecture
<div align="center">
  <img src="./readme/srcnn.png" width="700">  
  <p>The architecture of the Super-Resolution Network (SRCNN).</p>
</div>
The architecture constructed by three convolutional layers, and the kernel size are 9x9, 1x1, 3x2 respectively. It used RMS loss and stochastic gradient descent opeimizer for training in this repository, but original one was trained by MSE loss (using same optimizer). The input of the SRCNN is Low-Resolution (Bicubic Interpolated) image that same size of the output image, and the output is High-Resolution.  

## Results
<div align="center">
  <img src="./readme/1000.png" width="250"><img src="./readme/10000.png" width="250"><img src="./readme/100000.png" width="250">  
  <p>Reconstructed image in each iteration (1k, 10k, 100k iterations).</p>
</div>

<div align="center">
  <img src="./readme/lr.png" width="250"><img src="./readme/100000.png" width="250"><img src="./readme/hr.png" width="250">    
  <p>Comparison between the input (Bicubic Interpolated), reconstructed image (by SRCNN), and target (High-Resolution) image.</p>
</div>

## Requirements
* Python 3.6.8  
* Tensorflow 1.14.0  
* Numpy 1.14.0  
* Matplotlib 3.1.1  
