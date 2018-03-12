This is a tensorflow implemenation of [Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior).

# Setup
- Install python ~~2.6~~ 3.6, [tensorflow](https://www.tensorflow.org/install/) (tested on ~~1.4.0~~ 1.5.0+), numpy~~, and scipy~~.
- Run it: `python deepimg.py`
- Every 100 iterations, the current image is written to the `output` directory.
  - The input image will be blurred and written to `output/corrupted.png`. This is the starting image that the model attempts to sharpen.

# Known issues and discrepancies
- This only implements super resolution.
- This uses a Gaussian blur rather than a Lanczos2 kernel for the downsampling operator in E.
- ~~The output images suffer from a checkerboard artifact.~~

# Example
![Blurred pupper](corrupt.png?raw=true "Original")
![2000 iterations pupper](2000_pupper.png?raw=true "After 2000 iterations")
![4000 iterations pupper](4000_pupper.png?raw=true "After 4000 iterations")
