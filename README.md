# PennAiRSoftwareChallenge
Completing the PennAiR software challenge to apply to the software team

In general the .ipynb files are used for initial testing and run simultenously on 12 selected frames from a video or a photo if one was given. The .py files have more or less the same code as the .ipynb files but run on the entire video at the same time.

**The main file to look at is HSVsplitHard.py. It contains the final version of my code for processing the agnostic video. Read the description at the top of that file for a clear explanation of my processing method and instructions on how to run the code yourself if you want.**

**Agnostic output**
<video src="agnosticOutput1.mp4" width="320" height="240" controls></video>

The main.ipynb and test.ipynb files contain the original code to process the easy image.

The main.py file contains the original code for running on the easy video.

The mainHard.py and HSVsplitHard.py files contain different ways of processing the hard video.
 - The mainHard.py file uses a simple canny + dilate + erode + contour algorithm which gives a shaky binary mask output

The hsvTesting.ipynb file contains the same code as the HSVsplitHard.py file but in Jupyter cells
 - The HSVsplitHard.py file contains a method that performs significantly better than mainHard.py but relies on the polygons either being a different hue, saturation, or (very different) value than the background. The trapezoid (since it's white and black and hence has a saturation of zero) has slightly worse edges and, instead of using an HSV mask, uses Otsu's method after an HSV binary mask and some Gaussian blurring. The point is to wash out the noise to then filter it from the larger trapezoid since it should appear relatively separate in the grayscale distribution. I then union the color mask and grayscale mask together, detect contours, filter small and oddly shapes contours, and finally overlay a canny edge detection output of this binary mask onto the original image to get green outlines around the polygons.
 - hsvTesting generates four videos simultaneously (they are from different steps throughout the entire process) and runs at ~35fps, best one I got was 37.02 fps


I didn't fully understand the intrinsic matrix stuff. I wrote this function to calculate depth but I'm not sure if it's accurate. I also didn't understand how to get the "real" dimensions of the other shapes——won't you have to assume their sizes, since if you use a proportional value with the pixel sizes then they'll all be at the same depth?

```python
import numpy as np

# Camera intrinsics
fx = 2564.3186869
fy = 2569.70273111

def compute_3d_center(u, v, r_pixels, r_real_in_inches):
    # Compute depth
    Z = fx * r_real_in_inches / r_pixels  # or use fy, they are close
    # Compute real-world X, Y
    X = Z * u / fx
    Y = Z * v / fy
    return X, Y, Z
```