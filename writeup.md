##Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_YUV_o:9_pc:8_cb:3.png
[image3]: ./output_images/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---

### Writeup / README


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook `hog.ipynb`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces ('RGB', 'HSV', 'YUV' and 'YCbCr') and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(3, 3)`:

![alt text][image2]

**Side Note:** In the example image for the above, it plots images of the "features" for each channel.  I have no idea what that is supposed to be plotting exactly.

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and looked at how it affected the training data score in the SVM classifier below, and picked the one with the highest score.

In the end I settled with YUV

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

For a baseline, I first tested first without the HOG features and in RGB colorspace (So just the binned RGB image and the histogram features).  And **without the StandardScalar**. This gave 3168 features per image:

    Best parameters are:         {'kernel': 'linear', 'C': 1}
    Score on training data:	     0.953
    Score on test data:          0.953

I tested adding in the HOG features and tested in 'YUV' and 'HLS' colorspaces and with hog `cells_per_block` as 2 and 3, but it always gave almost exactly the same score - 0.94 to 0.95.  I even tested with both L1 and L2 losses on the hog function, with no notable difference.

I tested with the StandardScalar and instantly got a much better result:

    Best parameters are:         {'kernel': 'linear', 'C': 1}
    Score on training data:      0.979
    Score on test data:          0.980

I have to admit that I was surprised at just how large an impact the scalar made.

Doing a GridSearchCV for the best kernel and C on this gives:

    Best parameters are:         {'kernel': 'rbf', 'C': 5}
    Score on training data:      0.995
    Score on test data:          0.998

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?


I search the bottom half of the image only with the sliding window search, and only search close to the horizon for the smallest windows.  I decided to overlap by 0.5 except for the very smallest windows which I have no overlap.  I used 4 different sizes, with 64x64 being the smallest (since that was the training image size).

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

