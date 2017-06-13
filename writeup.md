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
[image4_0]: ./output_images/sliding_window_0.jpg
[image4_1]: ./output_images/sliding_window_1.jpg
[image4_2]: ./output_images/sliding_window_2.jpg
[image4_3]: ./output_images/sliding_window_3.jpg
[image4_4]: ./output_images/sliding_window_4.jpg
[image4_5]: ./output_images/sliding_window_5.jpg
[image_heatmap_0]: ./output_images/heatmap_0.jpg
[image_heatmap_1]: ./output_images/heatmap_1.jpg
[image_heatmap_2]: ./output_images/heatmap_2.jpg
[image_heatmap_3]: ./output_images/heatmap_3.jpg
[image_heatmap_4]: ./output_images/heatmap_4.jpg
[image_heatmap_5]: ./output_images/heatmap_5.jpg
[image_heatmap_5_integrated]: ./output_images/heatmap_integrated_5.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video_processed.mp4

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

I tested:
`orientations=9`, `pixels_per_cell=(8, 8)` `cells_per_block=(3, 3)` : Score 0.995 on test data

In the end I settled with YUV.

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


I search the bottom half of the image only with the sliding window search, and only search close to the horizon for the smallest windows.  I decided to overlap by 0.6 except for the very smallest windows in the vertical direction which I increased to an 80% overlap.  I used just 2 different sizes,  64x64 and 96x96.  I did try other sizes, but found that they didn't add to the performance.  The high overlap was necessary to get good accuracy on the video.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4_0]

![alt text][image4_1]

![alt text][image4_2]

![alt text][image4_3]

![alt text][image4_4]

![alt text][image4_5]

My pipeline is quite slow however.  I initially had much larger windows as well, but found that these did nothing for the accuracy, and instead added in some false positives, so I removed these.

I did immplement the suggested caching of the hog_features, but after timing it both ways, I found that it did not provide any performance increase, so I disabled it.  It is still in the code however. (In `extract_features()` in `hog.ipynb`).

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_processed.mp4)

And with a [link to my video result with a heat map overlaid](./project_video_processed_with_heat.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In `hog.ipynb` in `heat_map_test_image()` I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

I overlaid the heatmap on to the image.

Here's an example from the 6 test images:

![alt text][image_heatmap_0]

![alt text][image_heatmap_1]

![alt text][image_heatmap_2]

![alt text][image_heatmap_3]

![alt text][image_heatmap_4]

![alt text][image_heatmap_5]


For the video, I re-use the heat map for each frame, but add a decay to the heat map by decreasing the values by 50% after each frame.  This effectively gives me a decayed integration between frames.  I found that quite a high decay works best.  I also had to increase the threshold to 3 to filter out the occasional false positives when using the integration method.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here the resulting bounding boxes and heat map are drawn onto the last frame in the series:
![alt text][image_heatmap_5_integrated]
You can see that it has elongated the labelled box for the cars.  This is with a decay of 50%.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue that I have is performance.  On my laptop, it takes 4 hours to process the 50 second long video!

To deal with this, I decreased the overlap for testing and development.  I also implemented a cache for the hog features but this didn't help with performance.

The pipeline is likely to fail in many places, such as detecting cars closely in front, since I don't have any large windows.  It also fails to correctly identify two cars as two cars when they are too close or one is partially obscuring the other.  One possible solution for this could be to add a basic motion model - Once we detect a car, we could detect its direction of motion, and assume that it will travel roughly in the same direction.  And so we can approximately track even a fully occluded car.
