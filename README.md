

# DeepSeg

These are the code and data for the paper: [DeepSeg: Deep Learning-based Activity Segmentation Framework for Activity Recognition using WiFi](https://github.com/ChunjingXiao/DeepSeg/blob/master/DeepSeg_JIoT_Online.pdf), IEEE Internet of Things Journal, 2020. https://ieeexplore.ieee.org/document/9235578

DeepSeg aims at segmenting activities for WiFi Channel State Information (CSI)-based activity recognition.

Because fluctuation ranges of CSI amplitudes when activities occur are much larger than that when no activity presents, most existing works focus on designing threshold-based segmentation methods, which attempt to seek an optimal threshold to detect the start and end of an activity. If the fluctuation degree of CSI waveforms exceeds this threshold, an activity is considered to happen.

However, there exist some weaknesses for these threshold-based segmentation methods.
First, policies of noise removal and threshold calculation are usually determined based on subjective observations and experience, and some recommended policies might even be conflicted. Second, threshold-based segmentation methods may suffer from significant performance degradation when applying to the scenario including both fine-grained and coarse-grained activities. Third, motion segmentation and activity classification, which are closely interrelated, are usually treated as two separate states.

DeepSeg tries to adopt deep learning techniques to address these problems. DeepSeg is composed of the motion segmentation algorithm and the activity classification model. The descriptions about the codes are shown as follows:


# 02ExtractCsiAmplitude
This is used to extract amplitudes from raw CSI *.dat files, and save as *.mat files 

# 03DataCsiAmplitudeCut
This is used to cut the rows of data


# 04ManulSegmentActivity
This is used to manually mark start and end points of activities.


# 05ExtractSegmentTrainData
This is used to extract training data for the motion segmentation algorithm. 


# 06DiscretizeCsiForSegment
This is used to discreize continuous CSI data into bins  for segmentation


# 07ExtractActivitySample
This is used to extract training data for the activity classification model.

# 11CnnClassifyActivity
This is for training the activity classification model.

# 12CnnSegmentCode
This is for training the state inference model.

# 32FeedBackPython
This is for the joint training of the motion segmentation algorithm and the activity classification model.



