# Video-Segmentation

The purpose of this project is to develop an object identification algorithm using the DPMMClustersStreaming algorithm. The algorithm aims to identify a specific object in a video by analyzing
its characteristics and distinguishing it from the background. The objective of the algorithm is
to accurately identify and highlight the object of interest in different frames of the video. The
algorithm accomplishes this by utilizing the DPMMClustersStreaming algorithm, which clusters
pixels based on their similarity and assigns them to different groups.
The dpmmpythonStreaming library focuses on semi-supervised video object segmentation, where
the initial frame contains a mask of the object, and subsequent frames are segmented in an unsupervised manner. The DPMMClustersStreaming algorithm can benifit our project since each
cluster is associated with a set of Gaussian distributions that represent the visual characteristics
of the pixels within that cluster. These Gaussian distributions are used to make decisions about
where a pixel should be assigned during the segmentation process.
