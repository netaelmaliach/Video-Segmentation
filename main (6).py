from julia.api import Julia
jl = Julia(compiled_modules=False)
from dpmmpythonStreaming.dpmmwrapper import DPMMPython
from dpmmpythonStreaming.priors import niw
import numpy as np
import cv2
import os
from PIL import Image
from scipy.stats import multivariate_normal


# Helper function, like example from julia package, change the image to 2D, so we can use fit
def pre_process_image(img):
    dims = img.shape
    histogram_arr = np.zeros((5, dims[0], dims[1]), dtype=np.float32)

    for i in range(dims[0]):
        for j in range(dims[1]):
            histogram_arr[:3, i, j] = img[i, j]
            histogram_arr[3, i, j] = i / dims[0]
            histogram_arr[4, i, j] = j / dims[1]

    histogram_arr = histogram_arr.reshape((5, dims[0] * dims[1]))

    return histogram_arr, dims


# Helper function, checks if pixel is more likely to be part of the object or the background
def is_pixel_of_obj(info_pix, obj_model_means, obj_model_covs, bgd_model_means, bgd_model_covs):
    obj_mle = 0  # mle - maximum likelihood
    bgd_mle = 0
    # print(info_pix)

    # Get the maximum probability of the pixel being of type object or background, and return the higher one
    for i in range(len(obj_model_means)):
        obj_probability = multivariate_normal.pdf(x=info_pix, mean=obj_model_means[i], cov=obj_model_covs[i])
        if obj_probability > obj_mle:
            obj_mle = obj_probability

    for i in range(len(bgd_model_means)):
        bgd_probability = multivariate_normal.pdf(x=info_pix, mean=bgd_model_means[i], cov=bgd_model_covs[i])
        if bgd_probability > bgd_mle:
            bgd_mle = bgd_probability

    if obj_mle > bgd_mle:
        return True
    else:
        return False


def test():

    # load the video and extract the first frame
    video_path = '/Users/roro/Documents/unsupervised_learning/pythonProject/dog3.mp4'
    video_capture = cv2.VideoCapture(video_path)

    # turn video into frames
    frames_dir = '/Users/roro/Documents/unsupervised_learning/pythonProject/frames'
    os.makedirs(frames_dir, exist_ok=True)

    frame_count = 0
    ret, frame = video_capture.read()  # read first frame

    # save frame as image
    frame_filename = f'frame_{frame_count}.jpg'
    frame_path = os.path.join(frames_dir, frame_filename)
    cv2.imwrite(frame_path, frame)

    # display the first frame, since it's the first frame have the user select the object to track
    window_name = 'Select Object'
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, frame)

    # ask the user to select the object by drawing a box
    object_box = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    # extract the object foreground using grabcut
    mask = np.zeros(frame.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(frame, mask, object_box, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
    print("Thank you!")

    # create a binary mask where the foreground pixels are 1 and the background pixels are 0
    mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # apply binary mask to the frame to extract the object and background
    object_frame = frame * mask_binary[:, :, np.newaxis]
    background_frame = frame * (1 - mask_binary[:, :, np.newaxis])

    # save the extracted object and background as separate images
    output_path = '/Users/roro/Documents/unsupervised_learning/pythonProject/object.png'
    cv2.imwrite(output_path, object_frame)
    output_path = '/Users/roro/Documents/unsupervised_learning/pythonProject/background.png'
    cv2.imwrite(output_path, background_frame)

    object_frame = cv2.imread('object.png')
    background_frame = cv2.imread('background.png')

    # preprocess each frame
    obj_data, dims = pre_process_image(object_frame)
    bgd_data, dims = pre_process_image(background_frame)

    # @TODO
    prior = DPMMPython.create_prior(5, 0, 1, 1, 1)

    # on the first frame we need fit_init
    obj_model = DPMMPython.fit_init(obj_data, 100.0, prior=prior, burnout=5, gt=None, epsilon=0.0000001)
    bgd_model = DPMMPython.fit_init(bgd_data, 100.0, prior=prior, burnout=5, gt=None, epsilon=0.0000001)

    # to save information of each model
    obj_model_means = []
    obj_model_covs = []
    bgd_model_means = []
    bgd_model_covs = []

    # save the mean and cov of object model
    for i in range(len(obj_model.group.local_clusters)):
        obj_model_means.append(obj_model.group.local_clusters[i].cluster_params.cluster_params.distribution.μ)
        obj_model_covs.append(obj_model.group.local_clusters[i].cluster_params.cluster_params.distribution.Σ)

    # save the mean and cov of background model
    for i in range(len(bgd_model.group.local_clusters)):
        bgd_model_means.append(bgd_model.group.local_clusters[i].cluster_params.cluster_params.distribution.μ)
        bgd_model_covs.append(bgd_model.group.local_clusters[i].cluster_params.cluster_params.distribution.Σ)

    frame_count += 1

    # initialize output video
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('output.mp4', fourcc=fourcc, fps=20.0, frameSize=(frame_width, frame_height))

    # make directory to save new frames
    new_frames_dir = '/Users/roro/Documents/unsupervised_learning/pythonProject/new_frames'
    os.makedirs(new_frames_dir, exist_ok=True)

    while True:
        print(f"Calculating frame number {frame_count}")

        # read next frame
        ret, frame = video_capture.read()

        # if the frame wasn't successfully read
        if not ret:
            break

        # save frame as an image
        frame_filename = f'frame_{frame_count}.jpg'
        frame_path = os.path.join(frames_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        # Go over the pixels in the frame, send to function that determines if
        # the pixel is part of the object or background using the gaussians from the beginning.
        # each pixel is saved in object_data or background_data according to the return value

        updated_object_frame_data = []
        updated_bgd_frame_data = []
        paint_object = np.zeros_like(frame)  # to highlight the object

        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                info_pix = np.zeros(5, dtype=np.float32)
                info_pix[:3] = frame[i, j]
                info_pix[3] = i / dims[0]
                info_pix[4] = j / dims[1]
                if is_pixel_of_obj(info_pix, obj_model_means, obj_model_covs, bgd_model_means, bgd_model_covs):
                    # print("Object pixel")
                    updated_object_frame_data.append(info_pix)
                    paint_object[i, j] = [128, 0, 128]  # purple
                else:
                    # print("Background pixel")
                    updated_bgd_frame_data.append(info_pix)

        # Add the highlight over the object
        new_frame = cv2.addWeighted(paint_object, 0.3, frame, 0.6, 0)

        # save new frame as image
        new_frame_name = f"new_frame_{frame_count}.jpg"
        new_frame_path = os.path.join(new_frames_dir, new_frame_name)
        cv2.imwrite(new_frame_path, new_frame)

        # display the resulting frame
        output_video.write(new_frame)
        cv2.imshow('Segmented Video', new_frame)

        # cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # Release the video capture object and close any open windows
    video_capture.release()
    output_video.release()
    cv2.destroyAllWindows()


def git_test():
    data, gt = DPMMPython.generate_gaussian_data(10000, 2, 10, 100.0)
    batch1 = data[:, 0:5000]
    batch2 = data[:, 5000:]
    prior = DPMMPython.create_prior(2, 0, 1, 1, 1)
    model = DPMMPython.fit_init(batch1, 100.0, prior=prior, verbose=True, burnout=5, gt=None, epsilon=0.0000001)
    labels = DPMMPython.get_labels(model)
    model = DPMMPython.fit_partial(model, 1, 2, batch2)
    labels = DPMMPython.get_labels(model)
    print(labels)


if __name__ == '__main__':
    test()

