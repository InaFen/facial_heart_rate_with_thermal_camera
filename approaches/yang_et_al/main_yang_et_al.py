import sys
import os
import numpy as np

project_root = "/home/arbeit/Dokumente/master_thesis/"
sys.path.insert(0, project_root)

from approaches.new_approach.utils_landmark_detection import *
from utils_face_tracking import track_face, filter_out_pixels_below_threshold


path_video = 'recorded_videos/thermal_video_12_03.avi'
path_video_tracked_face = 'recorded_videos/yang_et_al/output_tracked_video.mp4'
path_video_thresholded_pixel="recorded_videos/yang_et_al/output_thresholded_video.mp4"
path_video_facemesh = "recorded_videos/yang_et_al/output_facemesh_video.mp4"
path_pixels_log = "saved_data/yang_et_al/roi_pixel_log.npy"
path_video_thermal_with_roi_masks = "recorded_videos/yang_et_al/output_video_with_roi_masks.mp4"
path_roi_pixel_data = "saved_data/yang_et_al/roi_pixel_data.npy"


#track_face(in_path= path_video, out_path=path_video_tracked_face)

#filter_out_pixels_below_threshold(video_path=path_video, output_path=path_video_thresholded_pixel, threshold_value=160)

# get face with mediapipe (not original pipeline)
roi_landmarks = {"whole face": [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
    127, 162, 21, 54, 103, 67, 109
]}

roi_landmarks_subgrid_face = {"temp1": [
    338, 297, 299,296,336, 337
],
"temp2": [
    10,338,337,336,9,151
],
"temp3": [
    67,69,66,109,108,107
],
"temp4": [54,68,63,105,104,103
]
}

if not os.path.isfile(path_pixels_log):
    roi_pixel_log = apply_facemesh_to_thermal_video(input_path=path_video, output_path=path_video_facemesh, roi_landmarks=roi_landmarks_subgrid_face)
    os.makedirs(os.path.dirname(path_pixels_log), exist_ok=True)
    np.save(path_pixels_log, roi_pixel_log, allow_pickle=True)
else:
    roi_pixel_log = np.load(path_pixels_log, allow_pickle=True)

if not os.path.isfile(path_roi_pixel_data):
    roi_pixel_data = apply_roi_masks_to_video(video_path=path_video, roi_mask_log=roi_pixel_log, output_path= path_video_thermal_with_roi_masks)
    os.makedirs(os.path.dirname(path_roi_pixel_data), exist_ok=True)
    np.save(path_roi_pixel_data, roi_pixel_data, allow_pickle=True)
else:
    roi_pixel_data = np.load(path_roi_pixel_data, allow_pickle=True)

plot_roi_mean_signals(roi_pixel_data=roi_pixel_data) #TODO double check if all frames are used


# TODO Schritt (zentrale Koordinaten bekommen)
for frame_idx, frame_data in enumerate(roi_pixel_data):
    for roi_name, roi_info in frame_data.items():

        coords = roi_info["coords"]
        centroid = compute_centroid(coords)

        roi_info["centroid"] = centroid

# Graph erstellen
