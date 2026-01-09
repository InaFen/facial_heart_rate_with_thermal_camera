# Get a thermal video with a consistent FPS

## First step: Record frames with src/video_processing_methods/capture_lepton_timestamps
e.g. 
capture_lepton_timestamps \ 

  --device /dev/v4l/by-id/usb-GroupGets_PureThermal__fw:v1.3.0__80030034-5101-3133-3332-373300000000-video-index0 \ 

  --duration 10 \ 

  --telemetry end \ 

  --width 160 --height 122 \ 

  --save-ts /home/arbeit/Dokumente/master_thesis/saved_data/raw_lepton_data/frame 

## Second step: Interpolate if needed to get consistent FPS
src/video_processing_methods/get_consistent_fps_for_gray_images.py

## Third step:
Create video with src/video_processing_methods/thermal_video_generator.py

The video is used to get the ROIs but the .gray files are used to get actual temperature

 