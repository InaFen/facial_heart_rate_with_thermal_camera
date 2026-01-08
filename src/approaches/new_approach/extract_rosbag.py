import rosbag
import cv2
from cv_bridge import CvBridge

bag_path = "recorded_videos/ros_files/LCAS_Physiological_Monitoring_Dataset_rosbagfiles/2016-11-22-14-09-37-Joao_onlythermal.bag"
image_topic = "/camera/image_raw"
output_video = "output.avi"

bridge = CvBridge()
bag = rosbag.Bag(bag_path)

video_writer = None
fps = 30  # adjust if needed

for topic, msg, t in bag.read_messages(topics=[image_topic]):
    # Convert ROS Image to OpenCV
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    if video_writer is None:
        height, width, _ = cv_image.shape
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video_writer = cv2.VideoWriter(
            output_video, fourcc, fps, (width, height)
        )

    video_writer.write(cv_image)

bag.close()
video_writer.release()
print("Video saved as", output_video)
