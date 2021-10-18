from collections import deque
import cv2

from Frame import Frame


# the max number of frames to store in the frame buffer
FRAME_BUFFER_MAX_LENGTH = 40


def get_next_frame(video):
    '''
    Given a VideoCapture containing footage to stabilize, read in the next frame if available.
    Return the frame if available and None otherwise.
    '''

    frame_successful, pixels_bgr = video.read()
    if not frame_successful:  # all the video's frames had already been read
        return None

    return Frame(pixels_bgr)


def play_video(video, window_name):
    '''
    Play the given cv2.VideoCapture in a window with the given name.
    '''

    # get frame rate;
    # adapted from https://learnopencv.com/how-to-find-frame-rate-or-frames-per-second-fps-in-opencv-python-cpp/
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_length_ms = int(1000 / fps)

    # read in video;
    # adapted from https://www.geeksforgeeks.org/python-play-a-video-using-opencv/
    while video.isOpened():
        frame = get_next_frame(video)
        if frame is None:
            print('video is done playing')
            break
        else:
            cv2.imshow(window_name, frame.pixels_bgr)
            # close window when q pressed; see https://stackoverflow.com/a/57691103
            if cv2.waitKey(frame_length_ms) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


def main():
    # TODO get video path from command line args
    video_path = 'videos/data_small-shaky-5.avi'

    video = cv2.VideoCapture(video_path)

    prev_frame = get_next_frame(video)
    if prev_frame is None:
        raise IOError(f'Video at <{video_path}> does not contain any frames.')

    frame_buffer = deque([prev_frame], maxlen=FRAME_BUFFER_MAX_LENGTH)
    # see https://stackoverflow.com/a/42618215
    fast_feature_detector = cv2.FastFeatureDetector_create()

    while video.isOpened():
        frame = get_next_frame(video)
        if frame is None:
            break

        frame_buffer.append(frame)

        prev_frame.compute_unstabilized_mesh_velocities(fast_feature_detector, frame)

    video.release()


if __name__ == '__main__':
    main()
