import cv2

def get_next_frame(video):
    '''
    Given a VideoCapture containing footage to stabilize, read in the next frame if available.
    Return the frame if available and None otherwise.
    '''

    frame_successful, frame = video.read()
    if not frame_successful:  # all the video's frames had already been read
        return None

    return frame


def main():
    # TODO get video path from command line args
    video_path = 'videos/data_small-shaky-5.avi'

    video = cv2.VideoCapture(video_path)

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
            cv2.imshow(video_path, frame)
            # close window when q pressed; see https://stackoverflow.com/a/57691103
            if cv2.waitKey(frame_length_ms) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    main()
