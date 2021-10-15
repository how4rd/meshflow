

class Frame:

    '''
    A `Frame` represents a frame of footage in a video.

    It contains the following attributes:
    `width`: The `Frame`'s width (number of columns) in pixels.
    `height`: The `Frame`'s height (number of rows) in pixels.
    `pixels_color`: A (`width` by `height` by `3`) array containing the `Frame`'s pixels in color.
    `pixels_bw`: A (`width` by `height`) array containing the `Frame`'s pixels in black and white.
    `features`: a list of the coordinates in the `Frame` corresponding to features.
    `velocities`: A (`MESH_WIDTH + 1` by `MESH_HEIGHT + 1`) array containing the velocity of each
        node relative to the previous `Frame`. `None` if no such `Frame` exists or not yet computed.
    '''

    # The width and height (number of columns and rows) in the frame's mesh
    # note that there are (`MESH_WIDTH + 1`) nodes per row, and (`MESH_WIDTH + 1`) per column.
    MESH_WIDTH = 16
    MESH_HEIGHT = 16

    # The width and height (number of columns and rows) in the frame's mesh when breaking it down
    # into subregions to eliminate outlying regions
    OUTLIER_SUBREGIONS_WIDTH = 4
    OUTLIER_SUBREGIONS_HEIGHT = 4

    def __init__(self, pixels_color, fast_feature_detector):
        raise NotImplementedError

    def compute_unstabilized_mesh_velocities(self, feature_detector, next_frame=None):
        '''
        Given a feature detector (cv.Feature2D) and the next `Frame` in the video, estimate the
        velocity of the nodes in this `Frame` and set the ``, ``, ``
        '''
        # TODO find velocities:

        # detect features using the given feature detector, then move compare them with the previous
        # frame using calcOpticalFlowPyrLK

        # calculate global homography between prev_frema and current frame

        # remove outlier features by splitting into subregions and running ransac on each

        # calculate velociities by looking at velocities of each feature between previous frames

        raise NotImplementedError