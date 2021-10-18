import cv2
import math
import numpy as np

class Frame:
    '''
    A `Frame` represents a frame of footage in a video.

    It contains the following attributes:
    `width`: The `Frame`'s width (number of columns) in pixels.
    `height`: The `Frame`'s height (number of rows) in pixels.
    `pixels_bgr`: A (`width` by `height` by `3`) array containing the `Frame`'s pixels in BGR.
    `pixels_gray`: A (`width` by `height`) array containing the `Frame`'s pixels in grayscale.
    `mesh_velocities`: A (`MESH_COL_COUNT + 1` by `MESH_ROW_COUNT + 1`) array containing the
        velocity of each node relative to the previous `Frame`. `None` if no such `Frame` exists or
        not yet computed.
    `mesh_col_width`: The width in pixels of each column in the mesh.
    `mesh_row_height`: The height in pixels of each row in the mesh.
    '''


    # How many rows and columns the frame's mesh contains;
    # note that there are (`MESH_COL_COUNT + 1`) nodes per row, and (`MESH_COL_COUNT + 1`) per column.
    MESH_COL_COUNT = 16
    MESH_ROW_COUNT = 16


    # The width and height (number of columns and rows) in the frame's mesh when breaking it down
    # into subregions to eliminate outlying regions
    OUTLIER_SUBREGIONS_ROW_COUNT = 4
    OUTLIER_SUBREGIONS_COL_COUNT = 4


    # The width and height of the ellipse drawn around each feature to match it with mesh nodes,
    # expressed in units of mesh rows and columns.
    FEATURE_ELLIPSE_WIDTH_MESH_COLS = 3
    FEATURE_ELLIPSE_HEIGHT_MESH_ROWS = 3


    # The minimum number of corresponding features that must correspond between two frames to
    # perform a homography
    HOMOGRAPHY_MIN_NUMBER_CORRESPONDING_FEATURES = 4


    def __init__(self, pixels_bgr):
        self.pixels_bgr = pixels_bgr
        self.pixels_gray = cv2.cvtColor(pixels_bgr, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.pixels_gray.shape
        self.mesh_velocities = None
        self.mesh_col_width = self.width / self.MESH_COL_COUNT
        self.mesh_row_height = self.height / self.MESH_ROW_COUNT


    def compute_unstabilized_mesh_velocities(self, feature_detector, next_frame=None):
        '''
        Given a feature detector (cv2.Feature2D) and the next `Frame` in the video (or `None` if
        none exists), estimate the velocity of the nodes in this `Frame` and set the `velocities`
        attribute with the result.
        '''

        if next_frame is None:
            return

        # break image into sub-images and find the non-outlying features in each by applying a
        # homography using RANSAC
        # TODO parallelize

        # each item is a tuple of a feature and its velocity
        all_features_with_velocities = []

        window_width = math.ceil(self.width / self.OUTLIER_SUBREGIONS_COL_COUNT)
        window_height = math.ceil(self.height / self.OUTLIER_SUBREGIONS_ROW_COUNT)

        # mesh_node_velocities[row][col] contains a list of tuples (v_x, v_y) corresponding to all
        # the velocities of features nearby the node at the given row and column
        mesh_node_velocities = [[[] for _ in range(self.MESH_COL_COUNT + 1)] for _ in range(self.MESH_ROW_COUNT + 1)]

        for window_left_x in range(0, self.width, window_width):
            for window_top_y in range(0, self.height, window_height):
                # gather features
                current_window_feature_positions, next_window_feature_positions = self._get_unstabilized_features_in_window(
                    feature_detector, next_frame,
                    window_left_x, window_top_y, window_width, window_height
                )

                if current_window_feature_positions is None:
                    break

                # calculate features' velocities
                current_window_velocities = next_window_feature_positions - current_window_feature_positions
                # see https://stackoverflow.com/a/44409124
                current_window_positions_velocities = np.c_[current_window_feature_positions, current_window_velocities]

                # apply features' velocities to nearby mesh nodes
                for feature_position_and_velocity in current_window_positions_velocities:
                    # the feature's coordinates in units of mesh rows and columns
                    feature_col = feature_position_and_velocity[0][0] / self.mesh_col_width
                    feature_row = feature_position_and_velocity[0][1] / self.mesh_row_height
                    feature_velocity = (feature_position_and_velocity[0][2], feature_position_and_velocity[0][3])

                    # Draw an ellipse around each feature
                    # of width self.FEATURE_ELLIPSE_WIDTH_MESH_COLS
                    # and height self.FEATURE_ELLIPSE_HEIGHT_MESH_ROWS,
                    # and apply the feature's velocity to all mesh nodes that fall within this
                    # ellipse.
                    # To do this, we can iterate through all the rows that the ellipse covers.
                    # For each row, we can use the equation for an ellipse centered on the
                    # feature to determine which columns the ellipse covers. The resulting
                    # (row, column) pairs correspond to the nodes in the ellipse.
                    ellipse_top_row_inclusive = max(0, math.ceil(feature_row - self.FEATURE_ELLIPSE_HEIGHT_MESH_ROWS / 2))
                    ellipse_bottom_row_exclusive = 1 + min(self.MESH_ROW_COUNT, math.floor(feature_row + self.FEATURE_ELLIPSE_HEIGHT_MESH_ROWS / 2))

                    # ellipse_visualization = ['.' * (self.MESH_COL_COUNT + 1) for _ in range(self.MESH_ROW_COUNT + 1)]
                    for node_row in range(ellipse_top_row_inclusive, ellipse_bottom_row_exclusive):
                        # half-width derived from ellipse equation
                        ellipse_slice_half_width = self.FEATURE_ELLIPSE_WIDTH_MESH_COLS * \
                            math.sqrt((1/4) - ((node_row - feature_row) / self.FEATURE_ELLIPSE_HEIGHT_MESH_ROWS) ** 2)
                        ellipse_left_col_inclusive = max(0, math.ceil(feature_col - ellipse_slice_half_width))
                        ellipse_right_col_exclusive = 1 + min(self.MESH_ROW_COUNT, math.floor(feature_col + ellipse_slice_half_width))

                        for node_col in range(ellipse_left_col_inclusive, ellipse_right_col_exclusive):
                            # ellipse_visualization[node_row] = ellipse_visualization[node_row][:node_col] + '*' + ellipse_visualization[node_row][node_col+1:]
                            mesh_node_velocities[node_row][node_col].append(feature_velocity)

                    # print('\n'.join(ellipse_visualization))
                    # print('\n')

        # TODO perform first median filter: sort each node's velocities and use the median
        # TODO perform second median filter: replace each node's velocity with the median velocity
        # of its neighbors
        print(mesh_node_velocities)





        # TODO match velocities to mesh nodes
        # for each feature, inspect its coordinates to determine which nodes it applies to

    def _get_unstabilized_features_in_window(self, feature_detector, next_frame, window_left_x, window_top_y, window_width, window_height):
        '''
        Helper function for compute_unstabilized_mesh_velocities.
        Given a feature detector (cv2.Feature2D), the next `Frame` in the video, and parameters
        specifying a window of the video during the current and the next frame, return a tuple
        `(current_features, next_features)`.
        `current_features` is an CV_32FC2 array (see https://stackoverflow.com/a/47617999)
        containing the coordinates of each non-outlying feature in the current frame that was
        successfully tracked into the next frame, and `next_features` contains those features'
        coordinates in the next frame.

        If not enough features to perform a homography are found, return `(None, None)`.
        '''

        # Set up windows.
        # Note that the windows are indexed relative to their own top left corner, not the original
        # frame's. Therefore, we must add (window_left_x, window_top_y) to the computed feature
        # coordinates to express them relative to the original frame.
        current_window = self.pixels_gray[window_top_y:window_top_y+window_height,
                                          window_left_x:window_left_x+window_width]
        next_window = next_frame.pixels_gray[window_top_y:window_top_y+window_height,
                                             window_left_x:window_left_x+window_width]

        # match all available features between frames

        # current_features is an CV_32FC2 array containing the coordinates of each keypoint;
        # see https://stackoverflow.com/a/55398871 and https://stackoverflow.com/a/47617999
        current_keypoints = feature_detector.detect(current_window)
        if len(current_keypoints) < self.HOMOGRAPHY_MIN_NUMBER_CORRESPONDING_FEATURES:
            return (None, None)

        current_features = np.float32(
            cv2.KeyPoint_convert(current_keypoints)[:, np.newaxis, :]
        )

        next_features, matched_features_mask, _ = cv2.calcOpticalFlowPyrLK(
            current_window, next_window, current_features, None
        )

        matched_features_mask = matched_features_mask.flatten().astype(dtype=bool)
        current_features = current_features[matched_features_mask]
        next_features = next_features[matched_features_mask]
        if len(current_features) < self.HOMOGRAPHY_MIN_NUMBER_CORRESPONDING_FEATURES:
            return (None, None)

        # remove outlying features with RANSAC

        _, matched_features_mask = cv2.findHomography(
            current_features, next_features, method=cv2.RANSAC
        )
        outlier_features_mask = matched_features_mask.flatten().astype(dtype=bool)
        current_features = current_features[outlier_features_mask]
        next_features = next_features[outlier_features_mask]

        # as noted above, we must add a constant offset to feature coordinates to express them
        # relative to the original window's top left corner, not the window's
        return (
            current_features + [window_left_x, window_top_y],
            next_features + [window_left_x, window_top_y]
        )

