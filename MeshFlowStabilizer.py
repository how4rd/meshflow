from collections import deque
import cv2
import math
import numpy as np
from scipy.ndimage import median_filter

from Frame import Frame

class MeshFlowStabilizer:
    '''
    A MeshFlowStabilizer stabilizes videos using the MeshFlow algorithm outlined in
    "MeshFlow: Minimum Latency Online Video Stabilization" by S. Liu et al.
    '''

    # How many rows and columns the mesh contains;
    # note that there are (MESH_COL_COUNT + 1) vertices per row, and (MESH_COL_COUNT + 1) per column.
    MESH_COL_COUNT = 16
    MESH_ROW_COUNT = 16

    # The width and height (number of columns and rows) in the mesh when breaking it down
    # into subregions to eliminate outlying regions.
    OUTLIER_SUBREGIONS_ROW_COUNT = 4
    OUTLIER_SUBREGIONS_COL_COUNT = 4

    # The width and height of the ellipse drawn around each feature to match it with mesh vertices,
    # expressed in units of mesh rows and columns.
    FEATURE_ELLIPSE_WIDTH_MESH_COLS = 3
    FEATURE_ELLIPSE_HEIGHT_MESH_ROWS = 3

    # The minimum number of corresponding features that must correspond between two frames to
    # perform a homography
    HOMOGRAPHY_MIN_NUMBER_CORRESPONDING_FEATURES = 4

    # the dimensions of a homography matrix
    HOMOGRAPHY_MATRIX_NUM_ROWS = 3
    HOMOGRAPHY_MATRIX_NUM_COLS = 3

    def __init__(self):
        self.feature_detector = cv2.FastFeatureDetector_create()


    def stabilize(self, input_path, output_path):
        '''
        Read in the video at the given input path and output a stabilized version to the given
        output path.
        '''

        unstabilized_video = cv2.VideoCapture(input_path)
        # get video properties; see https://stackoverflow.com/a/39953739
        video_width = int(unstabilized_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(unstabilized_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_num_frames = int(unstabilized_video.get(cv2.CAP_PROP_FRAME_COUNT))

        # homographies[frame_index] contains the homography matrix between
        # frames frame_index and frame_index+1;
        # note that there are video_num_frames - 1 valid values for frame_index:
        # 0, ..., video_num_frames - 2 (the first frame to the second-to-last frame)
        homographies = np.empty(
            (video_num_frames - 1, self.HOMOGRAPHY_MATRIX_NUM_ROWS, self.HOMOGRAPHY_MATRIX_NUM_COLS)
        )

        # vertex_profiles_by_frame_index[frame_index][row][col] contains the x- and y- velocity of
        # the vertex at the given row and col during the given frame_index
        # note that there are video_num_frames - 1 valid values for frame_index:
        # 0, ..., video_num_frames - 2 (the first frame to the second-to-last frame)
        vertex_profiles_by_frame_index = np.empty(
            (video_num_frames - 1, self.MESH_ROW_COUNT + 1, self.MESH_COL_COUNT + 1, 2)
        )


        # process the first frame (which has no previous frame)
        prev_frame = self._get_next_frame(unstabilized_video)
        if prev_frame is None:
            raise IOError(f'Video at <{input_path}> does not contain any frames.')
        frames = [prev_frame]

        # process all subsequent frames (which do have previous frames)
        for frame_index in range(1, video_num_frames):
            current_frame = self._get_next_frame(unstabilized_video)
            if current_frame is None:
                raise IOError(
                    f'Video at <{input_path}> did not have frame {frame_index} of '
                    f'{video_num_frames} (indexed from 0).'
                )
            frames.append(current_frame)

            # calculcate homography between prev and current frames;
            # when calculating unstabilized velocities and performing the optimization, we assume
            # the homography has been applied already and only calculate residual velocities
            prev_features, current_features = self._get_all_matched_features_between_images(
                prev_frame.pixels_bgr, current_frame.pixels_bgr
            )
            homography, _ = cv2.findHomography(prev_features, current_features)
            homographies[frame_index - 1] = homography

            prev_frame.mesh_residual_velocities = self._get_mesh_residual_velocities(prev_frame, current_frame, homography)
            vertex_profiles_by_frame_index[frame_index - 1] = prev_frame.mesh_residual_velocities

            prev_frame = current_frame

        unstabilized_video.release()


    def _get_next_frame(self, video):
        '''
        Given a VideoCapture containing footage to stabilize, read in the next Frame if available.
        Return the Frame if available and None otherwise.
        '''

        frame_successful, pixels_bgr = video.read()
        if not frame_successful:  # all the video's frames had already been read
            return None

        return Frame(pixels_bgr)


    def _get_mesh_residual_velocities(self, early_frame, late_frame, homography):
        '''
        Given two adjacent Frames (the "early" and "late" Frames) and a homography to apply to the
        late Frame,
        estimate the residual velocities (the remaining velocity after the homography has been
        applied) of the vertices in the early Frame.
        Return the result as a (MESH_COL_COUNT + 1) by (MESH_ROW_COUNT + 1) by 2 array containing
        the x- and y-velocity of each vertex in the early Frame relative to the late Frame.
        '''

        mesh_nearby_feature_velocities = self._get_mesh_nearby_feature_velocities(early_frame, late_frame, homography)

        # Perform first median filter:
        # sort each vertex's velocities by x-component, then by y-component, and use the median
        # element as the vertex's velocity.
        mesh_residual_velocities_unsmoothed = np.array([
            [
                (
                    sorted(x_velocities)[len(x_velocities)//2],
                    sorted(y_velocities)[len(y_velocities)//2]
                )
                if x_velocities else (0, 0)
                for x_velocities, y_velocities in row
            ]
            for row in mesh_nearby_feature_velocities
        ])

        # Perform second median filter:
        # replace each vertex's velocity with the median velocity of its neighbors.
        # Note that the OpenCV implementation cannot not handle 2-channel images (like
        # mesh_residual_velocities_unsmoothed, which has channels for x- and y-velocities),
        # which is why we use the SciPy implementation instead.
        return median_filter(mesh_residual_velocities_unsmoothed, size=3)


    def _get_mesh_nearby_feature_velocities(self, early_frame, late_frame, homography):
        '''
        Helper function for _get_mesh_residual_velocities.
        Given two adjacent Frames (the "early" and "late" Frames) and a homography to apply to the
        late Frame,
        return a list that maps each vertex in the mesh to the residual velocities of its nearby
        features.
        Specifically, the output of this function is a list mesh_nearby_feature_velocities where
        mesh_nearby_feature_velocities[row][col] contains a tuple (x_velocities, y_velocities)
        containing all the x- and y-velocities of features nearby the vertex at the given row and
        column.
        '''

        frame_height, frame_width, _ = early_frame.pixels_bgr.shape
        window_width = math.ceil(frame_width / self.OUTLIER_SUBREGIONS_COL_COUNT)
        window_height = math.ceil(frame_height / self.OUTLIER_SUBREGIONS_ROW_COUNT)

        mesh_nearby_feature_velocities = [
            [([], []) for _ in range(self.MESH_COL_COUNT + 1)]
            for _ in range(self.MESH_ROW_COUNT + 1)
        ]

        # TODO parallelize
        for window_left_x in range(0, frame_width, window_width):
            for window_top_y in range(0, frame_height, window_height):
                early_window = early_frame.pixels_bgr[window_top_y:window_top_y+window_height,
                                                      window_left_x:window_left_x+window_width]
                late_window = late_frame.pixels_bgr[window_top_y:window_top_y+window_height,
                                                    window_left_x:window_left_x+window_width]
                window_offset = [window_left_x, window_top_y]

                self._place_window_feature_velocities_into_list(
                    early_window, late_window, window_offset, frame_width, frame_height,
                    homography,
                    mesh_nearby_feature_velocities
                )

        return mesh_nearby_feature_velocities


    def _place_window_feature_velocities_into_list(self, early_window, late_window, window_offset, frame_width, frame_height, homography, mesh_nearby_feature_velocities):
        '''
        Helper function for _get_mesh_nearby_feature_velocities.
        Given windows into two adjacent Frames (subsections of the "early" and "late" Frames'
        pixels), the offset location (position of top left corner) of those windows within their
        original Frames, the Frame's dimensions, a homography to apply to the late Frame, and an
        incomplete list mesh_nearby_feature_velocities of the sort outputted by
        _get_mesh_nearby_feature_velocities,
        update mesh_nearby_feature_velocities so it contains the velocities of features nearby mesh
        vertices in the window, assuming the given homography has been applied to the
        '''

        # gather features
        early_window_feature_positions, late_window_feature_positions_no_homography = self._get_feature_positions_in_window(
            early_window, late_window, window_offset
        )

        if early_window_feature_positions is None:
            return

        late_window_feature_positions = self._get_positions_with_homography_applied(
            late_window_feature_positions_no_homography, homography
        )

        # calculate features' velocities; see https://stackoverflow.com/a/44409124 for
        # combining the positions and velocities into one matrix
        current_window_velocities = late_window_feature_positions - early_window_feature_positions
        current_window_positions_velocities = np.c_[early_window_feature_positions, current_window_velocities]

        # apply features' velocities to nearby mesh vertices
        for feature_position_and_velocity in current_window_positions_velocities:
            feature_x, feature_y, feature_x_velocity, feature_y_velocity = feature_position_and_velocity[0]
            feature_row = (feature_y / frame_height) * self.MESH_ROW_COUNT
            feature_col = (feature_x / frame_width) * self.MESH_COL_COUNT

            # Draw an ellipse around each feature
            # of width self.FEATURE_ELLIPSE_WIDTH_MESH_COLS
            # and height self.FEATURE_ELLIPSE_HEIGHT_MESH_ROWS,
            # and apply the feature's velocity to all mesh vertices that fall within this
            # ellipse.
            # To do this, we can iterate through all the rows that the ellipse covers.
            # For each row, we can use the equation for an ellipse centered on the
            # feature to determine which columns the ellipse covers. The resulting
            # (row, column) pairs correspond to the vertices in the ellipse.
            ellipse_top_row_inclusive = max(0, math.ceil(feature_row - self.FEATURE_ELLIPSE_HEIGHT_MESH_ROWS / 2))
            ellipse_bottom_row_exclusive = 1 + min(self.MESH_ROW_COUNT, math.floor(feature_row + self.FEATURE_ELLIPSE_HEIGHT_MESH_ROWS / 2))

            for vertex_row in range(ellipse_top_row_inclusive, ellipse_bottom_row_exclusive):
                # half-width derived from ellipse equation
                ellipse_slice_half_width = self.FEATURE_ELLIPSE_WIDTH_MESH_COLS * math.sqrt((1/4) - ((vertex_row - feature_row) / self.FEATURE_ELLIPSE_HEIGHT_MESH_ROWS) ** 2)
                ellipse_left_col_inclusive = max(0, math.ceil(feature_col - ellipse_slice_half_width))
                ellipse_right_col_exclusive = 1 + min(self.MESH_COL_COUNT, math.floor(feature_col + ellipse_slice_half_width))

                for vertex_col in range(ellipse_left_col_inclusive, ellipse_right_col_exclusive):
                    mesh_nearby_feature_velocities[vertex_row][vertex_col][0].append(feature_x_velocity)
                    mesh_nearby_feature_velocities[vertex_row][vertex_col][1].append(feature_y_velocity)


    def _get_positions_with_homography_applied(self, positions, homography):
        '''
        Given a CV_32FC2 array of positions and a homography matrix,
        return a copy of the positions where each has had the homography applied.
        '''


        positions_reshaped = positions.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(positions_reshaped, homography)


    def _get_feature_positions_in_window(self, early_window, late_window, window_offset):
        '''
        Helper function for _place_window_feature_velocities_into_list.
        Given windows into two adjacent frames (subsections of the "early" and "late" Frames'
        pixels) and the offset (position of top left corner) of those windows within their original
        Frames, return a tuple
        (early_features, late_features),
        where early_features is an CV_32FC2 array (see https://stackoverflow.com/a/47617999)
        containing the coordinates of each non-outlying feature in the early Frame that was
        successfully tracked into the late Frame, and late_features contains those features'
        coordinates in the late Frame.

        If not enough features to perform a homography are found, return (None, None).
        '''

        # gather all features that track between frames
        early_features_including_outliers, late_features_including_outliers = self._get_all_matched_features_between_images(early_window, late_window)
        if early_features_including_outliers is None:
            return (None, None)


        # eliminate outlying features using RANSAC
        _, outlier_features = cv2.findHomography(
            early_features_including_outliers, late_features_including_outliers, method=cv2.RANSAC
        )
        outlier_features_mask = outlier_features.flatten().astype(dtype=bool)
        early_features = early_features_including_outliers[outlier_features_mask]
        late_features = late_features_including_outliers[outlier_features_mask]

        # Add a constant offset to feature coordinates to express them
        # relative to the original window's top left corner, not the window's
        return (early_features + window_offset, late_features + window_offset)


    def _get_all_matched_features_between_images(self, early_window, late_window):
        '''
        Helper function.
        Given windows into two adjacent Frames (subsections of the "early" and "late" Frames'
        pixels), detect features in the early window using the MeshFlowStabilizer's feature_detector
        and track them into the late window using cv2.calcOpticalFlowPyrLK.
        Return a tuple
        (early_features, late_features)
        with the positions of the early and late Frames' features relative to the window as a
        CV_32FC2 array, or
        (None, None)
        if fewer than self.HOMOGRAPHY_MIN_NUMBER_CORRESPONDING_FEATURES such features are found.
        '''

        # convert a KeyPoint list into a CV_32FC2 array containing the coordinates of each KeyPoint;
        # see https://stackoverflow.com/a/55398871 and https://stackoverflow.com/a/47617999
        early_keypoints = self.feature_detector.detect(early_window)
        if len(early_keypoints) < self.HOMOGRAPHY_MIN_NUMBER_CORRESPONDING_FEATURES:
            return (None, None)

        early_features_including_unmatched = np.float32(cv2.KeyPoint_convert(early_keypoints)[:, np.newaxis, :])
        late_features_including_unmatched, matched_features, _ = cv2.calcOpticalFlowPyrLK(
            early_window, late_window, early_features_including_unmatched, None
        )

        matched_features_mask = matched_features.flatten().astype(dtype=bool)
        early_features = early_features_including_unmatched[matched_features_mask]
        late_features = late_features_including_unmatched[matched_features_mask]

        if len(early_features) < self.HOMOGRAPHY_MIN_NUMBER_CORRESPONDING_FEATURES:
            return (None, None)

        return (early_features, late_features)


def main():
    # TODO get video path from command line args
    input_path = 'videos/data_small-shaky-5.avi'
    output_path = 'videos/data_small-shaky-5-smoothed.avi'
    stabilizer = MeshFlowStabilizer()
    stabilizer.stabilize(input_path, output_path)


if __name__ == '__main__':
    main()
