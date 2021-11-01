import cv2
import math
import numpy as np
from scipy.ndimage import median_filter

# TODO remove when finished testing
import sys
np.set_printoptions(threshold=sys.maxsize)


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

    # In the energy function used to smooth the image, the number of frames to inspect both before
    # and after each frame when computing that frame's regularization term. As a result, the
    # regularization term involved a sum over up to 2 * TEMPORAL_SMOOTHING_RADIUS frame indexes.
    # This constnat is denoted as \Omega_t in the original paper.
    TEMPORAL_SMOOTHING_RADIUS = 10

    # the number of iterations of the Jacobi method to perform when minimizing the energy function.
    OPTIMIZATION_NUM_ITERATIONS = 100

    def __init__(self):
        self.feature_detector = cv2.FastFeatureDetector_create()


    def stabilize(self, input_path, output_path):
        '''
        Read in the video at the given input path and output a stabilized version to the given
        output path.

        Input:

        * input_path: The path to a video.
        * output_path: The path where the stabilized version of the video should be placed.

        Output:

        (The stabilized video is saved to output_path.)
        '''

        num_frames, unstabilized_frames, homographies, vertex_unstabilized_residual_displacements_by_frame_index = self._get_unstabilized_video_properties(input_path)
        vertex_stabilized_residual_displacements_by_frame_index = self._get_stabilized_residual_displacements(num_frames, vertex_unstabilized_residual_displacements_by_frame_index)


    def _get_unstabilized_video_properties(self, input_path):
        '''
        Helper method for stabilize.
        Iterate through the given unstabilized video and return properties relevant to subsequent
        stabilization steps.

        Input:

        * input_path: The path to the unstabilized video.

        Output:

        A tuple of the following properties of the unstabilized video, in order.

        * num_frames: The number of frames in the unstabilized video.
        * unstabilized_frames: A list of the frames in the unstabilized video, each represented as a
            NumPy array.
        * homographies: A NumPy array of shape
            (num_frames, self.HOMOGRAPHY_MATRIX_NUM_ROWS, self.HOMOGRAPHY_MATRIX_NUM_COLS)
            containing global homographies between frames.
            In particular, homographies[frame_index] contains the homography matrix between frames
            frame_index - 1 and frame_index (that is, the homography to construct frame_index).
            Since frame 0 has no prior frame, homographies[0] is the identity homography.
            NOTE the valid values for frame_index are 0, ..., num_frames - 1 (the first frame to the
            last frame).
        * vertex_unstabilized_residual_displacements_by_frame_index: A NumPy array of shape
            (num_frames, self.MESH_ROW_COUNT, self.MESH_COL_COUNT, 2)
            containing the unstabilized residual displacements of each vertex in the MeshFlow mesh.
            In particular,
            vertex_unstabilized_residual_displacements_by_frame_index[frame_index][row][col][0]
            contains the residual x-displacement (the x-displacement in addition to any imposed by
            global homographies) of the mesh vertex at the given row and col from frame 0 to frame
            frame_index, both inclusive.
            vertex_unstabilized_residual_displacements_by_frame_index[frame_index][row][col][1]
            contains the corresponding y-displacement.
            NOTE the valid values for frame_index are 0, ..., num_frames - 1 (the first frame to the
            last frame).
        '''

        unstabilized_video = cv2.VideoCapture(input_path)
        # for getting num_frames, see https://stackoverflow.com/a/39953739
        num_frames = int(unstabilized_video.get(cv2.CAP_PROP_FRAME_COUNT))

        homographies = np.empty(
            (num_frames, self.HOMOGRAPHY_MATRIX_NUM_ROWS,
             self.HOMOGRAPHY_MATRIX_NUM_COLS)
        )

        # vertex_unstabilized_residual_velocities_by_frame_index[frame_index][row][col]
        # contains the x- and y-
        # components of the residual velocity (velocity in addition to global velocity imposed by
        # frame-to-frame homographies) of the vertex at the given row and col during the given
        # frame_index.
        # Note that there are num_frames - 1 valid values for frame_index:
        # 0, ..., num_frames - 2 (the first frame to the second-to-last frame)
        vertex_unstabilized_residual_velocities_by_frame_index = np.empty(
            (num_frames - 1, self.MESH_ROW_COUNT + 1, self.MESH_COL_COUNT + 1, 2)
        )

        vertex_unstabilized_residual_displacements_by_frame_index = np.empty(
            (num_frames, self.MESH_ROW_COUNT + 1, self.MESH_COL_COUNT + 1, 2)
        )

        # process the first frame (which has no previous frame)
        prev_frame = self._get_next_frame(unstabilized_video)
        if prev_frame is None:
            raise IOError(
                f'Video at <{input_path}> does not contain any frames.')
        unstabilized_frames = [prev_frame]
        vertex_unstabilized_residual_displacements_by_frame_index[0].fill(0)
        homographies[0] = np.identity(3)

        # process all subsequent frames (which do have a previous frame)
        for frame_index in range(1, num_frames):
            current_frame = self._get_next_frame(unstabilized_video)
            if current_frame is None:
                raise IOError(
                    f'Video at <{input_path}> did not have frame {frame_index} of '
                    f'{num_frames} (indexed from 0).'
                )
            unstabilized_frames.append(current_frame)

            # calculcate homography between prev and current frames;
            # when calculating unstabilized velocities and performing the optimization, we assume
            # the homography has been applied already and only calculate residual velocities
            prev_features, current_features = self._get_all_matched_features_between_images(
                prev_frame, current_frame
            )
            homography, _ = cv2.findHomography(prev_features, current_features)
            homographies[frame_index] = homography

            vertex_unstabilized_residual_velocities_by_frame_index[frame_index-1] = self._get_mesh_residual_velocities(prev_frame, current_frame, homography)
            vertex_unstabilized_residual_displacements_by_frame_index[frame_index] = vertex_unstabilized_residual_displacements_by_frame_index[frame_index-1] + vertex_unstabilized_residual_velocities_by_frame_index[frame_index-1]

            prev_frame = current_frame

        unstabilized_video.release()

        return (num_frames, unstabilized_frames, homographies, vertex_unstabilized_residual_displacements_by_frame_index)


    def _get_next_frame(self, video):
        '''
        Helper method for _get_unstabilized_video_properties.

        Return the next frame of the given video.

        Input:

        * video: A VideoCapture object.

        Output:

        * next_frame: If available, the next frame in the video as a NumPy array, and None
            otherwise.
        '''

        frame_successful, pixels = video.read()
        return pixels if frame_successful else None


    def _get_mesh_residual_velocities(self, early_frame, late_frame, homography):
        '''
        Helper method for _get_unstabilized_video_properties.

        Given two adjacent frames (the "early" and "late" frames) and a homography to apply to the
        late frame, estimate the residual velocities (the remaining velocity after the homography
        has been applied) of the vertices in the early frame.

        Input:

        * early_frame: A NumPy array representing the frame before late_frame.
        * late_frame: A NumPy array representing the frame after early_frame.
        * homography: A homography matrix to apply to late_frame.

        Output:

        * mesh_residual_velocities: A NumPy array of shape
            (MESH_ROW_COUNT + 1, MESH_COL_COUNT + 1, 2)
            where the entry mesh_residual_velocities[row][col][0]
            contains the x-velocity of the mesh vertex at the given row and col during early_frame,
            and mesh_residual_velocities[row][col][1] contains the corresponding y-velocity.
            NOTE since time is discrete and in units of frames, a vertex's velocity during
            early_frame is the same as its displacement from early_frame to late_frame.
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
        Helper method for _get_mesh_residual_velocities.

        Given two adjacent frames and a homography to apply to the later frame,
        return a list that maps each vertex in the mesh to the residual velocities of its nearby
        features.

        Input:

        * early_frame: A NumPy array representing the frame before late_frame.
        * late_frame: A NumPy array representing the frame after early_frame.
        * homography: A homography matrix to apply to late_frame so that subsequent displacement
            calculations determine residual displacements (displacements in addition to the
            homography).

        Output:

        * mesh_nearby_feature_velocities: A list of tuples.
            Specifically, mesh_nearby_feature_velocities[row][col] contains a tuple
            (x_velocities, y_velocities)
            containing all the x- and y-velocities of features nearby the vertex at the given row
            and column.
        '''

        frame_height, frame_width, _ = early_frame.shape
        window_width = math.ceil(frame_width / self.OUTLIER_SUBREGIONS_COL_COUNT)
        window_height = math.ceil(frame_height / self.OUTLIER_SUBREGIONS_ROW_COUNT)

        mesh_nearby_feature_velocities = [
            [([], []) for _ in range(self.MESH_COL_COUNT + 1)]
            for _ in range(self.MESH_ROW_COUNT + 1)
        ]

        # TODO parallelize
        for window_left_x in range(0, frame_width, window_width):
            for window_top_y in range(0, frame_height, window_height):
                early_window = early_frame[window_top_y:window_top_y+window_height,
                                           window_left_x:window_left_x+window_width]
                late_window = late_frame[window_top_y:window_top_y+window_height,
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
        Helper method for _get_mesh_nearby_feature_velocities.

        Update mesh_nearby_feature_velocities so it contains the velocities of features nearby mesh
        vertices in the window, assuming the given homography has been applied.

        Input:

        * early_window: A NumPy array (or a view into one) representing a subsection of the pixels
            in the frame before late_window.
        * late_window: A NumPy array (or a view into one) representing a subsection of the pixels
            in the frame after early_window.
        * offset_location: A tuple (x, y) representing the offset of the windows within their frame,
            relative to the frame's top left corner.
        * frame_width: the width of the windows' frames.
        * frame_height: the height of the windows' frames.
        * homography: A homography matrix to apply to late_frame so that subsequent displacement
            calculations determine residual displacements (displacements in addition to the
            homography).
        * mesh_nearby_feature_velocities: A not-yet-completed list of tuples.
            Specifically, mesh_nearby_feature_velocities[row][col] will contain a tuple
            (x_velocities, y_velocities)
            containing all the x- and y-velocities of features nearby the vertex at the given row
            and column.

        Output:

        (mesh_nearby_feature_velocities has been updated to include values for all the mesh vertices
        that fall within this window.)
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
        Apply the given homography to the given positions.

        Input:

        * positions: A CV_32FC2 NumPy array (see https://stackoverflow.com/a/47617999) of positions.
        * homography: A homography matrix to apply to each position.

        Output:

        * new_positions: A copy of positions where each position has undergone the homography.
        '''

        positions_reshaped = positions.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(positions_reshaped, homography)


    def _get_feature_positions_in_window(self, early_window, late_window, window_offset):
        '''
        Helper method for _place_window_feature_velocities_into_list.

        Track and return features that appear between the two given frames, eliminating outliers
        using by applying a homography using RANSAC.

        Input:

        * early_window: A NumPy array (or a view into one) representing a subsection of the pixels
            in the frame before late_window.
        * late_window: A NumPy array (or a view into one) representing a subsection of the pixels
            in the frame after early_window.
        * offset_location: A tuple (x, y) representing the offset of the windows within their frame,
            relative to the frame's top left corner.

        Output:

        A tuple with the following values, in order.

        * early_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each non-outlying feature in early_window that was
            successfully tracked in late_window. These coordinates are expressed relative to the
            frame, not the window. If fewer than
            self.HOMOGRAPHY_MIN_NUMBER_CORRESPONDING_FEATURES such features were found,
            early_features is None.
        * late_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each non-outlying feature in late_window that was
            successfully tracked from early_window. These coordinates are expressed relative to the
            frame, not the window. If fewer than
            self.HOMOGRAPHY_MIN_NUMBER_CORRESPONDING_FEATURES such features were found,
            late_features is None.
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
        Helper method for _get_unstabilized_video_properties and _get_feature_positions_in_window.

        Detect features in the early window using the MeshFlowStabilizer's feature_detector
        and track them into the late window using cv2.calcOpticalFlowPyrLK.

        Input:

        * early_window: A NumPy array (or a view into one) representing a subsection of the pixels
            in the frame before late_window.
        * late_window: A NumPy array (or a view into one) representing a subsection of the pixels
            in the frame after early_window.

        Output:

        * early_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each feature in early_window that was
            successfully tracked in late_window. These coordinates are expressed relative to the
            window. If fewer than
            self.HOMOGRAPHY_MIN_NUMBER_CORRESPONDING_FEATURES such features were found,
            early_features is None.
        * late_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each feature in late_window that was
            successfully tracked from early_window. These coordinates are expressed relative to the
            window. If fewer than
            self.HOMOGRAPHY_MIN_NUMBER_CORRESPONDING_FEATURES such features were found,
            late_features is None.
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


    def _get_stabilized_residual_displacements(self, num_frames, vertex_unstabilized_residual_displacements_by_frame_index):
        '''
        Helper method for stabilize.

        Return each vertex's residual displacement at each frame in the stabilized video.

        Specifically, find the residual displacements that minimize an energy function.
        The energy function takes residual displacements as input and outputs a number corresponding
        to how how shaky the input is.

        The output array of stabilized residual displacements is calculated using the
        Jacobi method. For each mesh vertex, the method solves the equation
        A p = b
        for vector p,
        where entry p[i] contains the vertex's stabilized residual displacement at frame i.
        The entries in matrix A and vector b were derived by finding the partial derivative of the
        energy function with respect to each p[i] and setting them all to 0. Thus, solving for p in
        A p = b results in residual displacements that produce a local extremum (which we can safely
        assume is a local minimum) in the energy function.

        Input:

        * vertex_unstabilized_residual_displacements_by_frame_index: A NumPy array containing the
            unstabilized residual displacements of each vertex in the MeshFlow mesh, as outputted
            by _get_unstabilized_video_properties.

        Output:

        * vertex_stabilized_residual_displacements_by_frame_index: A NumPy array of shape
            (num_frames, self.MESH_ROW_COUNT, self.MESH_COL_COUNT, 2)
            containing the stabilized residual displacements of each vertex in the MeshFlow mesh.
            In particular,
            vertex_stabilized_residual_displacements_by_frame_index[frame_index][row][col][0]
            contains the residual x-displacement (the x-displacement in addition to any imposed by
            global homographies) of the mesh vertex at the given row and col from frame 0 to frame
            frame_index, both inclusive.
            vertex_unstabilized_residual_displacements_by_frame_index[frame_index][row][col][1]
            contains the corresponding y-displacement.
            NOTE the valid values for frame_index are 0, ..., num_frames - 1 (the first frame to the
            last frame).
        '''

        # row_indexes[row][col] = row, col_indexes[row][col] = col
        row_indexes, col_indexes = np.indices((num_frames, num_frames))

        # regularization_weights[t, r] is a weight constant applied to the regularization term.
        # In the paper, regularization_weights[t, r] is denoted as w_{t,r}.
        regularization_weights = np.exp(
            -np.square((3 / self.TEMPORAL_SMOOTHING_RADIUS) * (row_indexes - col_indexes))
        )

        # adaptive_weights[t] is a weight, derived from properties of the frames, applied to the
        # regularization term corresponding to the frame at index t
        # Note that the paper does not specify the weight to apply to the last frame (which does not
        # have a velocity), so we assume it is the same as the second-to-last frame.
        # In the paper, adaptive_weights[t] is denoted as \lambda_{t}.
        # TODO calculate based on homographies
        adaptive_weights = np.array([i/100 for i in range(num_frames)])

        # adaptive_weights = np.empty((num_frames,))
        # homography_affine_components = homographies.copy()
        # homography_affine_components[:, 2, :] = [0, 0, 1]
        # # eigenvalue_ratios[i] is the ratio of the two greatest eigenvalues for the matrix
        # # homography_affine_components[i];
        # # see https://stackoverflow.com/a/24395100
        # def test(x):
        #     print(x)
        #     print('has eigenvalues')
        #     eigenvalues = np.linalg.eigvals(x)
        #     print(eigenvalues)
        #     return eigenvalues
        # eigenvalue_ratios = np.array([
        #     test(matrix)
        #     for matrix in homography_affine_components
        # ])
        # # print(f'homography_affine_components (shape: {homography_affine_components.shape}):')
        # # print(homography_affine_components)
        # # eigenvalues = np.apply_along_axis(test, 1, homography_affine_components)
        # # print(f'eigenvalues (shape: {eigenvalues.shape}):')
        # # print(eigenvalue_ratios.shape)

        # off_diagonal_coefficients[t, r] is a coefficient derived from the regularization_weights and
        # adaptive_weights that appears in the partial derivatives of the energy function.
        # Using the paper's notation, off_diagonal_coefficients[t, r] is denoted as
        # \lambda_r w_{r, t} - \lambda_t w_{t, r}.
        # print(f'np.diag(adaptive_weights) (shape: {np.diag(adaptive_weights).shape}):')
        # print(np.diag(adaptive_weights))
        # print(f'regularization_weights (shape: {regularization_weights.shape}):')
        # print(regularization_weights)
        combined_adaptive_regularization_weights = np.matmul(np.diag(adaptive_weights), regularization_weights)
        # print(f'combined_adaptive_regularization_weights (shape: {combined_adaptive_regularization_weights.shape}')
        # print(combined_adaptive_regularization_weights)
        off_diagonal_coefficients = np.transpose(combined_adaptive_regularization_weights) - combined_adaptive_regularization_weights
        # print(f'off_diagonal_coefficients (shape: {off_diagonal_coefficients.shape}):')
        # print(off_diagonal_coefficients)

        # on_diagonal_coefficients is a diagonal matrix where on_diagonal_coefficients[t, t] contains a
        # coefficient that appears in partial derivatives of the energy function.
        # Using the paper's notation, on_diagonal_coefficients[t, t] is denoted as
        # 1 / \left(1 + \sum_{r \in \Omega_t, r \neq t} \left( \lambda_t w_{t, r} - \lambda_r w_{r, t} \right) \right)
        on_diagonal_coefficients = np.diag(np.reciprocal(1 - np.sum(off_diagonal_coefficients, axis=1)))
        # print(f'on_diagonal_coefficients (shape: {on_diagonal_coefficients.shape})')
        # print(on_diagonal_coefficients)

        # vertex_unstabilized_residual_displacements_by_frame_index is indexed by
        # frame_index, then row, then col, then velocity component.
        # Instead, vertex_unstabilized_residual_displacements_by_coord is indexed by
        # row, then col, then frame_index, then velocity component;
        # this rearrangement should allow for faster access during the optimization step.
        # print('vertex_unstabilized_residual_displacements_by_frame_index has shape', vertex_unstabilized_residual_displacements_by_frame_index.shape)
        vertex_unstabilized_residual_displacements_by_coord = np.moveaxis(
            vertex_unstabilized_residual_displacements_by_frame_index, 0, 2
        )
        vertex_stabilized_residual_displacements_by_coord = np.empty(vertex_unstabilized_residual_displacements_by_coord.shape)
        # print('vertex_unstabilized_residual_x_displacements_by_coord has shape', vertex_unstabilized_residual_displacements_by_coord.shape)
        # TODO parallelize
        for mesh_row in range(self.MESH_ROW_COUNT + 1):
            for mesh_col in range(self.MESH_COL_COUNT + 1):
                # print(f'vertex ({mesh_row}, {mesh_col}):')
                vertex_unstabilized_residual_displacements = vertex_unstabilized_residual_displacements_by_coord[mesh_row][mesh_col]
                # print('unstabilized:')
                print(vertex_unstabilized_residual_displacements)
                vertex_stabilized_residual_displacements = self._get_jacobi_method_output(
                    off_diagonal_coefficients, on_diagonal_coefficients,
                    vertex_unstabilized_residual_displacements,
                    vertex_unstabilized_residual_displacements
                )
                # print('stabilized:')
                # print(vertex_stabilized_residual_displacements)
                vertex_stabilized_residual_displacements_by_coord[mesh_row][mesh_col] = vertex_stabilized_residual_displacements

        vertex_stabilized_residual_displacements_by_frame_index = np.moveaxis(
            vertex_stabilized_residual_displacements_by_coord, 2, 0
        )
        return vertex_stabilized_residual_displacements_by_frame_index


    def _get_jacobi_method_output(self, off_diagonal_coefficients, on_diagonal_coefficients, x_start, b):
        '''
        Helper method for _get_stabilized_residual_displacements.
        Using the Jacobi method (see https://en.wikipedia.org/w/index.php?oldid=1036645158),
        approximate a solution for the vector x in the equation
        A x = b
        where A is a matrix of constants and b is a vector of constants.

        Return a value of x after performing self.OPTIMIZATION_NUM_ITERATIONS of the Jacobi method.

        Inputs:

        * off_diagonal_coefficients: A 2D NumPy array containing the off-diagonal entries of A.
            Specifically, off_diagonal_coefficients[i, j] = A_{i, j} where i != j, and all
            on-diagonal entries off_diagonal_coefficients[i, i] = 0.
            In the Wikipedia link, this matrix is L + U.
        * on_diagonal_coefficients: A 2D NumPy array containing the on-diagonal entries of A.
            Specifically, on_diagonal_coefficients[i, i] = A_{i, i}, and all off-diagonal entries
            of on_diagonal_coefficients are 0.
            In the Wikipedia link, this matrix is D.
        * x_start: A NumPy array containing an initial estimate for x.
        * b: A NumPy array containing the constant vector b.

        Outputs:

        * x: A NumPy array containing the value of x computed with the Jacobi method.
        '''

        x = x_start.copy()

        for _ in range(self.OPTIMIZATION_NUM_ITERATIONS):
            # print(f'\titeration {i}')
            # print('\t\off_diagonal_coefficients.shape:', off_diagonal_coefficients.shape)
            # print('\t\tx.shape:', x.shape)
            # print('\t\tproduct.shape', np.matmul(off_diagonal_coefficients, x).shape)
            # print('\t\tb.shape:', b.shape)
            # print('\t\tsum.shape:', (b + np.matmul(off_diagonal_coefficients, x)).shape)
            # print('\t\ton_diagonal_coefficients.shape:', on_diagonal_coefficients.shape)
            # print('\t\tproduct shape:', np.matmul(on_diagonal_coefficients, b + np.matmul(off_diagonal_coefficients, x)).shape)
            x = np.matmul(on_diagonal_coefficients, b + np.matmul(off_diagonal_coefficients, x))

        return x


def main():
    # TODO get video path from command line args
    input_path = 'videos/data_small-shaky-5.avi'
    output_path = 'videos/data_small-shaky-5-smoothed.avi'
    stabilizer = MeshFlowStabilizer()
    stabilizer.stabilize(input_path, output_path)


if __name__ == '__main__':
    main()
