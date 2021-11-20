import cv2
import math
import numpy as np
import statistics
import tqdm

class MeshFlowStabilizer:
    '''
    A MeshFlowStabilizer stabilizes videos using the MeshFlow algorithm outlined in
    "MeshFlow: Minimum Latency Online Video Stabilization" by S. Liu et al.
    '''


    '''
    Enum indicating which formula to use to compute the Jacobi method optimization.

    The values are:
    * OPTIMIZATION_FORMULA_ORIGINAL: Use the formula that the authors seem to have to used in the
        original paper. This formula is not explicitly described in the paper, but appears in
        "Bundled Camera Paths for Video Stabilization" by S. Liu et al. without its derivation.
    * OPTIMIZATION_FORMULA_DERIVED: Use the formula derived by setting the derivative of the energy
        function to 0 and solving for the variable.
    '''

    OPTIMIZATION_FORMULA_ORIGINAL = 0
    OPTIMIZATION_FORMULA_DERIVED = 1


    '''
    Enum indicating which definition to use for the energy function's adaptive weights.

    The values are:
    * ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL: Calculate the adaptive weights using the model suggested in the
        original paper.
    * ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED: Calculate the adaptive weights using a variant of the original model
        in which one of the terms has had its sign flipped. Suggested on GitHub TODO cite URL.
    * ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH: Set the adaptive weights using a constant high value.
        This model appears in the implementation on GitHub; see
        https://github.com/sudheerachary/Mesh-Flow-Video-Stabilization/issues/12#issuecomment-553737073.
    * ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW: Set the adaptive weights to a constant low value.
        This model is based on the authors' claim that smaller adaptive weights lead to less
        cropping and wobbling.
    '''

    ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL = 0
    ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED = 1
    ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH = 2
    ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW = 3


    # The adaptive weights' constant high and low values.
    ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH_VALUE = 100
    ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW_VALUE = 1


    def __init__(self, mesh_row_count=16, mesh_col_count=16,
        mesh_outlier_subregion_row_count=4, mesh_outlier_subregion_col_count=4,
        feature_ellipse_row_count=10, feature_ellipse_col_count=10,
        homography_min_number_corresponding_features=4,
        temporal_smoothing_radius=10, optimization_num_iterations=100,
        color_outside_image_area_bgr=(0, 0, 255)):
        '''
        Constructor.

        Input:

        * mesh_row_count: The number of rows contained in the mesh.
            NOTE There are 1 + mesh_row_count vertices per row.
        * mesh_col_count: The number of cols contained in the mesh.
            NOTE There are 1 + mesh_col_count vertices per column.
        * mesh_outlier_subregion_row_count: The height in rows of each subregion when breaking down
            the image down into subregions to eliminate outlying features.
        * mesh_outlier_subregion_col_count: The width of columns of each subregion when breaking
            down the image down into subregions to eliminate outlying features.
        * feature_ellipse_row_count: The height in rows of the ellipse drawn around each feature
            to match it with vertices in the mesh.
        * feature_ellipse_col_count: The width in columns of the ellipse drawn around each feature
            to match it with vertices in the mesh.
        * homography_min_number_corresponding_features: The minimum number of corresponding features
            that must correspond between two frames to perform a homography.
        * temporal_smoothing_radius: In the energy function used to smooth the image, the number of
            frames to inspect both before and after each frame when computing that frame's
            regularization term. As a result, the regularization term involves a sum over up to
            2 * temporal_smoothing_radius frame indexes.
            NOTE This constant is denoted as \Omega_{t} in the original paper.
        * optimization_num_iterations: The number of iterations of the Jacobi method to perform when
            minimizing the energy function.
        * color_outside_image_area_bgr: The color, expressed in BGR, to display behind the
            stabilized footage in the output.
            NOTE This color should be removed during cropping, but is customizable just in case.

        Output:

        (A MeshFlowStabilizer object.)
        '''

        self.mesh_col_count = mesh_col_count
        self.mesh_row_count = mesh_row_count
        self.mesh_outlier_subregion_row_count = mesh_outlier_subregion_row_count
        self.mesh_outlier_subregion_col_count = mesh_outlier_subregion_col_count
        self.feature_ellipse_row_count = feature_ellipse_row_count
        self.feature_ellipse_col_count = feature_ellipse_col_count
        self.homography_min_number_corresponding_features = homography_min_number_corresponding_features
        self.temporal_smoothing_radius = temporal_smoothing_radius
        self.optimization_num_iterations = optimization_num_iterations
        self.color_outside_image_area_bgr = color_outside_image_area_bgr

        self.feature_detector = cv2.FastFeatureDetector_create()


    def stabilize(self, input_path, output_path, optimization_formula=OPTIMIZATION_FORMULA_ORIGINAL, adaptive_weights_definition=ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL):
        '''
        Read in the video at the given input path and output a stabilized version to the given
        output path.

        Input:

        * input_path: The path to a video.
        * output_path: The path where the stabilized version of the video should be placed.
        * optimization_formula: Which method to use to compute the Jacobi method optimization.
        * adaptive_weights_definition: Which method to use for computing the energy function's adaptive
            weights.

        Output:

        (The stabilized video is saved to output_path.)

        In addition, the function returns a tuple of the following items in order.

        * cropping_ratio: The cropping ratio of the stabilized video. Per the original paper, the
            cropping ratio of each frame is the scale component of its unstabilized-to-cropped
            homography, and the cropping ratio of the overall video is the average of the frames'
            cropping ratios.
        * distortion_score: The distortion score of the stabilized video. Per the original paper,
            the distortion score of each frame is ratio of the two largest eigenvalues of the
            affine part of its unstabilized-to-cropped homography, and the distortion score of the
            overall video is the greatest of its frames' distortion scores.
        * stability_score: The stability score of the stabilized video. Per the original paper, the
            stability score for each vertex is derived from the representation of its vertex profile
            (vector of velocities) in the frequency domain. Specifically, it is the fraction of the
            representation's total energy that is contained within its second to sixth lowest
            frequencies. The stability score of the overall video is the average of the vertices'
            stability scores.
        '''

        if not (optimization_formula == MeshFlowStabilizer.OPTIMIZATION_FORMULA_ORIGINAL or
                optimization_formula == MeshFlowStabilizer.OPTIMIZATION_FORMULA_DERIVED):
            raise ValueError(
                'Invalid value for `optimization_formula`. Expecting value of '
                '`MeshFlowStabilizer.OPTIMIZATION_FORMULA_ORIGINAL` or '
                '`MeshFlowStabilizer.OPTIMIZATION_FORMULA_DERIVED`.'
            )

        if not (adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL or
                adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED or
                adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH or
                adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW):
            raise ValueError(
                'Invalid value for `adaptive_weights_definition`. Expecting value of '
                '`MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL`, '
                '`MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED`, '
                '`MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH`, or'
                '`MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW`.'
            )

        unstabilized_frames, num_frames, frames_per_second, codec = self._get_unstabilized_frames_and_video_features(input_path)
        vertex_unstabilized_displacements_by_frame_index, homographies = self._get_unstabilized_vertex_displacements_and_homographies(num_frames, unstabilized_frames)
        vertex_stabilized_displacements_by_frame_index = self._get_stabilized_vertex_displacements(
            num_frames, optimization_formula, adaptive_weights_definition,
            vertex_unstabilized_displacements_by_frame_index, homographies
        )
        stabilized_frames, crop_boundaries = self._get_stabilized_frames_and_crop_boundaries(
            num_frames, unstabilized_frames,
            vertex_unstabilized_displacements_by_frame_index,
            vertex_stabilized_displacements_by_frame_index
        )
        cropped_frames = self._crop_frames(stabilized_frames, crop_boundaries)

        cropping_ratio, distortion_score = self._compute_cropping_ratio_and_distortion_score(num_frames, unstabilized_frames, cropped_frames)
        stability_score = self._compute_stability_score(num_frames, vertex_stabilized_displacements_by_frame_index)

        self._write_stabilized_video(output_path, num_frames, frames_per_second, codec, cropped_frames)
        self._display_unstablilized_and_stabilized_video_loop(num_frames, frames_per_second, unstabilized_frames, cropped_frames)

        return (cropping_ratio, distortion_score, stability_score)


    def _get_unstabilized_frames_and_video_features(self, input_path):
        '''
        Helper method for stabilize.
        Return each frame of the input video as a NumPy array along with miscellaneous video
        features.

        Input:

        * input_path: The path to the unstabilized video.

        Output:

        A tuple of the following items in order.

        * unstabilized_frames: A list of the frames in the unstabilized video, each represented as a
            NumPy array.
        * num_frames: The number of frames in the video.
        * frames_per_second: The video framerate in frames per second.
        * codec: The video codec.
        '''

        unstabilized_video = cv2.VideoCapture(input_path)
        # for getting num_frames, see https://stackoverflow.com/a/39953739
        num_frames = int(unstabilized_video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_per_second = unstabilized_video.get(cv2.CAP_PROP_FPS)
        codec = int(unstabilized_video.get(cv2.CAP_PROP_FOURCC))

        with tqdm.trange(num_frames) as t:
            t.set_description(f'Reading video from <{input_path}>')

            unstabilized_frames = []
            for frame_index in t:
                unstabilized_frame = self._get_next_frame(unstabilized_video)
                if unstabilized_frame is None:
                    raise IOError(
                        f'Video at <{input_path}> did not have frame {frame_index} of '
                        f'{num_frames} (indexed from 0).'
                    )
                unstabilized_frames.append(unstabilized_frame)

        unstabilized_video.release()

        return (unstabilized_frames, num_frames, frames_per_second, codec)


    def _get_next_frame(self, video):
        '''
        Helper method for _get_unstabilized_frames_and_video_features.

        Return the next frame of the given video.

        Input:

        * video: A VideoCapture object.

        Output:

        * next_frame: If available, the next frame in the video as a NumPy array, and None
            otherwise.
        '''

        frame_successful, pixels = video.read()
        return pixels if frame_successful else None


    def _get_unstabilized_vertex_displacements_and_homographies(self, num_frames, unstabilized_frames):
        '''
        Helper method for stabilize.
        Return the displacements for the given unstabilized frames.

        Input:

        * num_frames: The number of frames in the video.
        * unstabilized_frames: A list of the unstabilized frames, each represented as a NumPy array.

        Output:

        A tuple of the following items in order.

        * vertex_unstabilized_displacements_by_frame_index: A NumPy array of shape
            (num_frames, self.mesh_row_count, self.mesh_col_count, 2)
            containing the unstabilized displacements of each vertex in the MeshFlow mesh.
            In particular,
            vertex_unstabilized_displacements_by_frame_index[frame_index][row][col][0]
            contains the x-displacement of the mesh vertex at the given row and col from frame 0 to
            frame frame_index, both inclusive.
            vertex_unstabilized_displacements_by_frame_index[frame_index][row][col][1]
            contains the corresponding y-displacement.
        * homographies: A NumPy array of shape
            (num_frames, 3, 3)
            containing global homographies between frames.
            In particular, homographies[frame_index] contains a homography matrix between frames
            frame_index and frame_index + 1 (that is, the homography to construct frame_index + 1).
            Since no frame comes after num_frames - 1,
            homographies[num_frames-1] is the identity homography.
        '''

        vertex_unstabilized_displacements_by_frame_index = np.empty(
            (num_frames, self.mesh_row_count + 1, self.mesh_col_count + 1, 2)
        )
        vertex_unstabilized_displacements_by_frame_index[0].fill(0)

        homographies = np.empty((num_frames, 3, 3))
        homographies[-1] = np.identity(3)

        with tqdm.trange(num_frames - 1) as t:
            t.set_description('Computing unstabilized mesh displacements')
            for current_index in t:
                current_frame, next_frame = unstabilized_frames[current_index:current_index+2]
                current_velocity, homography = self._get_unstabilized_vertex_velocities(current_frame, next_frame)
                vertex_unstabilized_displacements_by_frame_index[current_index+1] = vertex_unstabilized_displacements_by_frame_index[current_index] + current_velocity
                homographies[current_index] = homography

        return (vertex_unstabilized_displacements_by_frame_index, homographies)


    def _get_unstabilized_vertex_velocities(self, early_frame, late_frame):
        '''
        Helper method for _get_unstabilized_vertex_displacements_and_homographies.

        Given two adjacent frames (the "early" and "late" frames), estimate the velocities of the
        vertices in the early frame.

        Input:

        * early_frame: A NumPy array representing the frame before late_frame.
        * late_frame: A NumPy array representing the frame after early_frame.

        Output:

        A tuple of the following items in order.

        * mesh_velocities: A NumPy array of shape
            (mesh_row_count + 1, mesh_col_count + 1, 2)
            where the entry mesh_velocities[row][col][0]
            contains the x-velocity of the mesh vertex at the given row and col during early_frame,
            and mesh_velocities[row][col][1] contains the corresponding y-velocity.
            NOTE since time is discrete and in units of frames, a vertex's velocity during
            early_frame is the same as its displacement from early_frame to late_frame.
        * early_to_late_homography: A NumPy array of shape (3, 3) representing the homography
            between early_frame and late_frame.
        '''

        # applying this homography to a coordinate in the early frame maps it to where it will be
        # in the late frame, assuming the point is not undergoing motion
        early_features, late_features = self._get_all_matched_features_between_images(
            early_frame, late_frame
        )
        early_to_late_homography, _ = cv2.findHomography(early_features, late_features)

        # Each vertex started in the early frame at a position given by vertex_x_y_by_row_coland.
        # If it has no velocity relative to the scene (i.e., the vertex is shaking with it), then to
        # get its position in the late frame, we apply early_to_late_homography to its early
        # position.
        # Its velocity takes it from the early position to its late position.
        frame_height, frame_width, = early_frame.shape[:2]
        vertex_x_y = self._get_vertex_x_y(frame_width, frame_height)
        vertex_velocities = cv2.perspectiveTransform(vertex_x_y, early_to_late_homography) - vertex_x_y
        vertex_velocities_by_row_col = np.reshape(vertex_velocities, (self.mesh_row_count + 1, self.mesh_col_count + 1, 2))
        vertex_x_velocities_by_row_col = vertex_velocities_by_row_col[:, :, 0]
        vertex_y_velocities_by_row_col = vertex_velocities_by_row_col[:, :, 1]

        # In addition to the above motion (which moves each vertex to its spot in the mesh in
        # late_frame), each vertex may undergo additional motion to match its nearby features.
        # After gathering these velocities, perform first median filter:
        # sort each vertex's velocities by x-component, then by y-component, and use the median
        # element as the vertex's velocity.
        vertex_nearby_feature_x_velocities_by_row_col, vertex_nearby_feature_y_velocities_by_row_col = self._get_unstabilized_vertex_nearby_feature_velocities(early_frame, late_frame, early_to_late_homography)

        mesh_median_nearby_feature_x_velocity_by_row_col = np.array([
            [
                statistics.median(x_velocities)
                if x_velocities else 0
                for x_velocities in row
            ]
            for row in vertex_nearby_feature_x_velocities_by_row_col
        ])
        mesh_median_nearby_feature_y_velocity_by_row_col = np.array([
            [
                statistics.median(y_velocities)
                if y_velocities else 0
                for y_velocities in row
            ]
            for row in vertex_nearby_feature_y_velocities_by_row_col
        ])
        vertex_x_velocities_by_row_col += mesh_median_nearby_feature_x_velocity_by_row_col
        vertex_y_velocities_by_row_col += mesh_median_nearby_feature_y_velocity_by_row_col

        # Perform second median filter:
        # replace each vertex's velocity with the median velocity of its neighbors.
        vertex_smoothed_x_velocities_by_row_col = cv2.medianBlur(vertex_x_velocities_by_row_col, 3)
        vertex_smoothed_y_velocities_by_row_col = cv2.medianBlur(vertex_y_velocities_by_row_col, 3)
        vertex_smoothed_velocities_by_row_col = np.dstack((vertex_smoothed_x_velocities_by_row_col, vertex_smoothed_y_velocities_by_row_col))
        return (vertex_smoothed_velocities_by_row_col, early_to_late_homography)


    def _get_unstabilized_vertex_nearby_feature_velocities(self, early_frame, late_frame, early_to_late_homography):
        '''
        Helper method for _get_unstabilized_vertex_velocities.

        Given two adjacent frames, return a list that maps each vertex in the mesh to the residual
        velocities of its nearby features.

        Input:

        * early_frame: A NumPy array representing the frame before late_frame.
        * late_frame: A NumPy array representing the frame after early_frame.
        * early_to_late_homography: A homography matrix that maps a point in early_frame to its
            corresponding location in late_frame, assuming the point is not undergoing motion

        Output:

        A tuple of the following items in order.

        * vertex_nearby_feature_x_velocities_by_row_col: A list
            where entry vertex_nearby_feature_x_velocities_by_row_col[row, col] contains a list of
            the x-velocities of all the features nearby the vertex at the given row and col.
        * vertex_nearby_feature_y_velocities_by_row_col: A list
            where entry vertex_nearby_feature_y_velocities_by_row_col[row, col] contains a list of
            the x-velocities of all the features nearby the vertex at the given row and col.
        '''

        frame_height, frame_width = early_frame.shape[:2]
        window_width = math.ceil(frame_width / self.mesh_outlier_subregion_col_count)
        window_height = math.ceil(frame_height / self.mesh_outlier_subregion_row_count)

        vertex_nearby_feature_x_velocities_by_row_col = [
            [[] for _ in range(self.mesh_col_count + 1)]
            for _ in range(self.mesh_row_count + 1)
        ]
        vertex_nearby_feature_y_velocities_by_row_col = [
            [[] for _ in range(self.mesh_col_count + 1)]
            for _ in range(self.mesh_row_count + 1)
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
                    early_window, late_window, early_to_late_homography,
                    window_offset, frame_width, frame_height,
                    vertex_nearby_feature_x_velocities_by_row_col,
                    vertex_nearby_feature_y_velocities_by_row_col
                )

        return (vertex_nearby_feature_x_velocities_by_row_col, vertex_nearby_feature_y_velocities_by_row_col)


    def _place_window_feature_velocities_into_list(self, early_window, late_window, early_to_late_homography, window_offset, frame_width, frame_height, vertex_nearby_feature_x_velocities_by_row_col, vertex_nearby_feature_y_velocities_by_row_col):
        '''
        Helper method for _get_unstabilized_vertex_nearby_feature_velocities.

        Update mesh_nearby_feature_velocities so it contains the velocities of features nearby mesh
        vertices in the window, assuming the given homography has been applied.

        Input:

        * early_window: A NumPy array (or a view into one) representing a subsection of the pixels
            in the frame before late_window.
        * late_window: A NumPy array (or a view into one) representing a subsection of the pixels
            in the frame after early_window.
        * early_to_late_homography: A homography matrix that maps a point in early_frame to its
            corresponding location in late_frame, assuming the point is not undergoing motion
        * offset_location: A tuple (x, y) representing the offset of the windows within their frame,
            relative to the frame's top left corner.
        * frame_width: the width of the windows' frames.
        * frame_height: the height of the windows' frames.
        * vertex_nearby_feature_x_velocities_by_row_col: A not-yet-completed list
            where entry vertex_nearby_feature_x_velocities_by_row_col[row, col] contains a list of
            the x-velocities of all the features nearby the vertex at the given row and col.
        * vertex_nearby_feature_y_velocities_by_row_col: A not-yet-completed list
            where entry vertex_nearby_feature_y_velocities_by_row_col[row, col] contains a list of
            the x-velocities of all the features nearby the vertex at the given row and col.

        Output:

        (Both vertex_nearby_feature_x_velocities_by_row_col and
        vertex_nearby_feature_y_velocities_by_row_col
        have been updated to include values for all the mesh vertices that fall within this window.)
        '''

        # gather features
        early_window_feature_positions, late_window_feature_positions = self._get_feature_positions_in_window(
            early_window, late_window, window_offset
        )

        if early_window_feature_positions is None:
            return

        # calculate features' velocities; see https://stackoverflow.com/a/44409124 for
        # combining the positions and velocities into one matrix

        # If a point were undergoing no motion, then its position in the late frame would be found
        # by applying early_to_late_homography to its position in the early frame.
        # The point's additional motion is what takes it from that position to its actual position.
        current_window_velocities = late_window_feature_positions - cv2.perspectiveTransform(early_window_feature_positions, early_to_late_homography)
        current_window_positions_velocities = np.c_[early_window_feature_positions, current_window_velocities]

        # apply features' velocities to nearby mesh vertices
        for feature_position_and_velocity in current_window_positions_velocities:
            feature_x, feature_y, feature_x_velocity, feature_y_velocity = feature_position_and_velocity[0]
            feature_row = (feature_y / frame_height) * self.mesh_row_count
            feature_col = (feature_x / frame_width) * self.mesh_col_count

            # Draw an ellipse around each feature
            # of width self.feature_ellipse_col_count
            # and height self.feature_ellipse_row_count,
            # and apply the feature's velocity to all mesh vertices that fall within this
            # ellipse.
            # To do this, we can iterate through all the rows that the ellipse covers.
            # For each row, we can use the equation for an ellipse centered on the
            # feature to determine which columns the ellipse covers. The resulting
            # (row, column) pairs correspond to the vertices in the ellipse.
            ellipse_top_row_inclusive = max(0, math.ceil(feature_row - self.feature_ellipse_row_count / 2))
            ellipse_bottom_row_exclusive = 1 + min(self.mesh_row_count, math.floor(feature_row + self.feature_ellipse_row_count / 2))

            for vertex_row in range(ellipse_top_row_inclusive, ellipse_bottom_row_exclusive):

                # half-width derived from ellipse equation
                ellipse_slice_half_width = self.feature_ellipse_col_count * math.sqrt((1/4) - ((vertex_row - feature_row) / self.feature_ellipse_row_count) ** 2)
                ellipse_left_col_inclusive = max(0, math.ceil(feature_col - ellipse_slice_half_width))
                ellipse_right_col_exclusive = 1 + min(self.mesh_col_count, math.floor(feature_col + ellipse_slice_half_width))

                for vertex_col in range(ellipse_left_col_inclusive, ellipse_right_col_exclusive):
                    vertex_nearby_feature_x_velocities_by_row_col[vertex_row][vertex_col].append(feature_x_velocity)
                    vertex_nearby_feature_y_velocities_by_row_col[vertex_row][vertex_col].append(feature_y_velocity)


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

        A tuple of the following items in order.

        * early_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each non-outlying feature in early_window that was
            successfully tracked in late_window. These coordinates are expressed relative to the
            frame, not the window. If fewer than
            self.homography_min_number_corresponding_features such features were found,
            early_features is None.
        * late_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each non-outlying feature in late_window that was
            successfully tracked from early_window. These coordinates are expressed relative to the
            frame, not the window. If fewer than
            self.homography_min_number_corresponding_features such features were found,
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
        Helper method for _get_unstabilized_frames_and_video_features and _get_feature_positions_in_window.

        Detect features in the early window using the MeshFlowStabilizer's feature_detector
        and track them into the late window using cv2.calcOpticalFlowPyrLK.

        Input:

        * early_window: A NumPy array (or a view into one) representing a subsection of the pixels
            in the frame before late_window.
        * late_window: A NumPy array (or a view into one) representing a subsection of the pixels
            in the frame after early_window.

        Output:

        A tuple of the following items in order.

        * early_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each feature in early_window that was
            successfully tracked in late_window. These coordinates are expressed relative to the
            window. If fewer than
            self.homography_min_number_corresponding_features such features were found,
            early_features is None.
        * late_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each feature in late_window that was
            successfully tracked from early_window. These coordinates are expressed relative to the
            window. If fewer than
            self.homography_min_number_corresponding_features such features were found,
            late_features is None.
        '''

        # convert a KeyPoint list into a CV_32FC2 array containing the coordinates of each KeyPoint;
        # see https://stackoverflow.com/a/55398871 and https://stackoverflow.com/a/47617999
        early_keypoints = self.feature_detector.detect(early_window)
        if len(early_keypoints) < self.homography_min_number_corresponding_features:
            return (None, None)

        early_features_including_unmatched = np.float32(cv2.KeyPoint_convert(early_keypoints)[:, np.newaxis, :])
        late_features_including_unmatched, matched_features, _ = cv2.calcOpticalFlowPyrLK(
            early_window, late_window, early_features_including_unmatched, None
        )

        matched_features_mask = matched_features.flatten().astype(dtype=bool)
        early_features = early_features_including_unmatched[matched_features_mask]
        late_features = late_features_including_unmatched[matched_features_mask]

        if len(early_features) < self.homography_min_number_corresponding_features:
            return (None, None)

        return (early_features, late_features)


    def _get_stabilized_vertex_displacements(self, num_frames, optimization_formula, adaptive_weights_definition, vertex_unstabilized_displacements_by_frame_index, homographies):
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

        * num_frames: The number of frames in the video.
        * optimization_formula: Which method to use to compute the Jacobi method optimization.
        * adaptive_weights_definition: Which definition to use for the energy function's adaptive
            weights.
        * vertex_unstabilized_displacements_by_frame_index: A NumPy array containing the
            unstabilized residual displacements of each vertex in the MeshFlow mesh, as outputted
            by _get_unstabilized_frames_and_video_features.
        * homographies: A NumPy array of homographies as generated by
            _get_unstabilized_vertex_displacements_and_homographies.

        Output:

        * vertex_stabilized_displacements_by_frame_index: A NumPy array of shape
            (num_frames, self.mesh_row_count, self.mesh_col_count, 2)
            containing the stabilized residual displacements of each vertex in the MeshFlow mesh.
            In particular,
            vertex_stabilized_displacements_by_frame_index[frame_index][row][col][0]
            contains the residual x-displacement (the x-displacement in addition to any imposed by
            global homographies) of the mesh vertex at the given row and col from frame 0 to frame
            frame_index, both inclusive.
            vertex_unstabilized_displacements_by_frame_index[frame_index][row][col][1]
            contains the corresponding y-displacement.
        '''


        frame_height, frame_width = vertex_unstabilized_displacements_by_frame_index[0].shape[:2]

        off_diagonal_coefficients, on_diagonal_coefficients = self._get_jacobi_method_input(num_frames, frame_height, frame_width, optimization_formula, adaptive_weights_definition, homographies)

        # vertex_unstabilized_displacements_by_frame_index is indexed by
        # frame_index, then row, then col, then velocity component.
        # Instead, vertex_unstabilized_displacements_by_coord is indexed by
        # row, then col, then frame_index, then velocity component;
        # this rearrangement should allow for faster access during the optimization step.
        vertex_unstabilized_displacements_by_coord = np.moveaxis(
            vertex_unstabilized_displacements_by_frame_index, 0, 2
        )
        vertex_stabilized_displacements_by_coord = np.empty(vertex_unstabilized_displacements_by_coord.shape)
        # TODO parallelize
        with tqdm.trange((self.mesh_row_count + 1) * (self.mesh_col_count + 1)) as t:
            t.set_description('Computing stabilized mesh displacements')
            for mesh_coords_flattened in t:
                mesh_row = mesh_coords_flattened // (self.mesh_row_count + 1)
                mesh_col = mesh_coords_flattened % (self.mesh_col_count + 1)
                vertex_unstabilized_displacements = vertex_unstabilized_displacements_by_coord[mesh_row][mesh_col]
                vertex_stabilized_displacements = self._get_jacobi_method_output(
                    off_diagonal_coefficients, on_diagonal_coefficients,
                    vertex_unstabilized_displacements,
                    vertex_unstabilized_displacements
                )
                vertex_stabilized_displacements_by_coord[mesh_row][mesh_col] = vertex_stabilized_displacements

            vertex_stabilized_displacements_by_frame_index = np.moveaxis(
                vertex_stabilized_displacements_by_coord, 2, 0
            )

        return vertex_stabilized_displacements_by_frame_index


    def _get_jacobi_method_input(self, num_frames, frame_width, frame_height, optimization_formula, adaptive_weights_definition, homographies):
        '''
        Helper method for _get_stabilized_displacements.
        The Jacobi method (see https://en.wikipedia.org/w/index.php?oldid=1036645158),
        approximates a solution for the vector x in the equation
        A x = b
        where A is a matrix of constants and b is a vector of constants.
        Return the values in matrix A given the video's features and the user's chosen method.

        Input:

        * num_frames: The number of frames in the video.
        * optimization_formula: Which method to use to compute the Jacobi method optimization.
        * adaptive_weights_definition: Which definition to use for the energy function's adaptive
            weights.
        * frame_width: the width of the video's frames.
        * frame_height: the height of the video's frames.
        * homographies: A NumPy array of homographies as generated by
            _get_unstabilized_vertex_displacements_and_homographies.

        Output:

        A tuple of the following items in order.

        * off_diagonal_coefficients: A 2D NumPy array containing the off-diagonal entries of A.
            Specifically, off_diagonal_coefficients[i, j] = A_{i, j} where i != j, and all
            on-diagonal entries off_diagonal_coefficients[i, i] = 0.
            In the Wikipedia link, this matrix is L + U.
        * on_diagonal_coefficients: A 1D NumPy array containing the on-diagonal entries of A.
            Specifically, on_diagonal_coefficients[i] = A_{i, i}.
        '''

        # row_indexes[row][col] = row, col_indexes[row][col] = col
        row_indexes, col_indexes = np.indices((num_frames, num_frames))

        # regularization_weights[t, r] is a weight constant applied to the regularization term.
        # In the paper, regularization_weights[t, r] is denoted as w_{t,r}.
        # NOTE that regularization_weights[i, i] = 0.
        regularization_weights = np.exp(
            -np.square((3 / self.temporal_smoothing_radius) * (row_indexes - col_indexes))
        )

        # adaptive_weights[t] is a weight, derived from properties of the frames, applied to the
        # regularization term corresponding to the frame at index t
        # Note that the paper does not specify the weight to apply to the last frame (which does not
        # have a velocity), so we assume it is the same as the second-to-last frame.
        # In the paper, adaptive_weights[t] is denoted as \lambda_{t}.
        adaptive_weights = self._get_adaptive_weights(num_frames, frame_width, frame_height, adaptive_weights_definition, homographies)
        # adaptive_weights = np.full((num_frames,), 10)

        # combined_adaptive_regularization_weights[t, r] = \lambda_{t} w_{t, r}
        combined_adaptive_regularization_weights = np.matmul(np.diag(adaptive_weights), regularization_weights)

        if optimization_formula == MeshFlowStabilizer.OPTIMIZATION_FORMULA_ORIGINAL:
            # the off-diagonal entry at cell [t, r] is written as
            # -2 * \lambda_{t} w_{t, r}
            off_diagonal_coefficients = -2 * combined_adaptive_regularization_weights

            # the on-diagonal entry at cell [t, t] is written as
            # 1 + 2 * \sum_{r \in \Omega_{t}, r \neq t} \lambda_{t} w_{t, r}.
            # NOTE Since w_{t, t} = 0,
            # we can ignore the r \neq t constraint on the sum and write the on-diagonal entry at
            # cell [t, t] as
            # 1 + 2 * \sum{r \in \Omega_{t}} \lambda_{t} w_{t, r}.
            on_diagonal_coefficients = 1 + 2 * np.sum(combined_adaptive_regularization_weights, axis=1)
        elif optimization_formula == MeshFlowStabilizer.OPTIMIZATION_FORMULA_DERIVED:
            # combined_adaptive_regularization_weights[t, r] = \lambda_{t} w_{t, r} - \lambda_{r} w_{r, t}
            combined_subtracted_adaptive_regularization_weights = combined_adaptive_regularization_weights - np.transpose(combined_adaptive_regularization_weights)
            # the off-diagonal entry at cell [t, r] is written as
            # -(\lambda_{t} w_{t, r} - \lambda_{r} w_{r, t})
            off_diagonal_coefficients = -combined_subtracted_adaptive_regularization_weights

            # the on-diagonal entry at cell [t, t] is written as
            # 1 + \sum_{r \in \Omega_{t}, r \neq t} (\lambda_{t} w_{t, r} - \lambda_{r} w_{r, t}).
            # NOTE Since \lambda_{t} w_{t, t} - \lambda_{t} w_{t, t} = 0,
            # we can ignore the r \neq t constraint on the sum and write the on-diagonal entry at
            # cell [t, t] as
            # 1 + \sum{r \in \Omega_{t}} (\lambda_{t} w_{t, r} - \lambda_{r} w_{r, t})
            on_diagonal_coefficients = 1 + np.sum(combined_adaptive_regularization_weights, axis=1)

        # set coefficients to 0 for appropriate t, r; see https://stackoverflow.com/a/36247680
        off_diagonal_mask = np.zeros(off_diagonal_coefficients.shape)
        for i in range(-self.temporal_smoothing_radius, self.temporal_smoothing_radius + 1):
            off_diagonal_mask += np.diag(np.ones(num_frames - abs(i)), i)
        off_diagonal_coefficients = np.where(off_diagonal_mask, off_diagonal_coefficients, 0)

        return (off_diagonal_coefficients, on_diagonal_coefficients)


    def _get_adaptive_weights(self, num_frames, frame_width, frame_height, adaptive_weights_definition, homographies):
        '''
        Helper method for _get_jacobi_method_input.
        Return the array of adaptive weights for use in the energy function.

        Input:

        * num_frames: The number of frames in the video.
        * frame_width: the width of the video's frames.
        * frame_height: the height of the video's frames.
        * adaptive_weights_definition: Which definition to use for the energy function's adaptive
            weights.
        * homographies: A NumPy array of homographies as generated by
            _get_unstabilized_vertex_displacements_and_homographies.

        Output:

        * adaptive_weights: A NumPy array of size
            (num_frames,).
            adaptive_weights[t] is a weight, derived from properties of the frames, applied to the
            regularization term corresponding to the frame at index t.
            Note that the paper does not specify the weight to apply to the last frame (which does
            not have a velocity), so we assume it is the same as the second-to-last frame.
            In the paper, adaptive_weights[t] is denoted as \lambda_{t}.
        '''

        if adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL or adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED:
            # the adaptive weights are determined by plugging the eigenvalues of each homography's
            # affine component into a linear model
            homography_affine_components = homographies.copy()
            homography_affine_components[:, 2, :] = [0, 0, 1]
            adaptive_weights = np.empty((num_frames,))

            for frame_index in range(num_frames):
                homography = homography_affine_components[frame_index]
                sorted_eigenvalue_magnitudes = np.sort(np.abs(np.linalg.eigvals(homography)))

                translational_element = math.sqrt((homography[0, 2] / frame_width) ** 2 + (homography[1, 2] / frame_height) ** 2)
                affine_component = sorted_eigenvalue_magnitudes[-2] / sorted_eigenvalue_magnitudes[-1]

                adaptive_weight_candidate_1 = -1.93 * translational_element + 0.95

                if adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL:
                    adaptive_weight_candidate_2 = 5.83 * affine_component + 4.88
                else:  # ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED
                    # TODO double-check this is the correct sign flip
                    adaptive_weight_candidate_2 = 5.83 * affine_component - 4.88

                adaptive_weights[frame_index] = max(
                    min(adaptive_weight_candidate_1, adaptive_weight_candidate_2), 0
                )
        elif adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH:
            adaptive_weights = np.full((num_frames,), self.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH_VALUE)
        elif adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW:
            adaptive_weights = np.full((num_frames,), self.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW_VALUE)

        return adaptive_weights


    def _get_jacobi_method_output(self, off_diagonal_coefficients, on_diagonal_coefficients, x_start, b):
        '''
        Helper method for _get_stabilized_displacements.
        Using the Jacobi method (see https://en.wikipedia.org/w/index.php?oldid=1036645158),
        approximate a solution for the vector x in the equation
        A x = b
        where A is a matrix of constants and b is a vector of constants.

        Return a value of x after performing self.optimization_num_iterations of the Jacobi method.

        Input:

        * off_diagonal_coefficients: A 2D NumPy array containing the off-diagonal entries of A.
            Specifically, off_diagonal_coefficients[i, j] = A_{i, j} where i != j, and all
            on-diagonal entries off_diagonal_coefficients[i, i] = 0.
            In the Wikipedia link, this matrix is L + U.
         * on_diagonal_coefficients: A 1D NumPy array containing the on-diagonal entries of A.
            Specifically, on_diagonal_coefficients[i] = A_{i, i}.
            In the Wikipedia link, this array is the diagonal entries of D.
        * x_start: A NumPy array containing an initial estimate for x.
        * b: A NumPy array containing the constant vector b.

        Output:

        * x: A NumPy array containing the value of x computed with the Jacobi method.
        '''

        x = x_start.copy()

        reciprocal_on_diagonal_coefficients_matrix = np.diag(np.reciprocal(on_diagonal_coefficients))

        for _ in range(self.optimization_num_iterations):
            x = np.matmul(reciprocal_on_diagonal_coefficients_matrix, b - np.matmul(off_diagonal_coefficients, x))

        return x


    def _get_vertex_x_y(self, frame_width, frame_height):
        '''
        Helper method for _get_stabilized_frames_and_crop_boundaries_and_crop_boundaries and _get_unstabilized_vertex_velocities.
        Return a NumPy array that maps [row, col] coordinates to [x, y] coordinates.

        Input:

        * frame_width: the width of the video's frames.
        * frame_height: the height of the video's frames.

        Output:

        row_col_to_vertex_x_y: A CV_32FC2 array (see https://stackoverflow.com/a/47617999)
            containing the coordinates [x, y] of vertices in the mesh. This array is ordered
            so that when this array is reshaped to
            (self.mesh_row_count + 1, self.mesh_col_count + 1, 2),
            the resulting entry in [row, col] contains the coordinates [x, y] of the vertex in the
            top left corner of the cell at the mesh's given row and col.
        '''

        return np.array([
            [[math.ceil((frame_width - 1) * (col / (self.mesh_col_count))),
              math.ceil((frame_height - 1) * (row / (self.mesh_row_count)))]]
            for row in range(self.mesh_row_count + 1)
            for col in range(self.mesh_col_count + 1)
        ], dtype=np.float32)


    def _get_stabilized_frames_and_crop_boundaries(self, num_frames, unstabilized_frames, vertex_unstabilized_displacements_by_frame_index, vertex_stabilized_displacements_by_frame_index):
        '''
        Helper method for stabilize.

        Return stabilized copies of the given unstabilized frames warping them according to the
        given transformation data, as well as boundaries representing how to crop these stabilized
        frames.

        Inspired by the Python MeshFlow implementation available at
        https://github.com/sudheerachary/Mesh-Flow-Video-Stabilization/blob/5780fe750cf7dc35e5cfcd0b4a56d408ce3a9e53/src/MeshFlow.py#L117.

        Input:

        * num_frames: The number of frames in the unstabilized video.
        * unstabilized_frames: A list of the unstabilized frames, each represented as a NumPy array.
        * vertex_unstabilized_displacements_by_frame_index: A NumPy array containing the
            unstabilized residual displacements of each vertex in the MeshFlow mesh, as generated by
            _get_unstabilized_vertex_displacements_and_homographies.
        * vertex_stabilized_displacements_by_frame_index: A NumPy array containing the
            stabilized residual displacements of each vertex in the MeshFlow mesh, as generated by
            _get_stabilized_vertex_displacements.

        Output:

        A tuple of the following items in order.

        * stabilized_frames: A list of the frames in the stabilized video, each represented as a
            NumPy array.
        * crop_boundaries: A tuple of the form
            (left_crop_x, top_crop_y, right_crop_x, bottom_crop_y)
            representing the x- and y-boundaries (all inclusive) of the cropped video.
        '''

        frame_height, frame_width = unstabilized_frames[0].shape[:2]

        # unstabilized_vertex_x_y and stabilized_vertex_x_y are CV_32FC2 NumPy arrays
        # (see https://stackoverflow.com/a/47617999)
        # of the coordinates of the mesh nodes in the stabilized video, indexed from the top left
        # corner and moving left-to-right, top-to-bottom.
        unstabilized_vertex_x_y = self._get_vertex_x_y(frame_width, frame_height)

        # row_col_to_unstabilized_vertex_x_y[row, col] and
        # row_col_to_stabilized_vertex_x_y[row, col]
        # contain the x and y positions of the vertex at the given row and col
        row_col_to_unstabilized_vertex_x_y = np.reshape(unstabilized_vertex_x_y, (self.mesh_row_count + 1, self.mesh_col_count + 1, 2))

        # residual_velocity_diffs_by_frame_index[frame_index] is a CV_32FC2 NumPy array
        # (see https://stackoverflow.com/a/47617999) containing the amount to add to each vertex
        # coordinate to transform it from its unstabilized position at frame frame_index to its
        # stabilized position at frame frame_index.
        # Since the current displacements are given by
        # vertex_unstabilized_displacements[frame_index],
        # and the final displacements are given by
        # vertex_stabilized_displacements[frame_index], adding the difference of the two
        # produces the desired result.
        stabilized_motion_mesh_by_frame_index = np.reshape(
            vertex_stabilized_displacements_by_frame_index - vertex_unstabilized_displacements_by_frame_index,
            (num_frames, -1, 1, 2)
        )

        # Construct map from the stabilized frame to the unstabilized frame.
        # If (x_s, y_s) in the stabilized video is taken from (x_u, y_u) in the unstabilized
        # video, then
        # stabilized_y_x_to_unstabilized_x[y_s, x_s] = x_u,
        # stabilized_y_x_to_unstabilized_y[y_s, x_s] = y_u, and
        # frame_stabilized_y_x_to_stabilized_x_y[y_s, x_s] = [x_u, y_u].
        # NOTE the inverted coordinate order. This setup allows us to index into map just like
        # we index into the image. Each point [x_u, y_u] in the array is in OpenCV's expected
        # order so we can easily apply homographies to those points.
        # NOTE If a given coordinate's value is not changed by the subsequent steps, then that
        # coordinate falls outside the stabilized image (so in the output image, that image
        # should be filled with a border color).
        # Since these arrays' default values fall outside the unstabilized image, remap will
        # fill in those coordinates in the stabilized image with the border color as desired.
        frame_stabilized_y_x_to_unstabilized_x_template = np.full((frame_height, frame_width), frame_width + 1)
        frame_stabilized_y_x_to_unstabilized_y_template = np.full((frame_height, frame_width), frame_height + 1)
        frame_stabilized_y_x_to_stabilized_x_y_template = np.swapaxes(np.indices((frame_width, frame_height), dtype=np.float32), 0, 2)
        frame_stabilized_x_y_template = frame_stabilized_y_x_to_stabilized_x_y_template.reshape((-1, 1, 2))

        # left_crop_x_by_frame_index[frame_index] contains the x-value where the left edge
        # where frame frame_index would be cropped to produce a rectangular image;
        # right_crop_x_by_frame_index, top_crop_y_by_frame_index, and
        # bottom_crop_y_by_frame_index are analogous
        left_crop_x_by_frame_index = np.full(num_frames, 0)
        right_crop_x_by_frame_index = np.full(num_frames, frame_width - 1)
        top_crop_y_by_frame_index = np.full(num_frames, 0)
        bottom_crop_y_by_frame_index = np.full(num_frames, frame_height - 1)

        stabilized_frames = []
        with tqdm.trange(num_frames) as t:
            t.set_description('Warping frames')
            for frame_index in t:
                unstabilized_frame = unstabilized_frames[frame_index]

                # Construct map from the stabilized frame to the unstabilized frame.
                # If (x_s, y_s) in the stabilized video is taken from (x_u, y_u) in the unstabilized
                # video, then
                # stabilized_y_x_to_unstabilized_x[y_s, x_s] = x_u,
                # stabilized_y_x_to_unstabilized_y[y_s, x_s] = y_u, and
                # frame_stabilized_y_x_to_stabilized_x_y[y_s, x_s] = [x_u, y_u].
                # NOTE the inverted coordinate order. This setup allows us to index into map just like
                # we index into the image. Each point [x_u, y_u] in the array is in OpenCV's expected
                # order so we can easily apply homographies to those points.
                # NOTE If a given coordinate's value is not changed by the subsequent steps, then that
                # coordinate falls outside the stabilized image (so in the output image, that image
                # should be filled with a border color).
                # Since these arrays' default values fall outside the unstabilized image, remap will
                # fill in those coordinates in the stabilized image with the border color as desired.
                frame_stabilized_y_x_to_unstabilized_x = np.copy(frame_stabilized_y_x_to_unstabilized_x_template)
                frame_stabilized_y_x_to_unstabilized_y = np.copy(frame_stabilized_y_x_to_unstabilized_y_template)
                frame_stabilized_x_y = np.copy(frame_stabilized_x_y_template)

                # Determine the coordinates of the mesh vertices in the stabilized video.
                # The current displacements are given by vertex_unstabilized_displacements, and
                # the desired displacements are given by vertex_stabilized_displacements,
                # so adding the difference of the two transforms the frame as desired.
                stabilized_vertex_x_y = unstabilized_vertex_x_y + stabilized_motion_mesh_by_frame_index[frame_index]

                row_col_to_stabilized_vertex_x_y = np.reshape(stabilized_vertex_x_y, (self.mesh_row_count + 1, self.mesh_col_count + 1, 2))
                # Look at each face of the mesh. Since we know the original and transformed coordinates
                # of its four vertices, we can construct a homography to fill in the remaining pixels
                # TODO parallelize
                for cell_top_left_row in range(self.mesh_row_count):
                    for cell_top_left_col in range(self.mesh_col_count):

                        # Construct a mask representing the stabilized cell.
                        # Since we know the cell's boundaries before and after stabilization, we can
                        # construct a homography representing this cell's warp and then apply it to
                        # the unstabilized cell (which is just a rectangle) to construct the stabilized
                        # cell.
                        unstabilized_cell_bounds = row_col_to_unstabilized_vertex_x_y[cell_top_left_row:cell_top_left_row+2, cell_top_left_col:cell_top_left_col+2].reshape(-1, 2)
                        stabilized_cell_bounds = row_col_to_stabilized_vertex_x_y[cell_top_left_row:cell_top_left_row+2, cell_top_left_col:cell_top_left_col+2].reshape(-1, 2)
                        unstabilized_to_stabilized_homography, _ = cv2.findHomography(unstabilized_cell_bounds, stabilized_cell_bounds)
                        stabilized_to_unstabilized_homography, _ = cv2.findHomography(stabilized_cell_bounds, unstabilized_cell_bounds)

                        unstabilized_cell_x_bounds, unstabilized_cell_y_bounds = np.transpose(unstabilized_cell_bounds)
                        unstabilized_cell_left_x = math.floor(np.min(unstabilized_cell_x_bounds))
                        unstabilized_cell_right_x = math.ceil(np.max(unstabilized_cell_x_bounds))
                        unstabilized_cell_top_y = math.floor(np.min(unstabilized_cell_y_bounds))
                        unstabilized_cell_bottom_y = math.ceil(np.max(unstabilized_cell_y_bounds))

                        unstabilized_cell_mask = np.zeros((frame_height, frame_width))
                        unstabilized_cell_mask[unstabilized_cell_top_y:unstabilized_cell_bottom_y+1, unstabilized_cell_left_x:unstabilized_cell_right_x+1] = 255
                        stabilized_cell_mask = cv2.warpPerspective(unstabilized_cell_mask, unstabilized_to_stabilized_homography, (frame_width, frame_height))

                        cell_unstabilized_x_y = cv2.perspectiveTransform(frame_stabilized_x_y, stabilized_to_unstabilized_homography)
                        cell_stabilized_y_x_to_unstabilized_x_y = cell_unstabilized_x_y.reshape((frame_height, frame_width, 2))
                        cell_stabilized_y_x_to_unstabilized_x, cell_stabilized_y_x_to_unstabilized_y = np.moveaxis(cell_stabilized_y_x_to_unstabilized_x_y, 2, 0)

                        # update the overall stabilized-to-unstabilized map, applying this cell's
                        # transformation only to those pixels that are actually part of this cell
                        frame_stabilized_y_x_to_unstabilized_x = np.where(stabilized_cell_mask, cell_stabilized_y_x_to_unstabilized_x, frame_stabilized_y_x_to_unstabilized_x)
                        frame_stabilized_y_x_to_unstabilized_y = np.where(stabilized_cell_mask, cell_stabilized_y_x_to_unstabilized_y, frame_stabilized_y_x_to_unstabilized_y)

                stabilized_frame = cv2.remap(
                    unstabilized_frame,
                    frame_stabilized_y_x_to_unstabilized_x.reshape((frame_height, frame_width, 1)).astype(np.float32),
                    frame_stabilized_y_x_to_unstabilized_y.reshape((frame_height, frame_width, 1)).astype(np.float32),
                    cv2.INTER_LINEAR,
                    borderValue=self.color_outside_image_area_bgr
                )

                # crop the frame
                # left edge: the maximum stabilized x_s that corresponds to the unstabilized
                # x_u = 0

                stabilized_image_x_matching_unstabilized_left_edge = np.where(np.abs(frame_stabilized_y_x_to_unstabilized_x - 0) < 1)[1]
                if stabilized_image_x_matching_unstabilized_left_edge.size > 0:
                    left_crop_x_by_frame_index[frame_index] = np.max(stabilized_image_x_matching_unstabilized_left_edge)

                # right edge: the minimum stabilized x_s that corresponds to the stabilized
                # x_u = frame_width - 1

                stabilized_image_x_matching_unstabilized_right_edge = np.where(np.abs(frame_stabilized_y_x_to_unstabilized_x - (frame_width - 1)) < 1)[1]
                if stabilized_image_x_matching_unstabilized_right_edge.size > 0:
                    right_crop_x_by_frame_index[frame_index] = np.min(stabilized_image_x_matching_unstabilized_right_edge)

                # top edge: the maximum stabilized y_s that corresponds to the unstabilized
                # y_u = 0

                stabilized_image_y_matching_unstabilized_top_edge = np.where(np.abs(frame_stabilized_y_x_to_unstabilized_y - 0) < 1)[0]
                if stabilized_image_y_matching_unstabilized_top_edge.size > 0:
                    top_crop_y_by_frame_index[frame_index] = np.max(stabilized_image_y_matching_unstabilized_top_edge)

                # bottom edge: the minimum stabilized y_s that corresponds to the unstabilized
                # y_u = frame_height - 1

                stabilized_image_y_matching_unstabilized_bottom_edge = np.where(np.abs(frame_stabilized_y_x_to_unstabilized_y - (frame_height - 1)) < 1)[0]
                if stabilized_image_y_matching_unstabilized_bottom_edge.size > 0:
                    bottom_crop_y_by_frame_index[frame_index] = np.min(stabilized_image_y_matching_unstabilized_bottom_edge)

                # left_line_start_point = np.array([stabilized_left_crop_x_by_frame_index[frame_index], 0], dtype=np.int32)
                # left_line_end_point = np.array([stabilized_left_crop_x_by_frame_index[frame_index], frame_height - 1], dtype=np.int32)
                # stabilized_frame = cv2.line(stabilized_frame, left_line_start_point, left_line_end_point, (255, 0, 0))

                # right_line_start_point = np.array([stabilized_right_crop_x_by_frame_index[frame_index], 0], dtype=np.int32)
                # right_line_end_point = np.array([stabilized_right_crop_x_by_frame_index[frame_index], frame_height - 1], dtype=np.int32)
                # stabilized_frame = cv2.line(stabilized_frame, right_line_start_point, right_line_end_point, (255, 0, 0))

                # top_line_start_point = np.array([0, stabilized_top_crop_y_by_frame_index[frame_index]], dtype=np.int32)
                # top_line_end_point = np.array([frame_width - 1, stabilized_top_crop_y_by_frame_index[frame_index]], dtype=np.int32)
                # stabilized_frame = cv2.line(stabilized_frame, top_line_start_point, top_line_end_point, (255, 0, 0))

                # bottom_line_start_point = np.array([0, stabilized_bottom_crop_y_by_frame_index[frame_index]], dtype=np.int32)
                # bottom_line_end_point = np.array([frame_width - 1, stabilized_bottom_crop_y_by_frame_index[frame_index]], dtype=np.int32)
                # stabilized_frame = cv2.line(stabilized_frame, bottom_line_start_point, bottom_line_end_point, (255, 0, 0))


                stabilized_frames.append(stabilized_frame)

        # the final video crop is the one that would adequately crop every single frame
        left_crop_x = np.max(left_crop_x_by_frame_index)
        right_crop_x = np.min(right_crop_x_by_frame_index)
        top_crop_y = np.max(top_crop_y_by_frame_index)
        bottom_crop_y = np.min(bottom_crop_y_by_frame_index)

        return (stabilized_frames, (left_crop_x, top_crop_y, right_crop_x, bottom_crop_y))


    def _crop_frames(self, uncropped_frames, crop_boundaries):
        '''
        Return copies of the given frames that have been cropped according to the given crop
        boundaries.

        Input:

        * uncropped_frames: A list of the frames to crop, each represented as a NumPy array.
        * crop_boundaries: A tuple of the form
            (left_crop_x, top_crop_y, right_crop_x, bottom_crop_y)
            representing the x- and y-boundaries (all inclusive) of the cropped video.

        Output:

        * cropped_frames: A list of the frames cropped according to the crop boundaries.

        '''

        frame_height, frame_width = uncropped_frames[0].shape[:2]
        left_crop_x, top_crop_y, right_crop_x, bottom_crop_y = crop_boundaries

        # There are two ways to scale up the image: increase its width to fill the original width,
        # scaling the height appropriately, or increase its height to fill the original height,
        # scaling the width appropriately. At least one of these options will result in the image
        # completely filling the frame.
        uncropped_aspect_ratio = frame_width / frame_height
        cropped_aspect_ratio = (right_crop_x + 1 - left_crop_x) / (bottom_crop_y + 1 - top_crop_y)

        if cropped_aspect_ratio >= uncropped_aspect_ratio:
            # the cropped image is proportionally wider than the original, so to completely fill the
            # frame, it must be scaled so its height matches the frame height
            uncropped_to_cropped_scale_factor = frame_height / (bottom_crop_y + 1 - top_crop_y)
        else:
            # the cropped image is proportionally taller than the original, so to completely fill
            # the frame, it must be scaled so its width matches the frame width
            uncropped_to_cropped_scale_factor = frame_width / (right_crop_x + 1 - left_crop_x)

        cropped_frames = []
        for uncropped_frame in uncropped_frames:
            cropped_frames.append(cv2.resize(
                uncropped_frame[top_crop_y:bottom_crop_y+1, left_crop_x:right_crop_x+1],
                (frame_width, frame_height),
                fx=uncropped_to_cropped_scale_factor,
                fy=uncropped_to_cropped_scale_factor
            ))

        return cropped_frames


    def _compute_cropping_ratio_and_distortion_score(self, num_frames, unstabilized_frames, cropped_frames):
        '''
        Helper function for stabilize.

        Compute the cropping ratio and distortion score for the given stabilization using the
        definitions of these metrics in the original paper.

        Input:

        * num_frames: The number of frames in the video.
        * unstabilized_frames: A list of the unstabilized frames, each represented as a NumPy array.
        * stabilized_frames: A list of the stabilized frames, each represented as a NumPy array.

        Output:

        A tuple of the following items in order.

        * cropping_ratio: The cropping ratio of the stabilized video. Per the original paper, the
            cropping ratio of each frame is the scale component of its unstabilized-to-cropped
            homography, and the cropping ratio of the overall video is the average of the frames'
            cropping ratios.
        * distortion_score: The distortion score of the stabilized video. Per the original paper,
            the distortion score of each frame is ratio of the two largest eigenvalues of the
            affine part of its unstabilized-to-cropped homography, and the distortion score of the
            overall video is the greatest of its frames' distortion scores.
        '''

        cropping_ratios = np.empty((num_frames), dtype=np.float32)
        distortion_scores = np.empty((num_frames), dtype=np.float32)

        with tqdm.trange(num_frames) as t:
            t.set_description('Computing cropping ratio and distortion score')
            for frame_index in t:
                unstabilized_frame = unstabilized_frames[frame_index]
                cropped_frame = cropped_frames[frame_index]
                unstabilized_features, cropped_features = self._get_all_matched_features_between_images(
                    unstabilized_frame, cropped_frame
                )

                unstabilized_to_cropped_homography, _ = cv2.findHomography(unstabilized_features, cropped_features)

                # the scaling component has x-component cropped_to_unstabilized_homography[0][0]
                # and y-component cropped_to_unstabilized_homography[1][1],
                # so the fraction of the enlarged video that actually fits in the frame is
                # 1 / (cropped_to_unstabilized_homography[0][0] * cropped_to_unstabilized_homography[1][1])
                cropping_ratio = 1 / (unstabilized_to_cropped_homography[0][0] * unstabilized_to_cropped_homography[1][1])
                cropping_ratios[frame_index] = cropping_ratio

                affine_component = np.copy(unstabilized_to_cropped_homography)
                affine_component[2] = [0, 0, 1]
                eigenvalue_magnitudes = np.sort(np.abs(np.linalg.eigvals(affine_component)))
                distortion_score = eigenvalue_magnitudes[-2] / eigenvalue_magnitudes[-1]
                distortion_scores[frame_index] = distortion_score

        return (np.mean(cropping_ratios), np.min(distortion_scores))



    def _compute_stability_score(self, num_frames, vertex_stabilized_displacements_by_frame_index):
        '''
        Helper function for stabilize.

        Compute the stability score for the given stabilization using the definitions of these
        metrics in the original paper.

        Input:

        * num_frames: The number of frames in the video.
        * vertex_stabilized_displacements_by_frame_index: A NumPy array containing the
            stabilized residual displacements of each vertex in the MeshFlow mesh, as generated by
            _get_stabilized_vertex_displacements.

        Output:

        * stability_score: The stability score of the stabilized video. Per the original paper, the
            stability score for each vertex is derived from the representation of its vertex profile
            (vector of velocities) in the frequency domain. Specifically, it is the fraction of the
            representation's total energy that is contained within its second to sixth lowest
            frequencies. The stability score of the overall video is the average of the vertices'
            average x- and y-stability scores.
        '''

        vertex_stabilized_x_dispacements_by_row_and_col, vertex_stabilized_y_dispacements_by_row_and_col = np.swapaxes(vertex_stabilized_displacements_by_frame_index, 0, 3)
        vertex_x_profiles_by_row_and_col = np.diff(vertex_stabilized_x_dispacements_by_row_and_col)
        vertex_y_profiles_by_row_and_col = np.diff(vertex_stabilized_y_dispacements_by_row_and_col)

        vertex_x_freq_energies_by_row_and_col = np.square(np.abs(np.fft.fft(vertex_x_profiles_by_row_and_col)))
        vertex_y_freq_energies_by_row_and_col = np.square(np.abs(np.fft.fft(vertex_y_profiles_by_row_and_col)))

        vertex_x_total_freq_energy_by_row_and_col = np.sum(vertex_x_freq_energies_by_row_and_col, axis=2)
        vertex_y_total_freq_energy_by_row_and_col = np.sum(vertex_y_freq_energies_by_row_and_col, axis=2)

        vertex_x_low_freq_energy_by_row_and_col = np.sum(vertex_x_freq_energies_by_row_and_col[:, :, 1:6], axis=2)
        vertex_y_low_freq_energy_by_row_and_col = np.sum(vertex_y_freq_energies_by_row_and_col[:, :, 1:6], axis=2)

        x_stability_scores_by_row_and_col = vertex_x_low_freq_energy_by_row_and_col / vertex_x_total_freq_energy_by_row_and_col
        y_stability_scores_by_row_and_col = vertex_y_low_freq_energy_by_row_and_col / vertex_y_total_freq_energy_by_row_and_col

        x_stability_score = np.mean(x_stability_scores_by_row_and_col)
        y_stability_score = np.mean(y_stability_scores_by_row_and_col)

        return (x_stability_score + y_stability_score) / 2.0


    def _display_unstablilized_and_stabilized_video_loop(self, num_frames, frames_per_second, unstabilized_frames, stabilized_frames):
        '''
        Helper function for stabilize.

        Display a loop of the stabilized and unstabilized videos.

        Input:

        * num_frames: The number of frames in the video.
        * frames_per_second: The video framerate in frames per second.
        * unstabilized_frames: A list of the unstabilized frames, each represented as a NumPy array.
        * stabilized_frames: A list of the stabilized frames, each represented as a NumPy array.

        Output:

        (The unstabilized and stabilized videos loop. Pressing the Q key closes them.)
        '''

        milliseconds_per_frame = int(1000/frames_per_second)
        while True:
            for i in range(num_frames):
                cv2.imshow('unstabilized and stabilized video', np.vstack((unstabilized_frames[i], stabilized_frames[i])))
                if cv2.waitKey(milliseconds_per_frame) & 0xFF == ord('q'):
                    return


    def _write_stabilized_video(self, output_path, num_frames, frames_per_second, codec, stabilized_frames):
        '''
        Helper method for stabilize.
        Write the given stabilized frames as a video to the given path.

        Input:
        * output_path: The path where the stabilized version of the video should be placed.
        * num_frames: The number of frames in the video.
        * frames_per_second: The video framerate in frames per second.
        * codec: The video codec.
        * stabilized_frames: A list of the frames in the stabilized video, each represented as a
            NumPy array.

        Output:

        (The video is saved to output_path.)
        '''

        # adapted from https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
        frame_height, frame_width = stabilized_frames[0].shape[:2]
        video = cv2.VideoWriter(
            output_path,
            codec,
            frames_per_second,
            (frame_width, frame_height)
        )

        with tqdm.trange(num_frames) as t:
            t.set_description(f'Writing stabilized video to <{output_path}>')
            for frame_index in t:
                video.write(stabilized_frames[frame_index])

        video.release()


def main():
    # TODO get video path from command line args
    input_path = 'videos/data_small-shaky-5.m4v'
    output_path = 'videos/data_small-shaky-5_stabilized.m4v'
    stabilizer = MeshFlowStabilizer()
    cropping_ratio, distortion_score, stability_score = stabilizer.stabilize(
        input_path, output_path,
        optimization_formula=MeshFlowStabilizer.OPTIMIZATION_FORMULA_ORIGINAL,
        adaptive_weights_definition=MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH
    )
    print('cropping ratio:', cropping_ratio)
    print('distortion score:', distortion_score)
    print('stability score:', stability_score)


if __name__ == '__main__':
    main()