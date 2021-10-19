class Frame:
    '''
    A Frame represents a frame of footage in a video.

    It contains the following attributes:
    pixels_bgr: The frame's pixels expressed in the BGR color space.
    mesh_velocities: The unstabilized velocities of this frame's mesh node pixels, relative to the
        next frame. Specificially, a (MESH_COL_COUNT + 1 by MESH_ROW_COUNT + 1 by 2) array
        containing the x - and y-velocity of each node relative to the next Frame.
    '''

    __slots__ = ('pixels_bgr', 'mesh_velocities')

    def __init__(self, pixels_bgr):
        self.pixels_bgr = pixels_bgr