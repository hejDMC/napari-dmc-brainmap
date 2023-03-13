@dataclass
class DataInputs(InputContainer):
    """Container for image-related ("Data") inputs."""

    signal_array: numpy.ndarray = None
    background_array: numpy.ndarray = None
    voxel_size_z: float = 5
    voxel_size_y: float = 2
    voxel_size_x: float = 2

    def as_core_arguments(self) -> dict:
        """
        Passes voxel size data as one tuple instead of 3 individual floats
        """
        data_input_dict = super().as_core_arguments()
        data_input_dict["voxel_sizes"] = (
            self.voxel_size_z,
            self.voxel_size_y,
            self.voxel_size_x,
        )
        # del operator doesn't affect self, because asdict creates a copy of
        # fields.
        del data_input_dict["voxel_size_z"]
        del data_input_dict["voxel_size_y"]
        del data_input_dict["voxel_size_x"]
        return data_input_dict

    @property
    def nplanes(self):
        return len(self.signal_array)

    @classmethod
    def widget_representation(cls) -> dict:
        return dict(
            data_options=html_label_widget("Data:"),
            voxel_size_z=cls._custom_widget(
                "voxel_size_z", custom_label="Voxel size (z)"
            ),
            voxel_size_y=cls._custom_widget(
                "voxel_size_y", custom_label="Voxel size (y)"
            ),
            voxel_size_x=cls._custom_widget(
                "voxel_size_x", custom_label="Voxel size (x)"
            ),
        )