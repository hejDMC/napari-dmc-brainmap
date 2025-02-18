
def registration_widget():
    # todo probe_track in sharpy_track
    # todo think about solution to check and load atlas data
    @magicgui(
        layout='vertical',
        call_button='start registration GUI'
    )
    def widget(
            viewer: Viewer
    ) -> None:
        reg_viewer = RegistrationViewer(viewer)
        reg_viewer.show()
    return widget
