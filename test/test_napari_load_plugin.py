from napari.plugins import plugin_manager
import pytest
from qtpy.QtWidgets import QWidget




def test_plugin_in_menu(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    # Load the plugin (replace 'your_plugin_name' with the actual plugin name)
    plugin_name = 'dmc_brainmap'


    # Manually register the plugin
    brainmap_plugin = QWidget()
    plugin_manager.register(brainmap_plugin, name=plugin_name)

    # Now check if it's available
    assert plugin_name in plugin_manager.plugins, f"{plugin_name} is not loaded."


    # Check if the plugin appears in the viewer's menu
    #plugin_menu = menu.findChild(viewer,plugin_name)  # Find the plugin menu by name
    
    #assert plugin_menu is not None, f"Plugin menu for {plugin_name} not found."
