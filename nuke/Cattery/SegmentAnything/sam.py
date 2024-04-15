from PySide2 import QtWidgets, QtOpenGL

import nuke

SAM_TABLE_TOOLTIP = "<b>tracks</b><br>sam"


def find_widget_by_tooltip(tooltip):
    stack = QtWidgets.QApplication.instance().allWidgets()
    while stack:
        widget = stack.pop()
        if widget.toolTip() == tooltip:
            return widget


def find_widget_by_text(text):
    text = str(text)
    stack = QtWidgets.QApplication.instance().allWidgets()
    stack = [widget for widget in stack if isinstance(widget, QtWidgets.QLineEdit)]
    while stack:
        widget = stack.pop()
        if widget.text() == text:
            return widget


def get_widget_from_node(node_name: str) -> QtOpenGL.QGLWidget:
    """Retrieve the QGLWidget of DAG graph"""
    stack = QtWidgets.QApplication.instance().allWidgets()
    while stack:
        widget = stack.pop()
        if widget.objectName() == node_name:
            return widget


counter = 0


def hide_columns():
    # counter += 1
    table_widget = find_widget_by_tooltip(SAM_TABLE_TOOLTIP)
    table_view = table_widget.findChild(QtWidgets.QTableView)

    table_view.setColumnHidden(4, True)
    table_view.setColumnHidden(5, True)
    table_view.setColumnHidden(6, True)
    table_view.setColumnHidden(7, True)
    table_view.setColumnHidden(9, True)


def sam():
    if nuke.thisKnob().name() == "showPanel":
        node_name = nuke.thisNode().name()

        # Open the tracker node panel
        nuke.toNode(f"{node_name}.Tracker1").showControlPanel(True)

        # Hide the tracker node widget
        unique_id = nuke.thisNode().knob("unique_id").value()
        unique_id = int(unique_id + 1)
        unique_id_widget = find_widget_by_text(unique_id)
        tracker_window = unique_id_widget.window()
        tracker_window.hide()

        table_widget = find_widget_by_tooltip(SAM_TABLE_TOOLTIP)
        table_view = table_widget.findChild(QtWidgets.QTableView)
        model = table_view.model()

        # Hide the unneeded columns, also make sure it
        # stays hidden after any refresh.
        hide_columns()
        model.modelAboutToBeReset.connect(lambda: hide_columns())
        model.modelReset.connect(lambda: hide_columns())
        model.dataChanged.connect(lambda: hide_columns())

    if nuke.thisKnob().name() == "hidePanel":
        node_name = nuke.thisNode().name()

        table_widget = find_widget_by_tooltip(SAM_TABLE_TOOLTIP)
        table_view = table_widget.findChild(QtWidgets.QTableView)
        model = table_view.model()

        try:
            model.modelAboutToBeReset.disconnect()
            model.modelReset.disconnect()
            model.dataChanged.disconnect()
        except RuntimeError:
            pass

        # Close the tracker node panel
        nuke.toNode(f"{node_name}.Tracker1").hideControlPanel()


# nuke.toNode("Segment_Anything")["knobChanged"].setValue("sam.sam()")
