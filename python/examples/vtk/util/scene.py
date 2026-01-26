"""
Scene setup and text rendering utilities for VTK examples
"""

import vtk


def create_text_actor(text, font_size=40, color=(1.0, 1.0, 1.0),
                      position=(0.03, 0.30), justification='left'):
    """
    Create a VTK text actor with common settings

    Parameters
    ----------
    text : str
        Text to display
    font_size : int
        Font size in points
    color : tuple
        RGB color (0.0-1.0 range)
    position : tuple
        (x, y) position in normalized viewport coordinates (0.0-1.0)
    justification : str
        Text justification: 'left', 'center', or 'right'

    Returns
    -------
    vtk.vtkTextActor
        Configured text actor
    """
    actor = vtk.vtkTextActor()
    actor.SetInput(text)

    # Configure text properties
    text_prop = actor.GetTextProperty()
    text_prop.SetFontSize(font_size)
    text_prop.SetColor(*color)
    text_prop.SetVerticalJustificationToCentered()

    # Set justification
    if justification == 'left':
        text_prop.SetJustificationToLeft()
    elif justification == 'center':
        text_prop.SetJustificationToCentered()
    elif justification == 'right':
        text_prop.SetJustificationToRight()
    else:
        raise ValueError(f"Invalid justification: {justification}")

    # Set position in normalized viewport coordinates
    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    actor.SetPosition(*position)

    return actor


def create_renderer_with_text_strip(main_bg_color=(27/255, 43/255, 52/255),
                                    text_bg_color=(0.090, 0.143, 0.173)):
    """
    Create dual viewport layout: main renderer + bottom text strip

    Mimics the common C++ pattern:
    - Main viewport: top 88% (y: 0.12 - 1.0)
    - Text viewport: bottom 12% (y: 0.0 - 0.12)

    Parameters
    ----------
    main_bg_color : tuple
        RGB background color for main renderer
    text_bg_color : tuple
        RGB background color for text strip (darker tone)

    Returns
    -------
    tuple
        (main_renderer, text_renderer)
    """
    # Main renderer (top 88%)
    main_renderer = vtk.vtkRenderer()
    main_renderer.SetBackground(*main_bg_color)
    main_renderer.SetViewport(0.0, 0.12, 1.0, 1.0)

    # Text renderer (bottom 12%)
    text_renderer = vtk.vtkRenderer()
    text_renderer.SetBackground(*text_bg_color)
    text_renderer.SetViewport(0.0, 0.0, 1.0, 0.12)
    text_renderer.InteractiveOff()

    return main_renderer, text_renderer


def setup_basic_scene(window_size=(800, 600), bg_color=(27/255, 43/255, 52/255)):
    """
    Create basic VTK scene with renderer, window, and interactor

    Parameters
    ----------
    window_size : tuple
        (width, height) in pixels
    bg_color : tuple
        RGB background color

    Returns
    -------
    tuple
        (renderer, render_window, interactor)
    """
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(*bg_color)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(*window_size)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    return renderer, render_window, interactor
