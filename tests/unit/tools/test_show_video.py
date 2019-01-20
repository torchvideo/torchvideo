import importlib
import sys
from unittest.mock import Mock

import moviepy
import numpy as np
import pytest

import torchvideo
from torchvideo.tools import show_video
import moviepy.editor
import moviepy.video.io.html_tools

original_moviepy_editor = moviepy.editor


class TestShowVideo:
    frames = np.random.randn(10, 32, 32, 3)

    def test_raises_error_if_moviepy_not_available(self):
        # OK, so this is a nasty nasty test with import fuckery going on.
        try:
            # First we make moviepy.editor unimportable
            sys.modules["moviepy.editor"] = None
            # but we have already imported torchvideo.tools so that code is
            # cached with a reference to the *original* moviepy, so we need to reload
            # it to invoke the failure path of the import code
            importlib.reload(torchvideo.tools)
            from torchvideo.tools import show_video

            # Now we've imported torchvideo.tools where moviepy.editor was
            # *not* available.
            with pytest.raises(ImportError, match="please install moviepy"):
                show_video(self.frames)
        finally:
            # We've borked up the system modules and we have to put them right otherwise
            # any tests that depend on torchvideo.tools or moviepy.editor are
            # going to fail.
            # We first restore the original moviepy editor object back
            sys.modules["moviepy.editor"] = original_moviepy_editor
            # I don't think we need to reload moviepy (other tests pass without this
            # line), but for good measure let's reload it just in case it does
            # something useful.
            importlib.reload(moviepy)
            # We also want to reload torchvideo.tools so moviepy *is* available
            importlib.reload(torchvideo.tools)

    def test_uses_ipython_display_if_ipython_is_available(self, monkeypatch):
        display_mock = Mock()
        with monkeypatch.context() as ctx:
            ctx.setattr(moviepy.video.io.html_tools, "ipython_available", True)
            ctx.setattr(
                moviepy.editor.ImageSequenceClip, "ipython_display", display_mock
            )

            importlib.reload(torchvideo.tools)
            show_video(self.frames)

            display_mock.assert_called_once_with()

    def test_fallsback_on_pygame_display(self, monkeypatch):
        show_mock = Mock()
        with monkeypatch.context() as ctx:
            ctx.setattr(moviepy.video.io.html_tools, "ipython_available", False)
            ctx.setattr(moviepy.editor.ImageSequenceClip, "show", show_mock)

            importlib.reload(torchvideo.tools)
            show_video(self.frames)

            show_mock.assert_called_once_with()
