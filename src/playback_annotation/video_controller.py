from pathlib import Path

from PySide6.QtCore import QObject, QUrl, Signal

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

    VIDEO_AVAILABLE = True
except ImportError:
    QAudioOutput = None
    QMediaPlayer = None
    VIDEO_AVAILABLE = False


class VideoPlaybackController(QObject):
    status_changed = Signal(str)
    error_occurred = Signal(str)
    media_loaded = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._player = None
        self._audio_output = None
        self._video_output = None
        self._source_path: Path | None = None

    @property
    def is_available(self) -> bool:
        return VIDEO_AVAILABLE

    @property
    def has_media(self) -> bool:
        return self._source_path is not None and self._player is not None

    @property
    def position_ms(self) -> int:
        return self._player.position() if self._player is not None else 0

    @property
    def duration_ms(self) -> int:
        return self._player.duration() if self._player is not None else 0

    def set_video_output(self, video_output) -> None:
        self._video_output = video_output
        if self._player is not None:
            self._player.setVideoOutput(video_output)

    def load(self, path: str | Path, autoplay: bool = True) -> Path:
        if not VIDEO_AVAILABLE:
            raise RuntimeError(
                "This PySide6 installation does not include Qt Multimedia."
            )

        source_path = Path(path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {source_path}")

        self._ensure_player()
        self._source_path = source_path
        self._player.setSource(QUrl.fromLocalFile(str(source_path)))
        self.media_loaded.emit(source_path.name)
        self.status_changed.emit(source_path.name)

        if autoplay:
            self.play()

        return source_path

    def play(self) -> bool:
        if not self.has_media:
            return False
        self._player.play()
        self.status_changed.emit("Video playing")
        return True

    def pause(self) -> bool:
        if not self.has_media:
            return False
        self._player.pause()
        self.status_changed.emit("Video paused")
        return True

    def restart(self) -> bool:
        if not self.has_media:
            return False
        self._player.setPosition(0)
        self._player.play()
        self.status_changed.emit("Video restarted")
        return True

    def stop(self) -> bool:
        if self._player is None:
            return False
        self._player.stop()
        self._player.setPosition(0)
        if self._source_path is not None:
            self.status_changed.emit("Video stopped")
        return True

    def seek_ms(self, position_ms: int) -> bool:
        if not self.has_media:
            return False

        resolved = max(0, int(position_ms))
        duration = self.duration_ms
        if duration > 0:
            resolved = min(resolved, duration)

        self._player.setPosition(resolved)
        return True

    def clear(self) -> None:
        if self._player is not None:
            self._player.stop()
            self._player.setPosition(0)
            self._player.setSource(QUrl())
        self._source_path = None
        self.status_changed.emit("No video selected")

    def _ensure_player(self) -> None:
        if self._player is not None:
            return

        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_output)

        if self._video_output is not None:
            self._player.setVideoOutput(self._video_output)

        self._player.errorOccurred.connect(self._handle_error)

    def _handle_error(self, _error, error_string: str = "") -> None:
        message = error_string or "The selected video could not be played."
        self.status_changed.emit(message)
        self.error_occurred.emit(message)