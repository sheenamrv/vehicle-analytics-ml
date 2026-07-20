import json
import socket
import threading
from typing import Any
from PySide6.QtCore import QObject, Signal


class UdpStreamReceiver(QObject):
    sample_received = Signal(dict)
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.host = "0.0.0.0"
        self.port = 0
        self.received_packets = 0
        self.invalid_packets = 0

        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    @property
    def is_listening(self) -> bool:
        thread = self._thread
        return (
            self._socket is not None
            and thread is not None
            and thread.is_alive()
            and not self._stop_event.is_set()
        )

    def start(self, host: str, port: int) -> bool:
        """Bind the UDP socket and start the receive worker."""

        self.stop()

        resolved_host = host.strip() or "0.0.0.0"
        resolved_port = int(port)

        if not 1 <= resolved_port <= 65535:
            self.error_occurred.emit(
                f"Invalid UDP port {resolved_port}. Use a value from 1 to 65535."
            )
            return False

        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        try:
            udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # A timeout lets the worker periodically check whether Stop was requested
            udp_socket.settimeout(0.25)
            udp_socket.bind((resolved_host, resolved_port))
        except OSError as error:
            udp_socket.close()
            self.error_occurred.emit(
                f"Could not start UDP stream on {resolved_host}:{resolved_port}: "
                f"{error}"
            )
            return False

        with self._lock:
            self.host = resolved_host
            self.port = resolved_port
            self.received_packets = 0
            self.invalid_packets = 0
            self._socket = udp_socket
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._receive_loop,
                name=f"UdpStreamReceiver-{resolved_port}",
                daemon=True,
            )
            self._thread.start()

        self.status_changed.emit(
            f"Listening on {self.host}:{self.port} — "
            "0 received, 0 invalid"
        )
        return True

    def stop(self) -> None:
        self._stop_event.set()

        with self._lock:
            udp_socket = self._socket
            worker = self._thread
            self._socket = None
            self._thread = None

        if udp_socket is not None:
            try:
                udp_socket.close()
            except OSError:
                pass

        if (
            worker is not None
            and worker.is_alive()
            and worker is not threading.current_thread()
        ):
            worker.join(timeout=1.0)

        if udp_socket is not None or worker is not None:
            self.status_changed.emit("UDP stream stopped")

    def _receive_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                udp_socket = self._socket

            if udp_socket is None:
                break

            try:
                payload, _sender = udp_socket.recvfrom(65_507)
            except socket.timeout:
                continue
            except OSError as error:
                # Closing the socket during Stop commonly produces an OSError.
                if not self._stop_event.is_set():
                    self.error_occurred.emit(f"UDP receive error: {error}")
                break

            try:
                samples = self.parse_packet(payload)
            except (UnicodeDecodeError, ValueError) as error:
                self.invalid_packets += 1
                self.error_occurred.emit(
                    f"Ignored UDP packet: {error} — "
                    f"{self.received_packets:,} received, "
                    f"{self.invalid_packets:,} invalid"
                )
                continue

            for sample in samples:
                self.received_packets += 1
                self.sample_received.emit(sample)

            self.status_changed.emit(
                f"Listening on {self.host}:{self.port} — "
                f"{self.received_packets:,} received, "
                f"{self.invalid_packets:,} invalid"
            )

    @staticmethod
    def parse_packet(payload: bytes) -> list[dict[str, Any]]:
        text = payload.decode("utf-8").strip()
        if not text:
            raise ValueError("empty packet")

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = UdpStreamReceiver._parse_key_value_packet(text)

        if isinstance(parsed, dict):
            return [parsed]

        if isinstance(parsed, list) and all(
            isinstance(item, dict) for item in parsed
        ):
            return parsed

        raise ValueError(
            "packet must be a JSON object, a list of JSON objects, "
            "or key=value pairs"
        )

    @staticmethod
    def _parse_key_value_packet(text: str) -> dict[str, Any]:
        result: dict[str, Any] = {}

        for token in text.replace(";", ",").split(","):
            token = token.strip()
            if not token:
                continue

            if "=" not in token:
                raise ValueError(
                    "custom packets must use key=value pairs separated "
                    "by semicolons"
                )

            key, raw_value = token.split("=", 1)
            key = key.strip()
            raw_value = raw_value.strip()

            if not key:
                raise ValueError("packet contains an empty field name")

            result[key] = UdpStreamReceiver._coerce_value(raw_value)

        if not result:
            raise ValueError("no fields found in packet")

        return result

    @staticmethod
    def _coerce_value(value: str) -> Any:
        lowered = value.lower()

        if lowered in {"true", "false"}:
            return lowered == "true"

        if lowered in {"null", "none"}:
            return None

        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value