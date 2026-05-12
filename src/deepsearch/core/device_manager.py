"""Device detection and routing for hardware acceleration.

Detects available accelerators and maps model slots to devices.
Order of preference: CUDA > ROCm > Metal > Intel GPU > CPU.
"""
from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger(__name__)


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"
    INTEL_GPU = "intel_gpu"
    INTEL_NPU = "intel_npu"
    CLOUD = "cloud"


@dataclass
class DeviceInfo:
    device: Device
    name: str
    vram_gb: float = 0.0
    available: bool = True


def _has_cuda() -> bool:
    try:
        import torch  # type: ignore[import]
        return torch.cuda.is_available()
    except ImportError:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def _cuda_vram_gb() -> float:
    try:
        import torch  # type: ignore[import]
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    return 0.0


def _cuda_name() -> str:
    try:
        import torch  # type: ignore[import]
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).name
    except Exception:
        pass
    return "NVIDIA GPU"


def _has_rocm() -> bool:
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _has_metal() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _has_intel_gpu() -> bool:
    try:
        import openvino as ov  # type: ignore[import]
        core = ov.Core()
        return "GPU" in core.available_devices
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["xpu-smi", "discovery"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _has_intel_npu() -> bool:
    try:
        import openvino as ov  # type: ignore[import]
        core = ov.Core()
        return "NPU" in core.available_devices
    except Exception:
        return False


def _on_battery() -> bool:
    """Return True if the system is running on battery (not plugged in)."""
    try:
        import psutil  # type: ignore[import]
        bat = psutil.sensors_battery()
        if bat is None:
            return False  # desktop — no battery
        return not bat.power_plugged
    except Exception:
        return False


def _available_ram_gb() -> float:
    """Return available system RAM in GB."""
    try:
        import psutil  # type: ignore[import]
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 0.0


class DeviceManager:
    """Detects hardware at startup and provides routing recommendations."""

    def __init__(self) -> None:
        self._devices: list[DeviceInfo] = []
        self._primary: Device = Device.CPU
        self._detected = False

    def detect(self) -> None:
        self._devices = [DeviceInfo(Device.CPU, "CPU", 0.0)]

        if _has_cuda():
            vram = _cuda_vram_gb()
            name = _cuda_name()
            self._devices.append(DeviceInfo(Device.CUDA, name, vram))
            log.info("CUDA detected: %s (%.1f GB VRAM)", name, vram)

        elif _has_rocm():
            self._devices.append(DeviceInfo(Device.ROCM, "AMD GPU (ROCm)", 0.0))
            log.info("ROCm detected")

        elif _has_metal():
            self._devices.append(DeviceInfo(Device.METAL, "Apple Silicon (Metal)", 0.0))
            log.info("Apple Metal detected")

        elif _has_intel_gpu():
            self._devices.append(DeviceInfo(Device.INTEL_GPU, "Intel GPU", 0.0))
            if _has_intel_npu():
                self._devices.append(DeviceInfo(Device.INTEL_NPU, "Intel NPU", 0.0))
            log.info("Intel GPU detected")

        # Set primary device (highest priority non-CPU)
        non_cpu = [d for d in self._devices if d.device != Device.CPU]
        self._primary = non_cpu[0].device if non_cpu else Device.CPU
        self._detected = True
        log.info("Primary device: %s", self._primary.value)

    @property
    def primary_device(self) -> Device:
        if not self._detected:
            self.detect()
        return self._primary

    @property
    def available_devices(self) -> list[DeviceInfo]:
        if not self._detected:
            self.detect()
        return self._devices

    def n_gpu_layers_for(self, model_size_gb: float) -> int:
        """Return recommended n_gpu_layers for llama.cpp.

        -1 = all layers on GPU (full offload)
         0 = CPU only
        """
        if not self._detected:
            self.detect()

        if self._primary == Device.CPU:
            return 0

        if self._primary == Device.METAL:
            return -1  # Metal always fully offloads

        if self._primary == Device.CUDA:
            vram = next(
                (d.vram_gb for d in self._devices if d.device == Device.CUDA), 0.0
            )
            return -1 if vram >= model_size_gb + 1.0 else max(0, int((vram / model_size_gb) * 32))

        if self._primary in (Device.ROCM, Device.INTEL_GPU):
            return -1

        return 0

    def openvino_device_string(self) -> str:
        """Return the OpenVINO device string for the primary accelerator."""
        mapping = {
            Device.INTEL_GPU: "GPU",
            Device.INTEL_NPU: "NPU",
            Device.CPU: "CPU",
        }
        return mapping.get(self._primary, "CPU")

    def llama_cpp_device_flag(self) -> str:
        """Human-readable device name for logging."""
        return self._primary.value

    @property
    def on_battery(self) -> bool:
        """True when system is running on battery power."""
        return _on_battery()

    @property
    def available_ram_gb(self) -> float:
        """Available system RAM in GB."""
        return _available_ram_gb()

    def recommend_search_mode(self, requested_mode: str = "fast") -> str:
        """Return the recommended search mode accounting for power state.

        On battery: downgrades 'deep' / 'cloud' to 'fast' to save power.
        When GPU memory is low: downgrades to 'fast'.
        Otherwise: returns the requested mode unchanged.
        """
        if self.on_battery:
            log.info("On battery — downgrading search mode to 'fast'")
            return "fast"
        return requested_mode

    def gpu_layer_budget(self) -> int:
        """Return n_gpu_layers adjusted for current memory pressure.

        Returns 0 (CPU-only) when available RAM drops below 2 GB.
        """
        if not self._detected:
            self.detect()
        if self.available_ram_gb < 2.0:
            log.warning("Low RAM (%.1f GB) — forcing CPU-only inference", self.available_ram_gb)
            return 0
        return self.n_gpu_layers_for(4.4)  # default for ~4 GB main model


# Singleton
_device_manager: DeviceManager | None = None


def get_device_manager() -> DeviceManager:
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
        _device_manager.detect()
    return _device_manager
