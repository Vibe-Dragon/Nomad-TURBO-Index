"""Where indexing and OCR run.

The person picks a device. The program picks the OCR engine for that device.
Indexing on a card is the CUDA embed server, so only an NVIDIA card is offered
for it. OCR can use an NVIDIA, AMD, or Intel card when a runtime for it is
installed. A saved card that is gone resolves to the CPU until it returns.
"""

from __future__ import annotations

import ctypes
import re
import subprocess
import threading
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ComputeDevice:
    id: str
    kind: str
    name: str
    index: int | None
    total_mib: int
    used_mib: int
    compute_major: int
    compute_minor: int
    pci: str


@dataclass(frozen=True)
class Backend:
    """The engine that will actually read pages. `reason` is set when a card
    was asked for and the CPU reader is used instead.
    """

    name: str
    sentence: str
    reason: str = ""


@dataclass(frozen=True)
class Assignment:
    index: ComputeDevice
    ocr: ComputeDevice
    backend: Backend
    share: bool

    @property
    def uses_gpu_ocr(self) -> bool:
        return self.backend.name != "tesseract"


def normalize_device_choice(value: object) -> str:
    text = str(value or "auto").strip()
    if text in ("", "auto"):
        return "auto"
    if text == "cpu":
        return "cpu"
    kind, sep, ident = text.partition(":")
    if sep and kind in ("nvidia", "amd", "intel") and ident:
        return text
    return "auto"


def cpu_device() -> ComputeDevice:
    return ComputeDevice("cpu", "cpu", "CPU", None, 0, 0, 0, 0, "")


def _split_cap(value: str) -> tuple[int, int]:
    major, sep, minor = str(value or "").partition(".")
    try:
        return int(major), int(minor) if sep else 0
    except ValueError:
        return 0, 0


def _nvidia_rows() -> list[ComputeDevice]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.used,compute_cap,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            index = int(parts[0])
            total = int(float(parts[3]))
            used = int(float(parts[4]))
        except ValueError:
            continue
        major, minor = _split_cap(parts[5])
        uuid = parts[1] or f"index-{index}"
        rows.append(
            ComputeDevice(
                f"nvidia:{uuid}",
                "nvidia",
                present_gpu_name("nvidia", parts[2] or "", "discrete"),
                index,
                total,
                used,
                major,
                minor,
                parts[6].lower(),
            )
        )
    return rows


_RAW_DEVICE = re.compile(r"^device\s+[0-9a-f]{4}$", re.I)
_CPU_NAME = re.compile(r"\b(processor|ryzen|core\s+i\d|xeon|epyc|threadripper)\b", re.I)
_GPU_WORD = re.compile(r"\b(radeon|geforce|quadro|rtx|gtx|arc|uhd|iris|graphics|gpu)\b", re.I)
_DRIVER_TAG = re.compile(r"\s*\((?:RADV|LLVM|Mesa|llvmpipe)[^)]*\)", re.I)
_VULKAN_LOCK = threading.Lock()
_VULKAN_CARDS: list[dict] | None = None
_VK_API_1_1 = (1 << 22) | (1 << 12)
_VK_PROPS2 = 1000059001
_VK_PCI = 1000212000
_VK_TYPES = {0: "other", 1: "integrated", 2: "discrete", 3: "virtual", 4: "cpu"}


def present_gpu_name(vendor: str, reported: str, device_type: str = "") -> str:
    """A label that stays meaningful on any machine.

    A product name from the driver is kept. A raw PCI id, or a processor
    name standing in for an integrated chip, becomes that vendor's GPU name.
    """
    text = _DRIVER_TAG.sub(" ", reported or "")
    text = re.sub(r"\s+", " ", text).strip()
    if _RAW_DEVICE.match(text):
        text = ""
    if text and _CPU_NAME.search(text) and not _GPU_WORD.search(text):
        text = ""
    if text:
        return text
    if vendor == "amd" and device_type == "integrated":
        return "AMD Radeon Graphics"
    if vendor == "intel" and device_type == "integrated":
        return "Intel Graphics"
    label = {"amd": "AMD", "intel": "Intel", "nvidia": "NVIDIA"}.get(vendor, "")
    return f"{label} GPU" if label else "GPU"


def _canonical_pci(pci: str) -> str:
    parts = str(pci or "").lower().split(":")
    if len(parts) < 3:
        return str(pci or "").lower()
    try:
        domain = int(parts[0], 16)
        bus = int(parts[1], 16)
    except ValueError:
        return str(pci or "").lower()
    return f"{domain:04x}:{bus:02x}:{parts[2]}"


def _short_pci(pci: str) -> str:
    canonical = _canonical_pci(pci)
    parts = canonical.split(":")
    if len(parts) < 3:
        return canonical
    if parts[0] == "0000":
        return f"{parts[1]}:{parts[2]}"
    return canonical


def _unique_labels(devices: list[ComputeDevice]) -> list[ComputeDevice]:
    """Two cards with one product name keep the bus address."""
    counts: dict[str, int] = {}
    for device in devices:
        counts[device.name] = counts.get(device.name, 0) + 1
    labeled = []
    for device in devices:
        if counts[device.name] > 1 and device.pci:
            labeled.append(replace(device, name=f"{device.name} ({_short_pci(device.pci)})"))
        else:
            labeled.append(device)
    return labeled


class _VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class _VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(_VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


class _VkPciBus(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("pciDomain", ctypes.c_uint32),
        ("pciBus", ctypes.c_uint32),
        ("pciDevice", ctypes.c_uint32),
        ("pciFunction", ctypes.c_uint32),
    ]


class _VkProperties(ctypes.Structure):
    _fields_ = [
        ("apiVersion", ctypes.c_uint32),
        ("driverVersion", ctypes.c_uint32),
        ("vendorID", ctypes.c_uint32),
        ("deviceID", ctypes.c_uint32),
        ("deviceType", ctypes.c_uint32),
        ("deviceName", ctypes.c_char * 256),
        ("pipelineCacheUUID", ctypes.c_uint8 * 16),
        ("_pad", ctypes.c_uint32),
        ("_rest", ctypes.c_uint8 * 528),
    ]


class _VkProperties2(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("properties", _VkProperties),
    ]


def _load_vulkan():
    for name in ("libvulkan.so.1", "libvulkan.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _vk_proc(lib, instance, name, prototype):
    getter = lib.vkGetInstanceProcAddr
    getter.restype = ctypes.c_void_p
    getter.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    address = getter(instance, name.encode())
    if not address:
        return None
    return prototype(address)


def _query_vulkan() -> list[dict]:
    """Product name, form, and PCI address for each graphics device."""
    lib = _load_vulkan()
    if lib is None:
        return []
    create = _vk_proc(
        lib,
        None,
        "vkCreateInstance",
        ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(_VkInstanceCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
    )
    if create is None:
        return []
    app = _VkApplicationInfo(sType=0, apiVersion=_VK_API_1_1)
    info = _VkInstanceCreateInfo(sType=1, pApplicationInfo=ctypes.pointer(app))
    instance = ctypes.c_void_p()
    if create(ctypes.byref(info), None, ctypes.byref(instance)) != 0 or not instance:
        return []
    try:
        enumerate_devices = _vk_proc(
            lib,
            instance,
            "vkEnumeratePhysicalDevices",
            ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)),
        )
        properties_of = _vk_proc(
            lib,
            instance,
            "vkGetPhysicalDeviceProperties2",
            ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(_VkProperties2)),
        )
        if enumerate_devices is None or properties_of is None:
            return []
        count = ctypes.c_uint32(0)
        if enumerate_devices(instance, ctypes.byref(count), None) != 0 or not count.value:
            return []
        handles = (ctypes.c_void_p * count.value)()
        if enumerate_devices(instance, ctypes.byref(count), ctypes.cast(handles, ctypes.POINTER(ctypes.c_void_p))) != 0:
            return []
        found = []
        for index in range(count.value):
            pci = _VkPciBus(sType=_VK_PCI, pciDomain=0xFFFFFFFF, pciBus=0xFFFFFFFF, pciDevice=0xFFFFFFFF, pciFunction=0xFFFFFFFF)
            props = _VkProperties2(sType=_VK_PROPS2, pNext=ctypes.addressof(pci))
            properties_of(handles[index], ctypes.byref(props))
            kind = _VK_TYPES.get(int(props.properties.deviceType), "other")
            if kind == "cpu":
                continue
            bus = ""
            if pci.pciDomain != 0xFFFFFFFF:
                bus = f"{int(pci.pciDomain):04x}:{int(pci.pciBus):02x}:{int(pci.pciDevice):02x}.{int(pci.pciFunction)}"
            name = props.properties.deviceName.split(b"\0", 1)[0].decode("utf-8", "replace")
            found.append({
                "pci": bus,
                "vendor": int(props.properties.vendorID),
                "device": int(props.properties.deviceID),
                "name": name,
                "type": kind,
            })
        return found
    finally:
        destroy = _vk_proc(
            lib,
            instance,
            "vkDestroyInstance",
            ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p),
        )
        if destroy is not None:
            destroy(instance, None)


def _vulkan_cards() -> list[dict]:
    global _VULKAN_CARDS
    if _VULKAN_CARDS is not None:
        return _VULKAN_CARDS
    with _VULKAN_LOCK:
        if _VULKAN_CARDS is None:
            try:
                _VULKAN_CARDS = _query_vulkan()
            except Exception:
                _VULKAN_CARDS = []
    return _VULKAN_CARDS


def _reported_gpu(pci: str, vendor_text: str, device_text: str) -> dict | None:
    cards = _vulkan_cards()
    canonical = _canonical_pci(pci)
    for card in cards:
        if card["pci"] and card["pci"] == canonical:
            return card
    try:
        key = (int(vendor_text, 16), int(device_text, 16))
    except ValueError:
        return None
    matches = [card for card in cards if (card["vendor"], card["device"]) == key]
    if len(matches) == 1:
        return matches[0]
    return None


def _pci_name(slot: str) -> str:
    if not slot:
        return ""
    try:
        out = subprocess.check_output(
            ["lspci", "-s", slot, "-mm"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    for line in out.splitlines():
        bits = [part.strip().strip('"') for part in line.split('"') if part.strip()]
        # lspci -mm: slot, class, vendor, device.
        if len(bits) >= 4:
            return bits[3]
    return ""


def _drm_rows(seen_pci: set[str]) -> list[ComputeDevice]:
    from pathlib import Path

    root = Path("/sys/class/drm")
    if not root.is_dir():
        return []
    vendors = {"0x1002": "amd", "0x8086": "intel", "0x10de": "nvidia"}
    counts = {"amd": 0, "intel": 0, "nvidia": 0}
    rows = []
    cards = [path for path in root.iterdir() if path.name.startswith("card") and path.name[4:].isdigit()]
    for card in sorted(cards, key=lambda path: int(path.name[4:])):
        device = card / "device"
        try:
            vendor = (device / "vendor").read_text().strip().lower()
            device_id = (device / "device").read_text().strip().lower()
            pci = device.resolve().name.lower()
        except OSError:
            continue
        kind = vendors.get(vendor)
        if kind is None or _canonical_pci(pci) in seen_pci:
            continue
        # nvidia-smi already named these. A drm row would be a second copy.
        if kind == "nvidia" and seen_pci:
            continue
        reported = _reported_gpu(pci, vendor, device_id)
        if reported:
            name = present_gpu_name(kind, reported["name"], reported["type"])
        else:
            name = present_gpu_name(kind, _pci_name(pci), "")
        index = counts[kind]
        counts[kind] += 1
        rows.append(ComputeDevice(f"{kind}:{pci}", kind, name, index, 0, 0, 0, 0, pci))
        seen_pci.add(pci)
    return rows


def list_devices() -> list[ComputeDevice]:
    """CPU, then NVIDIA cards from nvidia-smi, then other cards from sysfs."""
    nvidia = _nvidia_rows()
    seen = {_canonical_pci(device.pci) for device in nvidia if device.pci}
    return _unique_labels([cpu_device(), *nvidia, *_drm_rows(seen)])


def reader_label(name: str) -> str:
    """The short engine name used in the log."""
    text = str(name or "")
    if text == "easyocr":
        return "EasyOCR"
    if text.startswith("paddle"):
        return "PaddleOCR"
    if text == "openvino":
        return "OpenVINO"
    if text.startswith("onnx"):
        return "ONNX Runtime"
    return "Tesseract"


def saved_choice(session: object | None, install: object, field: str) -> str:
    getter = getattr(session, field, None) if session is not None else None
    if callable(getter):
        return normalize_device_choice(getter())
    return normalize_device_choice(getattr(install, field, "auto"))


def _cpu(devices: list[ComputeDevice]) -> ComputeDevice:
    for device in devices:
        if device.kind == "cpu":
            return device
    return cpu_device()


def pick_index(devices: list[ComputeDevice], choice: str) -> ComputeDevice:
    """The embed server uses CUDA. An AMD or Intel card stays on the CPU."""
    cpu = _cpu(devices)
    nvidia = [device for device in devices if device.kind == "nvidia"]
    choice = normalize_device_choice(choice)
    if choice == "cpu" or not nvidia:
        return cpu
    if choice.startswith("nvidia:"):
        for device in nvidia:
            if device.id == choice:
                return device
        return cpu
    return max(nvidia, key=lambda device: device.total_mib - device.used_mib)


def select_backend(device: ComputeDevice, info: dict | None) -> Backend:
    """Pick an engine from the card. The caller never names one."""
    info = info or {}
    if device.kind == "cpu":
        return Backend("tesseract", "The reader is Tesseract.")
    major = int(device.compute_major or info.get("ran_major") or 0)
    works = bool(info.get("cuda_works"))
    third = bool(info.get("sm120_wheel"))
    if device.kind == "nvidia":
        blackwell = major >= 12 or major == 10
        if works and third and (blackwell or major == 0):
            return Backend(
                "paddle-sm120",
                "The reader is PaddleOCR, using the third-party build for compute capability 12.0.",
            )
        if works and not third and blackwell:
            return Backend(
                "paddle-blackwell",
                "The reader is PaddleOCR, using the official build for this GPU generation.",
            )
        if works and not third and not blackwell:
            return Backend("paddle-cuda", "The reader is PaddleOCR.")
        if works and third and not blackwell:
            return Backend(
                "tesseract",
                "The reader is Tesseract.",
                "The installed GPU build is only for compute capability 12.0, so OCR stays on the CPU.",
            )
        if info.get("easyocr") and info.get("torch_works"):
            return Backend("easyocr", "The reader is EasyOCR.")
        if info.get("onnx_cuda") and info.get("rapid_onnx"):
            return Backend(
                "onnx-cuda",
                "The reader is PaddleOCR's models on this NVIDIA GPU through ONNX Runtime.",
            )
        detail = str(info.get("cuda_error") or "").strip()
        reason = detail or f"No GPU reader is installed for {device.name}, so OCR stays on the CPU."
        return Backend("tesseract", "The reader is Tesseract.", reason)
    if device.kind == "amd":
        if info.get("rocm_works"):
            return Backend("paddle-rocm", "The reader is PaddleOCR.")
        if info.get("rapid_onnx") and (info.get("onnx_migraphx") or info.get("onnx_rocm")):
            return Backend(
                "onnx-amd",
                "The reader is PaddleOCR's models on this AMD GPU through ONNX Runtime.",
            )
        return Backend(
            "tesseract",
            "The reader is Tesseract.",
            f"No AMD GPU reader is installed for {device.name}, so OCR stays on the CPU.",
        )
    if device.kind == "intel":
        if info.get("openvino_gpu") and info.get("rapid_openvino"):
            return Backend("openvino", "The reader is PaddleOCR's models on this Intel GPU through OpenVINO.")
        if info.get("onnx_openvino") and info.get("rapid_onnx"):
            return Backend(
                "onnx-openvino",
                "The reader is PaddleOCR's models on this Intel GPU through ONNX Runtime.",
            )
        return Backend(
            "tesseract",
            "The reader is Tesseract.",
            f"No Intel GPU reader is installed for {device.name}, so OCR stays on the CPU.",
        )
    return Backend("tesseract", "The reader is Tesseract.")


def pick_ocr(devices: list[ComputeDevice], choice: str, index: ComputeDevice, probes: dict | None) -> ComputeDevice:
    """Auto uses another card only when that card has a GPU reader."""
    cpu = _cpu(devices)
    gpus = [device for device in devices if device.kind != "cpu"]
    choice = normalize_device_choice(choice)
    probes = probes or {}
    if choice == "cpu" or not gpus:
        return cpu
    if choice != "auto":
        for device in gpus:
            if device.id == choice:
                return device
        return cpu
    others = []
    for device in gpus:
        if device.id == index.id:
            continue
        if select_backend(device, probes.get(device.id)).name != "tesseract":
            others.append(device)
    if not others:
        return cpu
    return max(others, key=lambda device: device.total_mib - device.used_mib)


def resolve_assignment(
    devices: list[ComputeDevice],
    index_choice: str,
    ocr_choice: str,
    probes: dict | None = None,
) -> Assignment:
    probes = probes or {}
    index = pick_index(devices, index_choice)
    ocr = pick_ocr(devices, ocr_choice, index, probes)
    backend = select_backend(ocr, probes.get(ocr.id))
    on_cpu = backend.name == "tesseract"
    share = (on_cpu and index.kind == "cpu") or ((not on_cpu) and index.id == ocr.id)
    return Assignment(index, ocr, backend, share)


def assignment_sentence(assignment: Assignment, *, parallel: bool = False) -> str:
    index_where = "the CPU" if assignment.index.kind == "cpu" else assignment.index.name
    if assignment.backend.name == "tesseract":
        ocr_where = "the CPU"
    else:
        ocr_where = assignment.ocr.name
    parts = [
        f"Indexing uses {index_where}.",
        f"OCR uses {ocr_where}. {assignment.backend.sentence}",
    ]
    if assignment.backend.reason:
        parts.append(assignment.backend.reason)
    elif assignment.share:
        parts.append("OCR waits while that device is indexing.")
    elif assignment.index.kind == "cpu":
        parts.append("Page drawing gives cores back while the CPU embedder or Qdrant is busy.")
    elif assignment.backend.name == "tesseract":
        if parallel:
            parts.append(
                "Experimental. OCR reads the next PDF or picture while that GPU indexes "
                "the file it already has, and gives cores back while Qdrant is writing or optimizing."
            )
        else:
            parts.append(
                "OCR waits until that file is stored. "
                "One file is read or indexed at a time. "
                "Parallel OCR is experimental and stays off until it is turned on."
            )
    else:
        parts.append("OCR and indexing run at the same time.")
    return " ".join(parts)
