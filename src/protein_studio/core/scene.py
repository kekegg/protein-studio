"""Scene model — a Blender-like collection of objects.

Each :class:`SceneObject` wraps a :class:`Molecule` together with its
current representation, visibility, and color scheme. The
:class:`Scene` is the single source of truth for the viewport and
exposes ``selected``, ``active``, and change-notification hooks so the
UI can react.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .colors import ColorScheme, SCHEME_PRESETS
from .molecule import Molecule


@dataclass
class SceneObject:
    """A scene item = molecule + how it is shown.

    Attributes
    ----------
    name : str
        Display name in the outliner (defaults to molecule name).
    molecule : Molecule
        The wrapped structure.
    representation : str
        One of ``representations.REPRESENTATIONS`` (e.g. "cartoon").
    color_scheme : ColorScheme
        Active color scheme.
    visible : bool
        Whether the object is rendered.
    position, rotation, scale : array-like
        Object-level transform. Mirrors Blender's object mode transform.
    """

    name: str
    molecule: Molecule
    representation: str = "cartoon"
    color_scheme: ColorScheme = field(default_factory=lambda: SCHEME_PRESETS["secondary"])
    visible: bool = True
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale:    np.ndarray = field(default_factory=lambda: np.ones(3))

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.molecule.name

    def transform_matrix(self) -> np.ndarray:
        """Compose a 4x4 transform from position/rotation/scale.

        Rotation is interpreted as XYZ Euler angles in radians.
        """
        rx, ry, rz = self.rotation
        sx, sy, sz = self.scale
        # Rotation matrices
        Rx = np.array([
            [1, 0,          0,         0],
            [0, np.cos(rx), -np.sin(rx), 0],
            [0, np.sin(rx),  np.cos(rx), 0],
            [0, 0, 0, 1],
        ])
        Ry = np.array([
            [ np.cos(ry), 0, np.sin(ry), 0],
            [ 0,          1, 0,          0],
            [-np.sin(ry), 0, np.cos(ry), 0],
            [ 0, 0, 0, 1],
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0, 0],
            [np.sin(rz),  np.cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        R = Rx @ Ry @ Rz
        S = np.diag([sx, sy, sz, 1.0])
        T = np.eye(4)
        T[:3, 3] = self.position
        return T @ R @ S

    def apply_transform_to_coords(self, coords: np.ndarray) -> np.ndarray:
        if coords.size == 0:
            return coords
        M = self.transform_matrix()
        ones = np.ones((len(coords), 1))
        homo = np.hstack([coords, ones])
        return (M @ homo.T).T[:, :3]


class Scene:
    """A collection of scene objects, plus selection state.

    The scene emits ``changed`` events through ``on_change`` callbacks so
    the UI can refresh incrementally.
    """

    def __init__(self) -> None:
        self.objects: List[SceneObject] = []
        self.active_name: Optional[str] = None
        self._listeners: List[Callable[[str, object], None]] = []

    # --------------------------------------------------- listener registration
    def on_change(self, cb: Callable[[str, object], None]) -> Callable[[], None]:
        """Register a change listener; returns an unsubscribe function."""
        self._listeners.append(cb)
        return lambda: self._listeners.remove(cb) if cb in self._listeners else None

    def _emit(self, kind: str, payload: object = None) -> None:
        for cb in list(self._listeners):
            try:
                cb(kind, payload)
            except Exception:
                pass

    # ----------------------------------------------------------- basic ops
    def add(self, obj: SceneObject) -> None:
        self.objects.append(obj)
        if self.active_name is None:
            self.active_name = obj.name
        self._emit("add", obj)

    def remove(self, name: str) -> None:
        self.objects = [o for o in self.objects if o.name != name]
        if self.active_name == name:
            self.active_name = self.objects[0].name if self.objects else None
        self._emit("remove", name)

    def clear(self) -> None:
        self.objects.clear()
        self.active_name = None
        self._emit("clear", None)

    def find(self, name: str) -> Optional[SceneObject]:
        for o in self.objects:
            if o.name == name:
                return o
        return None

    def set_active(self, name: str) -> None:
        if self.find(name) is not None:
            self.active_name = name
            self._emit("active", name)

    @property
    def active(self) -> Optional[SceneObject]:
        return self.find(self.active_name) if self.active_name else None

    # ----------------------------------------------------- mutation helpers
    def update_object(self, name: str, **fields) -> None:
        o = self.find(name)
        if not o:
            return
        for k, v in fields.items():
            if hasattr(o, k):
                setattr(o, k, v)
        self._emit("update", o)

    def notify(self, kind: str = "update") -> None:
        """Manually broadcast a change event (e.g. after a transform tweak)."""
        self._emit(kind, None)

    def n_objects(self) -> int:
        return len(self.objects)

    def n_atoms(self) -> int:
        return sum(o.molecule.n_atoms() for o in self.objects)