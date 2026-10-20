#!/usr/bin/env python3
"""Mini-USD — a compact teaching model of OpenUSD's composition engine.

When `pip install usd-core` is present the demos use the real `pxr` modules and
write genuine .usda files. When it is absent they fall back to THIS file's `Usd`,
`UsdGeom`, `Sdf` namespaces, which mimic the same API shape (Stage.CreateNew,
DefinePrim, sublayers, references, payloads, one variant set, instancing,
strongest-opinion-wins) just well enough for the four demos to print the same
results. Honesty note: this is a teaching model — a couple hundred lines cannot
be Pixar's LIVRPS composition engine. Trust pxr, not this, for real work.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace


class _Spec:
    """One prim's opinions in ONE layer — USD's unit of scene description."""
    def __init__(self, typeName: str = ""):
        self.typeName, self.attrs, self.rels, self.meta = typeName, {}, {}, {}
        self.children: dict[str, _Spec] = {}
        self.arcs: list[tuple[str, str, str]] = []      # (kind, assetPath, primPath)
        self.vsets: dict[str, dict] = {}                # name → {"sel": str, "variants": {v: _Spec}}


class Layer:
    _reg: dict[str, "Layer"] = {}

    def __init__(self, ident: str):
        self.identifier, self.root, self.subLayerPaths = str(ident), _Spec(), []
        Layer._reg[self.identifier] = self

    @staticmethod
    def FindOrOpen(ident) -> "Layer | None":
        ident = str(ident)
        if ident in Layer._reg:
            return Layer._reg[ident]
        for k, v in Layer._reg.items():                 # resolve relative paths ("chair.usda")
            if Path(k).name == Path(ident).name:
                return v
        return None

    def _spec(self, path, create=False, typeName="") -> "_Spec | None":
        node = self.root
        for part in [p for p in str(path).split("/") if p]:
            if part not in node.children:
                if not create:
                    return None
                node.children[part] = _Spec()
            node = node.children[part]
        if typeName and not node.typeName:
            node.typeName = typeName
        return node

    def _ser(self, spec: _Spec, name: str, ind: str) -> list[str]:
        head = f'{ind}def {spec.typeName} "{name}"' if spec.typeName else f'{ind}over "{name}"'
        tags = [f'kind = "{spec.meta["kind"]}"'] if spec.meta.get("kind") else []
        tags += ["instanceable = true"] if spec.meta.get("instanceable") else []
        tags += [f"prepend {k} = @{a}@<{t}>" for k, a, t in spec.arcs]
        lines = [head + (" (" + ", ".join(tags) + ")" if tags else ""), ind + "{"]
        lines += [f"{ind}    {a} = {v!r}" for a, v in spec.attrs.items()]
        lines += [f"{ind}    rel {r} = " + ", ".join(f"<{t}>" for t in tg) for r, tg in spec.rels.items()]
        for cn, cs in spec.children.items():
            lines += self._ser(cs, cn, ind + "    ")
        return lines + [ind + "}"]

    def Export(self, path) -> bool:
        hdr = ["#usda 1.0  (mini-USD teaching serializer — `pip install usd-core` for the real thing)"]
        if self.root.meta:
            hdr += ["("] + [f"    {k} = {v!r}" for k, v in self.root.meta.items()] + [")"]
        body: list[str] = []
        for n, s in self.root.children.items():
            body += self._ser(s, n, "")
        Path(path).write_text("\n".join(hdr + [""] + body) + "\n")
        return True

    def Save(self) -> bool:
        return self.Export(self.identifier)


class Stage:
    LoadNone, LoadAll = "none", "all"

    def __init__(self, root: Layer, load="all"):
        self.rootLayer, self._load, self._loaded = root, load, set()
        self.sessionLayer = Layer(f"anon-session-{id(self)}")
        self._edit, self._vtgt = root, None             # edit target · active variant context

    @staticmethod
    def CreateNew(ident) -> "Stage":
        return Stage(Layer(str(ident)))

    @staticmethod
    def Open(ident, load="all") -> "Stage":
        return Stage(ident if isinstance(ident, Layer) else Layer.FindOrOpen(ident), load)

    def GetRootLayer(self): return self.rootLayer
    def GetSessionLayer(self): return self.sessionLayer
    def SetEditTarget(self, layer): self._edit = layer
    def Load(self, path): self._loaded.add(str(path))

    def _stack(self) -> list[Layer]:                    # session → root → sublayers (strong → weak)
        out = [self.sessionLayer]
        def add(lyr):
            out.append(lyr)
            for sp in lyr.subLayerPaths:
                sub = Layer.FindOrOpen(sp)
                if sub:
                    add(sub)
        add(self.rootLayer)
        return out

    def _loaded_ok(self, path) -> bool:
        return self._load != "none" or str(path) in self._loaded

    def _contribs(self, path: str) -> list[_Spec]:      # every spec with opinions, strongest first
        path = str(path)
        if path in ("", "/"):
            base = [lyr.root for lyr in self._stack()]
        else:
            par, _, name = path.rpartition("/")
            base = []
            for ps in self._contribs(par or "/"):
                if name in ps.children:
                    base.append(ps.children[name])
                for vs in ps.vsets.values():            # children brought in by the selected variant
                    v = vs["variants"].get(vs.get("sel", ""))
                    if v and name in v.children:
                        base.append(v.children[name])
        out = list(base)
        for s in base:                                  # reference/payload arcs graft a target prim in
            for kind, asset, tgt in s.arcs:
                if kind == "payload" and not self._loaded_ok(path):
                    continue
                tl = Layer.FindOrOpen(asset)
                ts = tl._spec(tgt) if tl else None
                if ts:
                    out.append(ts)
        return out

    def _child_names(self, path) -> list[str]:
        names: list[str] = []
        for s in self._contribs(path):
            pools = [s.children] + [vs["variants"][vs["sel"]].children
                                    for vs in s.vsets.values() if vs.get("sel") in vs["variants"]]
            for pool in pools:
                names += [n for n in pool if n not in names]
        return names

    def DefinePrim(self, path, typeName="") -> "Prim":
        path = str(path)
        if self._vtgt and path.startswith(self._vtgt[0] + "/"):    # inside a variant edit context
            node = self._vtgt[1]
            for part in [p for p in path[len(self._vtgt[0]):].split("/") if p]:
                node = node.children.setdefault(part, _Spec())
            if typeName and not node.typeName:
                node.typeName = typeName
        else:
            self._edit._spec(path, create=True, typeName=typeName)
        return Prim(self, path)

    def OverridePrim(self, path): return self.DefinePrim(path)
    def GetPrimAtPath(self, path): return Prim(self, str(path))

    def Traverse(self):                                 # like pxr: skips unloaded payloads + instance insides
        def walk(path):
            for n in self._child_names(path):
                pr = Prim(self, (path if path != "/" else "") + "/" + n)
                if pr._unloaded():
                    continue
                yield pr
                if not pr.IsInstanceable():
                    yield from walk(pr._p)
        yield from walk("/")

    def GetPrototypes(self):                            # one prototype per unique instanced asset
        protos: dict = {}
        def walk(path):
            for n in self._child_names(path):
                pr = Prim(self, (path if path != "/" else "") + "/" + n)
                if pr._unloaded():
                    continue
                if pr.IsInstanceable():
                    protos.update({(a, t): 1 for _, a, t in [x for s in pr._specs() for x in s.arcs]})
                walk(pr._p)
        walk("/")
        return [SimpleNamespace(path=f"/__Prototype_{i + 1}") for i in range(len(protos))]

    def Export(self, path) -> bool:                     # like pxr Stage.Export: write the FLATTENED scene
        flat = Layer(str(path))
        flat.root.meta = dict(self.rootLayer.root.meta)
        def bake(src_path, dst: _Spec):
            for n in self._child_names(src_path):
                pr = Prim(self, (src_path if src_path != "/" else "") + "/" + n)
                if pr._unloaded():
                    continue
                spec = dst.children.setdefault(n, _Spec(pr.GetTypeName()))
                spec.meta = {k: v for s in reversed(pr._specs()) for k, v in s.meta.items()}
                for s in reversed(pr._specs()):
                    spec.attrs.update(s.attrs), spec.rels.update(s.rels)
                bake(pr._p, spec)
        bake("/", flat.root)
        return flat.Export(path)


class Prim:
    def __init__(self, stage, path): self._st, self._p = stage, str(path)
    def _specs(self): return self._st._contribs(self._p)
    def GetPath(self): return self._p
    def GetName(self): return self._p.rsplit("/", 1)[-1]
    def GetTypeName(self): return next((s.typeName for s in self._specs() if s.typeName), "")
    def IsValid(self): return bool(self._specs())
    __bool__ = IsValid

    def _unloaded(self):
        return any(k == "payload" for s in self._specs() for k, _, _ in s.arcs) \
            and not self._st._loaded_ok(self._p)

    def _edit_spec(self) -> _Spec:
        st = self._st
        if st._vtgt and self._p.startswith(st._vtgt[0] + "/"):
            node = st._vtgt[1]
            for part in [p for p in self._p[len(st._vtgt[0]):].split("/") if p]:
                node = node.children.setdefault(part, _Spec())
            return node
        return st._edit._spec(self._p, create=True)

    def GetChildren(self):
        return [Prim(self._st, (self._p if self._p != "/" else "") + "/" + n)
                for n in self._st._child_names(self._p)]

    def CreateAttribute(self, name, typeName=None, custom=True):
        self._edit_spec()
        return Attribute(self, name)

    def GetAttribute(self, name): return Attribute(self, name)

    def CreateRelationship(self, name, custom=True):
        self._edit_spec().rels.setdefault(name, [])
        return Relationship(self, name)

    def GetReferences(self):
        s = self._edit_spec()
        return SimpleNamespace(AddReference=lambda asset, prim="/":
                               (s.arcs.append(("references", str(asset), str(prim))), True)[1])

    def GetPayloads(self):
        s = self._edit_spec()
        return SimpleNamespace(AddPayload=lambda asset, prim="/":
                               (s.arcs.append(("payload", str(asset), str(prim))), True)[1])

    def SetInstanceable(self, v):
        self._edit_spec().meta["instanceable"] = bool(v)
        return True

    def IsInstanceable(self): return any(s.meta.get("instanceable") for s in self._specs())
    def GetVariantSets(self): return VariantSets(self)


class Attribute:
    def __init__(self, prim, name): self._pr, self._n = prim, name
    def Set(self, v):
        self._pr._edit_spec().attrs[self._n] = v
        return True
    def Get(self):
        return next((s.attrs[self._n] for s in self._pr._specs() if self._n in s.attrs), None)


class Relationship:
    def __init__(self, prim, name): self._pr, self._n = prim, name
    def AddTarget(self, path):
        self._pr._edit_spec().rels.setdefault(self._n, []).append(str(path))
        return True
    def GetTargets(self):
        return next((list(s.rels[self._n]) for s in self._pr._specs() if self._n in s.rels), [])


class VariantSets:
    def __init__(self, prim): self._pr = prim
    def AddVariantSet(self, name):
        return VariantSet(self._pr, self._pr._edit_spec().vsets.setdefault(name, {"sel": "", "variants": {}}))
    def GetVariantSet(self, name):
        d = next((s.vsets[name] for s in self._pr._specs() if name in s.vsets), None)
        return VariantSet(self._pr, d) if d else self.AddVariantSet(name)


class VariantSet:
    def __init__(self, prim, d): self._pr, self._d = prim, d
    def AddVariant(self, v): return self._d["variants"].setdefault(v, _Spec()) is not None
    def SetVariantSelection(self, v):
        self._d["sel"] = v
        return True
    def GetVariantSelection(self): return self._d.get("sel", "")
    def GetVariantEditContext(self):
        prim, d = self._pr, self._d
        class _Ctx:
            def __enter__(self2): prim._st._vtgt = (prim._p, d["variants"][d["sel"]])
            def __exit__(self2, *a): prim._st._vtgt = None
        return _Ctx()


class ModelAPI:
    def __init__(self, prim): self._pr = prim
    def SetKind(self, k):
        self._pr._edit_spec().meta["kind"] = str(k)
        return True
    def GetKind(self):
        return next((s.meta["kind"] for s in self._pr._specs() if "kind" in s.meta), "")


class _Schema:                                          # UsdGeom.Xform/Scope/Mesh stand-ins
    typeName = ""
    def __init__(self, prim): self._prim = prim
    @classmethod
    def Define(cls, stage, path): return cls(stage.DefinePrim(path, cls.typeName))
    def GetPrim(self): return self._prim
    def CreateDisplayColorAttr(self, value=None):
        a = self._prim.CreateAttribute("primvars:displayColor")
        if value is not None:
            a.Set([tuple(v) for v in value])
        return a
    def CreatePurposeAttr(self, value=None):
        a = self._prim.CreateAttribute("purpose")
        if value is not None:
            a.Set(str(value))
        return a


class _Xform(_Schema): typeName = "Xform"
class _Scope(_Schema): typeName = "Scope"
class _Mesh(_Schema): typeName = "Mesh"


def _set_up(stage, ax): stage.rootLayer.root.meta["upAxis"] = str(ax); return True
def _get_up(stage): return stage.rootLayer.root.meta.get("upAxis", "Y")          # USD default: Y
def _set_mpu(stage, v): stage.rootLayer.root.meta["metersPerUnit"] = float(v); return True
def _get_mpu(stage): return stage.rootLayer.root.meta.get("metersPerUnit", 0.01)  # USD default: cm!

# the three namespaces the demos import when `from pxr import …` fails
Usd = SimpleNamespace(Stage=Stage, ModelAPI=ModelAPI)
UsdGeom = SimpleNamespace(
    Xform=_Xform, Scope=_Scope, Mesh=_Mesh, Imageable=lambda prim: _Schema(prim),
    Tokens=SimpleNamespace(z="Z", y="Y", proxy="proxy", render="render", guide="guide"),
    SetStageUpAxis=_set_up, GetStageUpAxis=_get_up,
    SetStageMetersPerUnit=_set_mpu, GetStageMetersPerUnit=_get_mpu)
Sdf = SimpleNamespace(Layer=Layer, ValueTypeNames=SimpleNamespace(
    String="string", Float="float", Color3fArray="color3f[]", Token="token"))


def usd_mode(real: bool) -> None:
    """The module's own MODE line — REAL/SIM here means usd-core, not the LLM endpoint."""
    if real:
        print("▣ MODE: REAL — `usd-core` (pxr) is installed; writing genuine .usda stages to .sandbox/.")
    else:
        print("▣ MODE: SIM — `usd-core` not installed; using sim.py's mini-USD engine "
              "(same API shape, honest teaching model).")
    print("  # run it for real — OpenUSD straight from pip (no GPU, no Omniverse, ~30 MB):")
    print("  $ pip install usd-core")
    print()


# ── Ch 4 · the instancing lever (illustrative, order-of-magnitude honest) ────
# (setup, prims on the stage, triangles held in memory, est. memory)
INSTANCING = [
    ("4,000 chair copies (naive)",    24_000, 49_600_000, "~4.8 GB"),
    ("4,000 instances, 1 prototype",   4_006,     12_400, "~1.5 MB + 4,000 transforms"),
]

# ── standard SIM exports (model list · tok/s · canned streaming answer) ──────
_MODELS = ["nemotron-3-nano:30b-a3b", "gemma4:12b"]
_TOK = {"nemotron-3-nano:30b-a3b": 54.0, "gemma4:12b": 41.0}


def installed_models() -> list[str]:
    return list(_MODELS)


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40.0))


_CANNED = ("[simulated] A USD stage is not a file — it is the COMPOSED view of a layer "
           "stack. Each discipline keeps its own layer (site, architecture, MEP, furniture, "
           "sensors); a stronger layer overrides a weaker one without destroying its opinion; "
           "references and payloads pull whole assets in on demand; instancing makes the "
           "4,000th chair nearly free. That non-destructive composition is why OpenUSD is "
           "the scene language of the digital twin.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
