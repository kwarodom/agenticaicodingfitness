# Fixtures

Synthetic data used only when the runner is in `mock` mode (no `openshell` binary on the host).
The mock world lives in `runner/app/mock.py`; recorded real-gateway outputs go here as `*.json`
once captured on the DGX Spark (M1), e.g. `openshell-sandbox-list.json`, `openshell-inference-get.json`.

Rules (spec 4.5): mock output is watermarked "MOCK DATA" in the UI, `REEF_MODE=mock` is refused when
a real gateway is present, and no fixture may be presented as live state.
