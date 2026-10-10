PY ?= ../../.venv/bin/python
.PHONY: dev seed test e2e env-check
seed:            ## create alto_mini.db with 14 days of data (incl. the known gaps)
	$(PY) -m app.data
dev:             ## FastAPI + static UI on :8127
	$(PY) -m uvicorn app.main:app --port 8127 --reload
test:            ## unit + API tests
	$(PY) -m pytest -q tests
e2e:             ## Playwright smoke (needs node + @playwright/test and `make dev` running)
	npx playwright test e2e --reporter=line
env-check:
	bash scripts/env_check.sh
