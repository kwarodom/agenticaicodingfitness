.PHONY: dev test build serve runner frontend
runner:
	cd runner && uvicorn app.main:app --host 127.0.0.1 --port 4455 --reload
frontend:
	cd frontend && npm run dev -- --port 4454
dev:
	$(MAKE) -j2 runner frontend
build:
	cd frontend && npm run build
test:
	cd runner && python -m pytest -q
serve: build
	cd runner && REEF_TOKEN=$${REEF_TOKEN} uvicorn app.main:app --host 127.0.0.1 --port 4455 &
	cd frontend && npx serve -l 4454 dist
