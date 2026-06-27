GUNICORN_PID_FILE := /home/jabi_ai/Project/app/logs/gunicorn.pid
GUNICORN_LOG_FILE := /home/jabi_ai/Project/app/logs/gunicorn.log
GUNICORN_BIN := /home/jabi_ai/Project/venv/bin/gunicorn
APP_MODULE := app:main_app
GUNICORN_CMD := $(GUNICORN_BIN) -k uvicorn.workers.UvicornWorker $(APP_MODULE) \
				-w 4 \
				-b 0.0.0.0:80 \
				--timeout 120 \
				--pid $(GUNICORN_PID_FILE) \
				--access-logfile $(GUNICORN_LOG_FILE) \
				--error-logfile $(GUNICORN_LOG_FILE)

.PHONY: start-gunicorn stop-gunicorn restart-gunicorn

start-gunicorn:
	mkdir -p $(dir $(GUNICORN_LOG_FILE))
	$(GUNICORN_CMD) > /dev/null 2>&1 &
	echo "Gunicorn 프로세스가 실행되었습니다."

stop-gunicorn:
	if [ -f "$(GUNICORN_PID_FILE)" ]; then \
		echo "Gunicorn 프로세스를 중지합니다..."; \
		PID=$$(cat "$(GUNICORN_PID_FILE)"); \
		kill "$$PID"; \
		while kill -0 "$$PID" 2>/dev/null; do sleep 1; done; \
		rm "$(GUNICORN_PID_FILE)"; \
		echo "Gunicorn 프로세스가 성공적으로 중지되었습니다."; \
	else \
		echo "Gunicorn 프로세스가 실행 중이지 않습니다."; \
	fi

restart-gunicorn: stop-gunicorn start-gunicorn