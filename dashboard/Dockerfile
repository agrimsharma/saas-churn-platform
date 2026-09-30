FROM python:3.11-slim
RUN pip install --no-cache-dir "streamlit>=1.50.0" requests pandas
WORKDIR /app
COPY dashboard/app.py .
# backtest, example drafts etc. (compose mounts ./reports over this)
COPY reports/ /reports/
ENV REPORTS_DIR=/reports
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
