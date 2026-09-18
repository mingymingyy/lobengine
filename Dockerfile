FROM python:3.12-slim

WORKDIR /app

COPY web/requirements.txt ./web/requirements.txt
RUN pip install --no-cache-dir -r web/requirements.txt

COPY src ./src
COPY web ./web

# VENUE=okx|bybit, SYMBOL, DEPTH, HZ, PRICE_DECIMALS all overridable.
ENV PORT=8000 VENUE=okx SYMBOL=BTC-USDT
EXPOSE 8000
CMD ["python", "web/server.py"]
