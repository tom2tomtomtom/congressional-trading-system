# Railway/Heroku Process File
web: python src/api/app.py
worker: python src/realtime/websocket_server.py
release: python database/setup.py && python scripts/populate_congressional_database.py