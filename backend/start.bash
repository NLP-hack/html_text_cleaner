#!/bin/bash
export BACKEND_PORT=${BACKEND_PORT}

uvicorn app.main:app --host 0.0.0.0 --port $BACKEND_PORT