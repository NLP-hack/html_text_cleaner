#!/bin/bash
export FRONTEND_PORT=${FRONTEND_PORT}

streamlit run app/main.py --server.port $FRONTEND_PORT