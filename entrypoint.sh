#!/bin/bash

ENV_FILE="/app/.last_env"

if [ -f "$ENV_FILE" ]; then
    LAST_ENV=$(cat "$ENV_FILE")
else
    LAST_ENV=""
fi

if [ -t 0 ]; then
    echo "Welcome to the simulation container!"
    echo "Available environments: dedalus-env (alias: d), fenics-env (alias: f)"
    read -p "Which environment would you like to use? [last: $LAST_ENV] " ENV_CHOICE
else
    ENV_CHOICE=""
fi

case "$ENV_CHOICE" in
    d) ENV_NAME="dedalus-env" ;;
    f) ENV_NAME="fenics-env" ;;
    "") ENV_NAME="$LAST_ENV" ;;
    *) ENV_NAME="$ENV_CHOICE" ;;
esac

if [ -z "$ENV_NAME" ]; then
    echo "No environment selected. Starting default bash."
    exec bash
fi

if conda info --envs | grep -xq -- "$ENV_NAME"; then
    printf '%s\n' "$ENV_NAME" > "$ENV_FILE"
    exec conda run -n "$ENV_NAME" bash
else
    echo "Environment '$ENV_NAME' not found. Starting default bash."
    exec bash
fi
