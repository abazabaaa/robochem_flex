#!/bin/bash

INTERFACE=ethernet
HOST=""
PORT=""


while getopts i:h:p: flag
do
  case "${flag}" in
    i) INTERFACE=${OPTARG};;
    h) HOST=${OPTARG};;
    p) PORT=${OPTARG};;
  esac
done

COMMAND="python server_app/RamaBerry_listener.py --interface $INTERFACE"


if [ ! -z "$HOST" ]; then
  COMMAND="$COMMAND --host $HOST"
fi

if [ ! -z "$PORT" ]; then
  COMMAND="$COMMAND --port $PORT"
fi

echo "Launching server with command: $COMMAND"
cd /Documents/Python/RamaBerry_server > /dev/null 2>&1
$COMMAND
