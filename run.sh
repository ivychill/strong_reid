#!/usr/bin/env bash

case $1 in

    train)
        python main.py --gpus=2 --batch=64
        ;;

    test)
        python gen_result.py
        ;;

    *)
        echo "wrong argument"
		exit 1
        ;;
esac
exit 0