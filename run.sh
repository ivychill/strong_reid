#!/usr/bin/env bash

case $1 in

    train)
        python main_2loader.py --fp16 --gpus=2 --batch=96 --version=blend_r50
        ;;

    feat)
        python gen_mat.py
        ;;

    json)
        python gen_result.py
        ;;

    mmd)
        python mmd.py
        ;;

    *)
        echo "wrong argument"
		exit 1
        ;;
esac
exit 0