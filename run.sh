#!/usr/bin/env bash

case $1 in

    train)
        python main.py --gpus=3 --batch=64 --wd 1e-3
        ;;

    test)
        python gen_result.py --gpus=3
        ;;

    ensemble)
        python ensemble_result.py --gpus=3
        ;;

    *)
        echo "wrong argument"
		exit 1
        ;;
esac
exit 0