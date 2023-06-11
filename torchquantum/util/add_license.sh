#!/bin/bash
#script to loop through directories to merge files

walk_dir () {    
    shopt -s nullglob dotglob

    for pathname in "$1"/*; do
        if [ -d "$pathname" ]; then
            walk_dir "$pathname"
        else
            case "$pathname" in
                *.py|*.pypypy)
                    printf '%s\n' "$pathname"
                    if ! grep -q Copyright $pathname
                    then
                        cat license.py $pathname >$pathname.new && mv $pathname.new $pathname
                    fi
            esac
        fi
    done
}

DOWNLOADING_DIR="./"

walk_dir "$DOWNLOADING_DIR"