#!/bin/bash

# if IGNORE_M5 is not 1, the command will be read from m5 readfile
if ! [[ "$IGNORE_M5" == 1 ]]; then
    # m5 readfile returns the number of bytes read
    # if it returns 0, it means there's no inputted command
    if m5 readfile > script.sh; then
        chmod +x script.sh;
        echo "Executing the following commands"
        echo "//------"
        cat script.sh
        echo "//------"
        ./script.sh
        sleep 1
        m5 exit
    else
        echo "No inputted command. Dropping to bash."
        IGNORE_M5=1 /bin/bash
    fi
fi

