#!/bin/sh
for d in output/*; do
    f=$d/qa_output_main.json
    out=$d/qa_good.txt
    rm -f $out

    if [ -s "$f" ]; then
        if [ -n "$(grep '"classification": "bad"' $f)" ]; then
            echo $d bad
            echo 0 > $out
        else
            echo $d good
            echo 1 > $out
        fi
    else
        echo $d no data
    fi
done
