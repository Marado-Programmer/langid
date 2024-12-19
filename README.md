# LANGuage IDentification

To identify languages.

# Usage

Have a file like this where the first white spaces separates class and phrase:
``` txt
TR onlara bir şey vermek için onları görecektim ama çok yorgundum
EN i was going to see them to give them something but i was very tired
PT eu ia vê-los para lhes dar alguma coisa, mas estava muito cansado
```

```sh
sed -e "s/^[^[:space:]]*[[:space:]]*//" ./phrases | \
    zig build run -- vocab --method=bag_of_words --sort > ./vocab
cat ./phrases | ./zig-out/bin/idlang model --vocab=./vocab
```
