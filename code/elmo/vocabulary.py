#!/projects/ltg/python3/bin/python3
import sys

THRESHOLD = int(sys.argv[1])

words = {}

for line in sys.stdin:
    res = line.strip().split()
    for word in res:
        if word not in words:
            words[word] = 0
        words[word] += 1

print('Vocabulary:', len(words), file=sys.stderr)

a = sorted(words, key=words.get, reverse=True)[:THRESHOLD]
for w in a:
    print(w + '\t' + str(words[w]))
