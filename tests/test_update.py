

d1 = {"a": 2, "b" : 5}
d2 = {"a": 3, "c" : 7}
# d1.update(d2)

for k in d2.keys():
    if k in d1.keys():
        d1[k] = d1[k] + d2[k]
    else:
        d1[k] = d2[k]

print(d1)