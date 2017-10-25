from functools import reduce
def _split(wifis, wifi_infos):
    for wifi in wifi_infos:
            for _wifi in wifi.split(';'):
                wifis.add(_wifi.split('|')[0])
wifis = set()
wifis = reduce(_split,'b_34366982|-82|false;b_37756289|-53|false;')
print(wifis)
