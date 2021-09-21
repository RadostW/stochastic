import pychastic
wiener = pychastic.wiener.Wiener()
for x in range(10000):
    wiener.get_w(x)