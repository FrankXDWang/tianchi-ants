from multiprocessing import Pool

def do(data):
    return data

if __name__ == '__main__':
    data = [1,2,3]
    idata = iter(data)
    p = Pool()
    results = p.map_async(do, idata)
    p.close()
    p.join()
    print(results.get())
    print('Done!')
