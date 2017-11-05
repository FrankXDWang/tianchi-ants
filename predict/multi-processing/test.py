import multiprocessing

def task(i):
    return [elem for elem in range(i)]

if __name__ == '__main__':
    cpu_nums = multiprocessing.cpu_count()
    pool = multiprocessing.Pool()
    results = []
    for i in range(0, cpu_nums):
        result = pool.apply_async(task, args=(i,))
        print(result.get())
        results.append(result)
    pool.close()
    pool.join()
    #for result in results:
    #    print(result.get())
