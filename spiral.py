import numpy as np
import scipy.ndimage as nd

def spiral_gen(w):
    """ generator to produce the directions to build a spiral """
    i = 0
    s = 1
    n = w*w
    dirs = [(1,0),(0,1),(-1,0),(0,-1)]
    index = 0
    rounds = 0
    # we generate n - 1 directions because we don't yield 0,0 at the start
    # it might be cleaner if we did.
    while(i < n - 1):
        for j in xrange(0,s):
            if i >= n - 1:
                return
            yield dirs[index%4]
            i += 1
        rounds += 1
        index = rounds
        # observation we have a series of 1,1,2,2,3,3,4,5,5,6,6 for repeats of
        # direction so this tries to repeat dir. I guess we could achieve it
        # via rounds/2 % 4 for direction
        if rounds % 2 == 0:
            s += 1

for x in spiral_gen(1):
    print x

def test_spiral_gen():
    assert(list(spiral_gen(1)) == list())
    assert(list(spiral_gen(2)) == [(1,0),(0,1),(-1,0)])
    assert(list(spiral_gen(3)) == [(1,0),(0,1),(-1,0),
                                   (-1,0),(0,-1),(0,-1),
                                   (1,0),(1,0)])

def spiral(md):
    """
    this makes square spirals md is the max dimension
    """
    spiral = np.zeros((md,md),dtype=np.int32)
    sx = md/2
    sy = md/2
    if (md % 2 == 0):
        sx -= 1
        sy -= 1
    cnt = 1
    spiral[sx,sy] = cnt
    cnt += 1
    for d in spiral_gen(md):
        sx += d[1]
        sy += d[0]
        if sx >= 0 and sx < md and sy < md and sy >= 0:
            spiral[sx,sy] = cnt
        cnt += 1
    return spiral

def boxy_spiral(md,w,h):
    return nd.interpolation.zoom(spiral(md),(w,h),order=0)

def test_spiral():
    test_spiral_gen()
    answer = np.array([7,8,9,
                       6,1,2,
                       5,4,3   ],dtype=np.int32).reshape((3,3))
    print(spiral(3))
    assert(np.all(spiral(3) == answer))

    answer = np.array([21,22,23,24, 25,
                       20, 7, 8, 9, 10,
                       19, 6, 1, 2, 11,
                       18, 5, 4, 3, 12,
                       17,16,15,14, 13],dtype=np.int32).reshape((5,5))
    print(spiral(5))
    assert(np.all(spiral(5) == answer))

    print(spiral(8))
    print(spiral(9))
    print(spiral(9)/4)
    print(spiral(16))

    print(nd.interpolation.zoom(spiral(4),(40,30),order=0))


    answer = np.array([1],dtype=np.int32).reshape((1,1))
    print spiral(1)
    assert(np.all(spiral(1) == answer))
    answer = np.array([1,2,
                       4,3],dtype=np.int32).reshape((2,2))
    print spiral(2)
    assert(np.all(spiral(2) == answer))


if __name__ == "__main__":
    test_spiral()
