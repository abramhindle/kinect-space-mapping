import numpy as np

def spiral_gen(w):
    i = 0
    s = 1
    n = w*w
    dirs = [(1,0),(0,1),(-1,0),(0,-1)]
    index = 0
    rounds = 0
    while(i < n - 1):
        for j in xrange(0,s):
            if i >= n - 1:
                return
            yield dirs[index%4]
            i += 1
        rounds += 1
        index = rounds
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




    
def spiral(w,h):
    md = max(w,h)
    spiral = np.zeros((md,md),dtype=np.int32)
    # spiral = np.zeros((w,h),dtype=np.int32)
    sx = w/2 
    sy = h/2 
    cnt = 1
    spiral[sx,sy] = cnt
    cnt += 1
    maxcnt = md*md
    offset = 1
    llen = 2
    # directions of the spiral
    dirs = [(0,1),(-1,0),(0,-1),(1,0)]
    # start to the right
    sx += 1
    while(cnt < maxcnt):
        for dire in dirs:            
            for i in xrange(0,llen):
                print spiral
                print (sx,sy,[cnt],i)
                spiral[sy,sx] = cnt
                sx += dire[0]
                sy += dire[1]
                cnt += 1
        sx +=1 # to the right
        llen += 2
    return spiral

def test_spiral():
    test_spiral_gen()
    answer = np.array([7,8,9,
                       6,1,2,
                       5,4,3   ],dtype=np.int32).reshape((3,3))
    assert(spiral(3,3) == answer)
    answer = np.array([1],dtype=np.int32).reshape((1,1))
    assert(spiral(1,1) == answer)
    answer = np.array([3,4,
                       2,1],dtype=np.int32).reshape((2,2))
    assert(spiral(2,2) == answer)

    answer = np.array([21,22,23,24, 25,
                       20, 7, 8, 9, 10,
                       19, 6, 1, 2, 11,
                       18, 5, 4, 3, 12,
                       17,16,15,14, 13],dtype=np.int32).reshape((5,5))
    assert(spiral(5,5) == answer)

test_spiral()
