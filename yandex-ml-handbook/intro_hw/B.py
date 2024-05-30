import numpy as np

def most_frequent(nums):
    """
    Find the most frequent value in an array
    :param nums: array of ints
    :return: the most frequent value
    """
    # first try
    # freq = dict()
    # for num in nums:
    #     freq[num] = freq.get(num, 0) + 1
    # res, max_freq = None, -1
    # for key, value in freq.items():
    #     if value > max_freq:
    #         max_freq = value
    #         res = key
    # return res      
    
    # second try
    freq = np.bincount(nums)
    return np.argmax(freq)    