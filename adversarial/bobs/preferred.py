
def preferred(start=1, stop=1000):
    """
    Returns a list of "preferred numbers".
    
    These are nice round numbers that ascend in a
    roughly exponential pattern. The current
    implementation (and the default in all future
    versions) is the sequence 1, 2, 3, 5, 7, 10,
    15, 20, 30, 45, 70, and then repeating the
    numbers 10 to 70 multiplied by each successive
    power of 10.
    
    The start and stop are both included if they're
    part of the preferred number sequence.
    """
    for n in [1, 2, 3, 5, 7]:
        if limit is not None and n > limit:
            return
        yield n
    p = 0
    while True:
        for a in [10, 15, 20, 30, 45, 70]:
            n = a * 10 ** p
            if limit is not None and n > limit:
                return
            yield n
        p += 1


def preferred_sequence(start=1):
    """
    Returns an infinite sequence of "preferred numbers".
    
    These are nice round numbers that ascend in a
    roughly exponential pattern. The current
    implementation (and the default in all future
    versions) is the sequence 1, 2, 3, 5, 7, 10,
    15, 20, 30, 45, 70, and then repeating the
    numbers 10 to 70 multiplied by each successive
    power of 10.
    """
    


def preferred_real(start, stop):
    """
    Generates a sequence of nice round
    floating-point numbers,
    spaced roughly equally on a logarithmic scale,
    between the specified bounds. The sequence
    cycles through mantissas of 1, 1.5, 2, 3, 4.5, 7
    at each power of ten.
    
    The start and stop are both included if they're
    part of the preferred number sequence.
    """
    
    
