def dot_product(v, L):
    return sum([v[i] * L[i] for i in range(len(v))])

def len_vector(v):
    return sum([x ** 2 for x in v]) ** 0.5

def orthogonal_projection(v, L):
    """
    Compute the orthogonal projection of vector v onto line L.

    :param v: The vector to be projected
    :param L: The line vector defining the direction of projection
    :return: List representing the projection of v onto L
    """

    '''
    Projection (v, L) = (v . L / ||L|| * ||L||) * L
    '''
    numerater = dot_product(v, L)
    denominator = len_vector(L) ** 2

    return [(numerater / denominator) * x for x in L]
    
    
