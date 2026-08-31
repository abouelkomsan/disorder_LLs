# Magnetic-Brillouin-zone bookkeeping: the supercell, its reciprocal
# vectors, the allowed momenta, and folding back into the zone.

"""
    lattice(a, b, c, d) -> [a b; c d]

The supercell matrix, in units of the primitive reciprocal vectors `G1`, `G2`.
`abs(det)` is the number of flux quanta.
"""
lattice(a, b, c, d) = [a b; c d]

"""
    g1g2(klat, G1, G2)

Reciprocal vectors of the supercell: `g1`, `g2` are the momentum quanta such
that the allowed momenta are integer combinations of them.
"""
function g1g2(lattice,G1,G2)
    a = lattice[1,1]
    b = lattice[1,2]
    c = lattice[2,1]
    d = lattice[2,2]
    size = abs(det(lattice))
    sgn = sign(det(lattice))
    g1 = (1/size)*(d*sgn*G1 - c*sgn*G2)
    g2 = (1/size)*(-b*sgn*G1 + a*sgn*G2)
    return g1,g2,sgn*G1,sgn*G2,sgn
end

"""
    kpoints(klat)

Integer labels of the momenta in the magnetic Brillouin zone.
"""
function kpoints(lattice)
    a = lattice[1,1]
    b = lattice[1,2]
    c = lattice[2,1]
    d = lattice[2,2]
    limit = abs(det(lattice))
   # if b != 0 && c != 0
    xlist = []
    ylist = []
    for x in -limit:limit
        for y in -limit:limit
            if d*x-b*y >= ceil(-limit/2) && d*x-b*y <= ceil(limit/2)-1 && -c*x+a*y >= ceil(-limit/2) && -c*x+a*y <= ceil(limit/2)-1
                push!(xlist,x)
                push!(ylist,y)
            end
        end
    end
    klabel = [[xlist[i],ylist[i]] for i in eachindex(xlist)]
    return klabel
end

"""
    projector(k, klat)

Fold a momentum label back into the magnetic Brillouin zone:
`P(k) = k0` where `k = k0 + m*G1 + n*G2`.  Returns `(k0, m, n)`.
"""
function projector(k,lattice) #P(k) = k0 if k= k0+m*G1+n*G2
    m = k[1]
    n = k[2]
    latticesize = abs(det(lattice))
    sgn = sign(det(lattice))
    a = lattice[1,1]
    b = lattice[1,2]
    c = lattice[2,1]
    d = lattice[2,2]
    i =0
    j =0
    G1limit = m*d-n*b
    G2limit = n*a - m*c
    if G1limit > ceil(latticesize/2)-1 
        while G1limit > ceil(latticesize/2)-1 
            G1limit = G1limit - latticesize
            m = m-sgn*a
            n = n-sgn*c
            i = i+1
        end
    end
    if G2limit > ceil(latticesize/2)-1 
        while G2limit > ceil(latticesize/2)-1 
            G2limit = G2limit - latticesize
            m = m -sgn*b
            n = n -sgn*d
            j =j+1
        end
    end
    if G1limit < ceil(-latticesize/2)
        while G1limit < ceil(-latticesize/2)
            G1limit = G1limit + latticesize
            m = m+sgn*a
            n = n+sgn*c
            i = i-1
        end
    end
    if G2limit < ceil(-latticesize/2)
        while G2limit < ceil(-latticesize/2)
            G2limit = G2limit + latticesize
            m = m+sgn*b
            n = n+sgn*d
            j =j-1
        end
    end
    return [m,n],convert(Int8,sgn*i),convert(Int8,sgn*j)
end
