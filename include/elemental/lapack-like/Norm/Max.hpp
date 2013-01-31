/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef LAPACK_NORM_MAX_HPP
#define LAPACK_NORM_MAX_HPP

namespace elem {

template<typename F> 
inline typename Base<F>::type
MaxNorm( const Matrix<F>& A )
{
#ifndef RELEASE
    PushCallStack("MaxNorm");
#endif
    typedef typename Base<F>::type R;

    R maxAbs = 0;
    const int height = A.Height();
    const int width = A.Width();
    for( int j=0; j<width; ++j )
    {
        for( int i=0; i<height; ++i )
        {
            const R thisAbs = Abs(A.Get(i,j));
            maxAbs = std::max( maxAbs, thisAbs );
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
    return maxAbs;
}

template<typename F,Distribution U,Distribution V>
inline typename Base<F>::type
MaxNorm( const DistMatrix<F,U,V>& A )
{
#ifndef RELEASE
    PushCallStack("MaxNorm");
#endif
    typedef typename Base<F>::type R;

    R localMaxAbs = 0;
    const int localHeight = A.LocalHeight();
    const int localWidth = A.LocalWidth();
    for( int jLocal=0; jLocal<localWidth; ++jLocal )
    {
        for( int iLocal=0; iLocal<localHeight; ++iLocal )
        {
            const R thisAbs = Abs(A.GetLocal(iLocal,jLocal));
            localMaxAbs = std::max( localMaxAbs, thisAbs );
        }
    }

    R maxAbs;
    mpi::Comm reduceComm = ReduceComm<U,V>( A.Grid() );
    mpi::AllReduce( &localMaxAbs, &maxAbs, 1, mpi::MAX, reduceComm );
#ifndef RELEASE
    PopCallStack();
#endif
    return maxAbs;
}

} // namespace elem

#endif // ifndef LAPACK_NORM_MAX_HPP