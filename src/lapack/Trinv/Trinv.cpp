/*
   This file is part of Elemental, a library for distributed-memory dense 
   linear algebra.

   Copyright (c) 2009-2010 Jack Poulson <jack.poulson@gmail.com>.
   All rights reserved.

   This file is released under the terms of the license contained in the file
   LICENSE-PURE.
*/
#include "elemental/lapack_internal.hpp"
using namespace elemental;

template<typename T>
void
elemental::lapack::Trinv
( Shape shape, 
  Diagonal diagonal, 
  DistMatrix<T,MC,MR>& A  )
{
#ifndef RELEASE
    PushCallStack("lapack::Trinv");
#endif
    lapack::internal::TrinvVar3( shape, diagonal, A );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::lapack::internal::TrinvVar3
( Shape shape, 
  Diagonal diagonal, 
  DistMatrix<T,MC,MR>& A  )
{
#ifndef RELEASE
    PushCallStack("lapack::internal::TrinvVar3");
#endif
    if( shape == Lower )
        lapack::internal::TrinvLVar3( diagonal, A );
    else
        lapack::internal::TrinvUVar3( diagonal, A );
#ifndef RELEASE
    PopCallStack();
#endif
}

template void elemental::lapack::Trinv
( Shape shape, 
  Diagonal diagonal, 
  DistMatrix<float,MC,MR>& A );

template void elemental::lapack::internal::TrinvVar3
( Shape shape, 
  Diagonal diagonal, 
  DistMatrix<float,MC,MR>& A );

template void elemental::lapack::Trinv
( Shape shape, 
  Diagonal diagonal, 
  DistMatrix<double,MC,MR>& A );

template void elemental::lapack::internal::TrinvVar3
( Shape shape, 
  Diagonal diagonal, 
  DistMatrix<double,MC,MR>& A );

#ifndef WITHOUT_COMPLEX
template void elemental::lapack::Trinv
( Shape shape, 
  Diagonal diagonal, 
  DistMatrix<scomplex,MC,MR>& A );

template void elemental::lapack::internal::TrinvVar3
( Shape shape, 
  Diagonal diagonal,
  DistMatrix<scomplex,MC,MR>& A );

template void elemental::lapack::Trinv
( Shape shape, 
  Diagonal diagonal, 
  DistMatrix<dcomplex,MC,MR>& A );

template void elemental::lapack::internal::TrinvVar3
( Shape shape, 
  Diagonal diagonal,
  DistMatrix<dcomplex,MC,MR>& A );
#endif

