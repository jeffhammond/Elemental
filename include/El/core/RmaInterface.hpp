/*
   Copyright (c) 2009-2014, Jack Poulson
   Copyright (c) 2011, The University of Texas at Austin
   Copyright (c) 2014, Jeff Hammond (Intel)
   All rights reserved.

   Authors:
   Jeff Hammond adapted the RMA interface from the AXPY one.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_RMAINTERFACE_HPP
#define EL_RMAINTERFACE_HPP

namespace El {

namespace RmaTypeNS {
enum RmaType { LOCAL_TO_GLOBAL, GLOBAL_TO_LOCAL };
}
using namespace RmaTypeNS;

template<typename T>
class RmaInterface
{
public:
    RmaInterface();
    ~RmaInterface();

    RmaInterface( RmaType type,       DistMatrix<T,MC,MR>& Z );
    RmaInterface( RmaType type, const DistMatrix<T,MC,MR>& Z );

    void Attach( RmaType type,       DistMatrix<T,MC,MR>& Z );
    void Attach( RmaType type, const DistMatrix<T,MC,MR>& Z );

    void Axpy( T alpha,       Matrix<T>& Z, Int i, Int j );
    void Axpy( T alpha, const Matrix<T>& Z, Int i, Int j );

    void Detach();

private:
    void RmaLocalToGlobal( T alpha, const Matrix<T>& X, Int i, Int j );
    void RmaGlobalToLocal( T alpha,       Matrix<T>& Y, Int i, Int j );

};

} // namespace El

#endif // ifndef EL_RMAINTERFACE_HPP
