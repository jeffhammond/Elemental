/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

#define COLDIST STAR
#define ROWDIST MC

#include "./setup.hpp"

namespace El {

// Public section
// ##############

// Assignment and reconfiguration
// ==============================

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,MC,MR,BLOCK>& A )
{
    DEBUG_ONLY(CSE cse("[STAR,MC] = [MC,MR]"))
    auto A_STAR_VR = MakeUnique<DistMatrix<T,STAR,VR,BLOCK>>( A );
    auto A_STAR_VC =
      MakeUnique<DistMatrix<T,STAR,VC,BLOCK>>( this->Grid() );
    A_STAR_VC->AlignRowsWith(*this);
    *A_STAR_VC = *A_STAR_VR;
    A_STAR_VR.reset(); 

    *this = *A_STAR_VC;
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,MC,STAR,BLOCK>& A )
{
    DEBUG_ONLY(CSE cse("[STAR,MC] = [MC,STAR]"))
    auto A_MC_MR = MakeUnique<DistMatrix<T,MC,MR,BLOCK>>( A );
    auto A_STAR_VR = MakeUnique<DistMatrix<T,STAR,VR,BLOCK>>( *A_MC_MR );
    A_MC_MR.reset();

    auto A_STAR_VC =
      MakeUnique<DistMatrix<T,STAR,VC,BLOCK>>( this->Grid() );
    A_STAR_VC->AlignRowsWith(*this);
    *A_STAR_VC = *A_STAR_VR;
    A_STAR_VR.reset();

    *this = *A_STAR_VC;
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,STAR,MR,BLOCK>& A )
{ 
    DEBUG_ONLY(CSE cse("[STAR,MC] = [STAR,MR]"))
    LogicError("This routine is not yet written");
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,MD,STAR,BLOCK>& A )
{
    DEBUG_ONLY(CSE cse("[STAR,MC] = [MD,STAR]"))
    // TODO: More efficient implementation?
    DistMatrix<T,STAR,STAR,BLOCK> A_STAR_STAR( A );
    *this = A_STAR_STAR;
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,STAR,MD,BLOCK>& A )
{
    DEBUG_ONLY(CSE cse("[STAR,MC] = [STAR,MD]"))
    // TODO: More efficient implementation?
    DistMatrix<T,STAR,STAR,BLOCK> A_STAR_STAR( A );
    *this = A_STAR_STAR;
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,MR,MC,BLOCK>& A )
{ 
    DEBUG_ONLY(CSE cse("[STAR,MC] = [MR,MC]"))
    copy::ColAllGather( A, *this );
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,MR,STAR,BLOCK>& A )
{ 
    DEBUG_ONLY(CSE cse("[STAR,MC] = [MR,STAR]"))
    DistMatrix<T,MR,MC,BLOCK> A_MR_MC( A );
    *this = A_MR_MC;
    return *this;
}

template<typename T>
BDM& BDM::operator=( const BDM& A )
{ 
    DEBUG_ONLY(CSE cse("[STAR,MC] = [STAR,MC]"))
    copy::Translate( A, *this );
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,VC,STAR,BLOCK>& A )
{ 
    DEBUG_ONLY(CSE cse("[STAR,MC] = [VC,STAR]"))
    auto A_VR_STAR = MakeUnique<DistMatrix<T,VR,STAR,BLOCK>>( A );
    auto A_MR_MC = MakeUnique<DistMatrix<T,MR,MC,BLOCK>>( this->Grid() );
    A_MR_MC->AlignRowsWith(*this);
    *A_MR_MC = *A_VR_STAR;
    A_VR_STAR.reset();

    *this = *A_MR_MC;
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,STAR,VC,BLOCK>& A )
{ 
    DEBUG_ONLY(CSE cse("[STAR,MC] = [STAR,VC]"))
    copy::PartialRowAllGather( A, *this );
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,VR,STAR,BLOCK>& A )
{ 
    DEBUG_ONLY(CSE cse("[STAR,MC] = [VR,STAR]"))
    DistMatrix<T,MR,MC,BLOCK> A_MR_MC( A );
    *this = A_MR_MC;
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,STAR,VR,BLOCK>& A )
{ 
    DEBUG_ONLY(CSE cse("[STAR,MC] = [STAR,VR]"))
    DistMatrix<T,STAR,VC,BLOCK> A_STAR_VC(this->Grid());
    A_STAR_VC.AlignRowsWith(*this);
    *this = A_STAR_VC = A;
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,STAR,STAR,BLOCK>& A )
{
    DEBUG_ONLY(CSE cse("[STAR,MC] = [STAR,STAR]"))
    copy::RowFilter( A, *this );
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,CIRC,CIRC,BLOCK>& A )
{
    DEBUG_ONLY(CSE cse("[STAR,MC] = [CIRC,CIRC]"))
    DistMatrix<T,MR,MC,BLOCK> A_MR_MC( A.Grid() );
    A_MR_MC.AlignWith( *this );
    A_MR_MC = A;
    *this = A_MR_MC;
    return *this;
}

template<typename T>
BDM& BDM::operator=( const BlockMatrix<T>& A )
{
    DEBUG_ONLY(CSE cse("BDM = ABDM"))
    #define GUARD(CDIST,RDIST) \
      A.DistData().colDist == CDIST && A.DistData().rowDist == RDIST
    #define PAYLOAD(CDIST,RDIST) \
      auto& ACast = \
        dynamic_cast<const DistMatrix<T,CDIST,RDIST,BLOCK>&>(A); \
      *this = ACast;
    #include "El/macros/GuardAndPayload.h"
    return *this;
}

// Basic queries
// =============

template<typename T>
mpi::Comm BDM::DistComm() const EL_NO_EXCEPT 
{ return this->grid_->MCComm(); }
template<typename T>
mpi::Comm BDM::CrossComm() const EL_NO_EXCEPT
{ return ( this->Participating() ? mpi::COMM_SELF : mpi::COMM_NULL ); }
template<typename T>
mpi::Comm BDM::RedundantComm() const EL_NO_EXCEPT
{ return this->grid_->MRComm(); }
template<typename T>
mpi::Comm BDM::ColComm() const EL_NO_EXCEPT
{ return ( this->Participating() ? mpi::COMM_SELF : mpi::COMM_NULL ); }
template<typename T>
mpi::Comm BDM::RowComm() const EL_NO_EXCEPT
{ return this->grid_->MCComm(); }

template<typename T>
int BDM::ColStride() const EL_NO_EXCEPT
{ return 1; }
template<typename T>
int BDM::RowStride() const EL_NO_EXCEPT
{ return this->grid_->MCSize(); }
template<typename T>
int BDM::DistSize() const EL_NO_EXCEPT
{ return this->grid_->MCSize(); }
template<typename T>
int BDM::CrossSize() const EL_NO_EXCEPT
{ return 1; }
template<typename T>
int BDM::RedundantSize() const EL_NO_EXCEPT
{ return this->grid_->MRSize(); }

// Instantiate {Int,Real,Complex<Real>} for each Real in {float,double}
// ####################################################################

#define SELF(T,U,V) \
  template DistMatrix<T,COLDIST,ROWDIST,BLOCK>::DistMatrix \
  ( const DistMatrix<T,U,V,BLOCK>& A );
#define OTHER(T,U,V) \
  template DistMatrix<T,COLDIST,ROWDIST,BLOCK>::DistMatrix \
  ( const DistMatrix<T,U,V>& A ); \
  template DistMatrix<T,COLDIST,ROWDIST,BLOCK>& \
           DistMatrix<T,COLDIST,ROWDIST,BLOCK>::operator= \
           ( const DistMatrix<T,U,V>& A )
#define BOTH(T,U,V) \
  SELF(T,U,V); \
  OTHER(T,U,V)
#define PROTO(T) \
  template class DistMatrix<T,COLDIST,ROWDIST,BLOCK>; \
  BOTH( T,CIRC,CIRC); \
  BOTH( T,MC,  MR  ); \
  BOTH( T,MC,  STAR); \
  BOTH( T,MD,  STAR); \
  BOTH( T,MR,  MC  ); \
  BOTH( T,MR,  STAR); \
  OTHER(T,STAR,MC  ); \
  BOTH( T,STAR,MD  ); \
  BOTH( T,STAR,MR  ); \
  BOTH( T,STAR,STAR); \
  BOTH( T,STAR,VC  ); \
  BOTH( T,STAR,VR  ); \
  BOTH( T,VC,  STAR); \
  BOTH( T,VR,  STAR);

#define EL_ENABLE_QUAD
#include "El/macros/Instantiate.h"

} // namespace El