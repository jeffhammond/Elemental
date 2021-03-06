/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_APPLYPACKEDREFLECTORS_LLHB_HPP
#define EL_APPLYPACKEDREFLECTORS_LLHB_HPP

namespace El {
namespace apply_packed_reflectors {

//
// Since applying Householder transforms from vectors stored bottom-to-top
// implies that we will be forming a generalization of 
//
//  (I - tau_0 v_0^T conj(v_0)) (I - tau_1 v_1^T conj(v_1)) = 
//  I - [ v_0^T, v_1^T ] [  tau_0, -tau_0 tau_1 conj(v_0) v_1^T ] [ conj(v_0) ]
//                       [  0,      tau_1                       ] [ conj(v_1) ],
//
// which has a upper-triangular center matrix, say S, we will form S as 
// the inverse of a matrix T, which can easily be formed as
// 
//   triu(T,1) = triu( conj(V V^H) ),
//   diag(T) = 1/householderScalars or 1/conj(householderScalars),
//
// where V is the matrix of Householder vectors and householderScalars is the
// vector of Householder reflection coefficients.
//
// V is stored row-wise in the matrix.
//

template<typename F> 
void LLHBUnblocked
( Conjugation conjugation,
  Int offset, 
  const Matrix<F>& H,
  const Matrix<F>& householderScalars,
        Matrix<F>& A )
{
    DEBUG_CSE
    DEBUG_ONLY(
      if( H.Width() != A.Height() )
          LogicError("H's width must match A's height");
    )
    const Int diagLength = H.DiagonalLength(offset);
    DEBUG_ONLY(
      if( householderScalars.Height() != diagLength )
          LogicError
          ("householderScalars must be the same length as H's offset diag");
    )
    Matrix<F> hPanCopy, z;

    const Int iOff = ( offset>=0 ? 0      : -offset );
    const Int jOff = ( offset>=0 ? offset : 0       );

    for( Int k=diagLength-1; k>=0; --k )
    {
        const Int ki = k+iOff;
        const Int kj = k+jOff;

        auto hPan = H( IR(ki),      IR(0,kj+1) );
        auto ATop = A( IR(0,kj+1),  ALL        );
        const F tau = householderScalars(k);
        const F gamma = ( conjugation == CONJUGATED ? Conj(tau) : tau );

        // Convert to an explicit (scaled) Householder vector
        hPanCopy = hPan;
        hPanCopy(0,kj) = 1;

        // z := ATop' hPan^T
        Gemv( ADJOINT, F(1), ATop, hPanCopy, z );
        // ATop := (I - gamma hPan^T conj(hPan)) ATop = ATop - gamma hPan^T z'
        Ger( -gamma, hPanCopy, z, ATop );
    }
}

template<typename F> 
void LLHBBlocked
( Conjugation conjugation,
  Int offset, 
  const Matrix<F>& H,
  const Matrix<F>& householderScalars,
        Matrix<F>& A )
{
    DEBUG_CSE
    DEBUG_ONLY(
      if( H.Width() != A.Height() )
          LogicError("H's width must match A's height");
    )
    const Int diagLength = H.DiagonalLength(offset);
    DEBUG_ONLY(
      if( householderScalars.Height() != diagLength )
          LogicError
          ("householderScalars must be the same length as H's offset diag");
    )
    Matrix<F> HPanConj, SInv, Z;

    const Int iOff = ( offset>=0 ? 0      : -offset );
    const Int jOff = ( offset>=0 ? offset : 0       );

    const Int bsize = Blocksize();
    const Int kLast = LastOffset( diagLength, bsize );
    for( Int k=kLast; k>=0; k-=bsize )
    {
        const Int nb = Min(bsize,diagLength-k);
        const Int ki = k+iOff;
        const Int kj = k+jOff;

        auto HPan = H( IR(ki,ki+nb), IR(0,kj+nb) );
        auto ATop = A( IR(0,kj+nb),  ALL         );
        auto householderScalars1 = householderScalars( IR(k,k+nb), ALL );

        // Convert to an explicit matrix of (scaled) Householder vectors
        Conjugate( HPan, HPanConj );
        MakeTrapezoidal( LOWER, HPanConj, HPanConj.Width()-HPanConj.Height() );
        FillDiagonal( HPanConj, F(1), HPanConj.Width()-HPanConj.Height() );

        // Form the small triangular matrix needed for the UT transform
        Herk( UPPER, NORMAL, Base<F>(1), HPanConj, SInv );
        FixDiagonal( conjugation, householderScalars1, SInv );

        // Z := conj(HPan) ATop
        Gemm( NORMAL, NORMAL, F(1), HPanConj, ATop, Z );
        // Z := inv(SInv) conj(HPan) ATop
        Trsm( LEFT, UPPER, NORMAL, NON_UNIT, F(1), SInv, Z );
        // ATop := (I - HPan^T inv(SInv) conj(HPan)) ATop
        Gemm( ADJOINT, NORMAL, F(-1), HPanConj, Z, F(1), ATop );
    }
}

template<typename F> 
void LLHB
( Conjugation conjugation,
  Int offset, 
  const Matrix<F>& H,
  const Matrix<F>& householderScalars,
        Matrix<F>& A )
{
    DEBUG_CSE
    const Int numRHS = A.Width();
    const Int blocksize = Blocksize();
    if( numRHS < blocksize )
    {
        LLHBUnblocked( conjugation, offset, H, householderScalars, A );
    }
    else
    {
        LLHBBlocked( conjugation, offset, H, householderScalars, A );
    }
}

template<typename F> 
void LLHBUnblocked
( Conjugation conjugation,
  Int offset, 
  const AbstractDistMatrix<F>& H,
  const AbstractDistMatrix<F>& householderScalarsPre, 
        AbstractDistMatrix<F>& APre )
{
    DEBUG_CSE
    DEBUG_ONLY(
      if( H.Width() != APre.Height() )
          LogicError("H's width must match A's height");
      AssertSameGrids( H, householderScalarsPre, APre );
    )

    // We gather the entire set of Householder scalars at the start rather than
    // continually paying the latency cost of the broadcasts in a 'Get' call
    DistMatrixReadProxy<F,F,STAR,STAR>
      householderScalarsProx( householderScalarsPre );
    auto& householderScalars = householderScalarsProx.GetLocked();

    DistMatrixReadWriteProxy<F,F,MC,MR> AProx( APre );
    auto& A = AProx.Get();

    const Int diagLength = H.DiagonalLength(offset);
    DEBUG_ONLY(
      if( householderScalars.Height() != diagLength )
          LogicError
          ("householderScalars must be the same length as H's offset diag");
    )
    const Grid& g = H.Grid();
    auto hPan = unique_ptr<AbstractDistMatrix<F>>( H.Construct(g,H.Root()) );
    DistMatrix<F,STAR,MC> hPan_STAR_MC(g); 
    DistMatrix<F,MR,STAR> z_MR_STAR(g);

    const Int iOff = ( offset>=0 ? 0      : -offset );
    const Int jOff = ( offset>=0 ? offset : 0       );

    for( Int k=diagLength-1; k>=0; --k )
    {
        const Int ki = k+iOff;
        const Int kj = k+jOff;

        auto ATop = A( IR(0,kj+1), ALL );
        const F tau = householderScalars.GetLocal( k, 0 );
        const F gamma = ( conjugation == CONJUGATED ? Conj(tau) : tau );

        // Convert to an explicit (scaled) Householder vector
        LockedView( *hPan, H, IR(ki), IR(0,kj+1) );
        hPan_STAR_MC.AlignWith( ATop );
        Conjugate( *hPan, hPan_STAR_MC );
        hPan_STAR_MC.Set( 0, kj, F(1) );

        // z := ATop' hPan^T
        z_MR_STAR.AlignWith( ATop );
        Zeros( z_MR_STAR, ATop.Width(), 1 );
        LocalGemv( ADJOINT, F(1), ATop, hPan_STAR_MC, F(0), z_MR_STAR );
        El::AllReduce( z_MR_STAR.Matrix(), ATop.ColComm() );

        // ATop := (I - gamma hPan^T conj(hPan)) ATop = ATop - gamma hPan^T z'
        LocalGer( -gamma, hPan_STAR_MC, z_MR_STAR, ATop );
    }
}

template<typename F> 
void LLHBBlocked
( Conjugation conjugation,
  Int offset, 
  const AbstractDistMatrix<F>& H,
  const AbstractDistMatrix<F>& householderScalarsPre, 
        AbstractDistMatrix<F>& APre )
{
    DEBUG_CSE
    DEBUG_ONLY(
      if( H.Width() != APre.Height() )
          LogicError("H's width must match A's height");
      AssertSameGrids( H, householderScalarsPre, APre );
    )

    DistMatrixReadProxy<F,F,MC,STAR>
      householderScalarsProx( householderScalarsPre );
    auto& householderScalars = householderScalarsProx.GetLocked();

    DistMatrixReadWriteProxy<F,F,MC,MR> AProx( APre );
    auto& A = AProx.Get();

    const Int diagLength = H.DiagonalLength(offset);
    DEBUG_ONLY(
      if( householderScalars.Height() != diagLength )
          LogicError
          ("householderScalars must be the same length as H's offset diag");
    )
    const Grid& g = H.Grid();
    auto HPan = unique_ptr<AbstractDistMatrix<F>>( H.Construct(g,H.Root()) );
    DistMatrix<F> HPanConj(g);
    DistMatrix<F,STAR,VR> HPan_STAR_VR(g), Z_STAR_VR(g);
    DistMatrix<F,STAR,MC> HPan_STAR_MC(g); 
    DistMatrix<F,STAR,MR> Z_STAR_MR(g);
    DistMatrix<F,STAR,STAR> householderScalars1_STAR_STAR(g), SInv_STAR_STAR(g);

    const Int iOff = ( offset>=0 ? 0      : -offset );
    const Int jOff = ( offset>=0 ? offset : 0       );

    const Int bsize = Blocksize();
    const Int kLast = LastOffset( diagLength, bsize );
    for( Int k=kLast; k>=0; k-=bsize )
    {
        const Int nb = Min(bsize,diagLength-k);
        const Int ki = k+iOff;
        const Int kj = k+jOff;

        LockedView( *HPan, H, IR(ki,ki+nb), IR(0,kj+nb) );
        auto ATop = A( IR(0,kj+nb), ALL );
        auto householderScalars1 = householderScalars( IR(k,k+nb), ALL );

        // Convert to an explicit matrix of (scaled) Householder vectors
        Conjugate( *HPan, HPanConj );
        MakeTrapezoidal( LOWER, HPanConj, HPanConj.Width()-HPanConj.Height() );
        FillDiagonal( HPanConj, F(1), HPanConj.Width()-HPanConj.Height() );

        // Form the small triangular matrix needed for the UT transform
        HPan_STAR_VR = HPanConj;
        Zeros( SInv_STAR_STAR, nb, nb );
        Herk
        ( UPPER, NORMAL,
          Base<F>(1), HPan_STAR_VR.LockedMatrix(),
          Base<F>(0), SInv_STAR_STAR.Matrix() );
        El::AllReduce( SInv_STAR_STAR, HPan_STAR_VR.RowComm() );
        householderScalars1_STAR_STAR = householderScalars1;
        FixDiagonal
        ( conjugation, householderScalars1_STAR_STAR, SInv_STAR_STAR );

        // Z := conj(HPan) ATop
        HPan_STAR_MC.AlignWith( ATop );
        HPan_STAR_MC = HPan_STAR_VR;
        Z_STAR_MR.AlignWith( ATop );
        LocalGemm( NORMAL, NORMAL, F(1), HPan_STAR_MC, ATop, Z_STAR_MR );
        Z_STAR_VR.AlignWith( ATop );
        Contract( Z_STAR_MR, Z_STAR_VR );

        // Z := inv(SInv) conj(HPan) ATop
        LocalTrsm
        ( LEFT, UPPER, NORMAL, NON_UNIT, F(1), SInv_STAR_STAR, Z_STAR_VR );

        // ATop := (I - HPan^T inv(SInv) conj(HPan)) ATop
        Z_STAR_MR = Z_STAR_VR;
        LocalGemm
        ( ADJOINT, NORMAL, F(-1), HPan_STAR_MC, Z_STAR_MR, F(1), ATop );
    }
}

template<typename F> 
void LLHB
( Conjugation conjugation,
  Int offset, 
  const AbstractDistMatrix<F>& H,
  const AbstractDistMatrix<F>& householderScalars, 
        AbstractDistMatrix<F>& A )
{
    DEBUG_CSE
    const Int numRHS = A.Width();
    const Int blocksize = Blocksize();
    if( numRHS < blocksize )
    {
        LLHBUnblocked( conjugation, offset, H, householderScalars, A );
    }
    else
    {
        LLHBBlocked( conjugation, offset, H, householderScalars, A );
    }
}

} // namespace apply_packed_reflectors
} // namespace El

#endif // ifndef EL_APPLYPACKEDREFLECTORS_LLHB_HPP
