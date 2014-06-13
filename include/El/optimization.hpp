/*
   Copyright (c) 2009-2014, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_OPTIMIZATION_HPP
#define EL_OPTIMIZATION_HPP

namespace El {

// Basis pursuit: min || x ||_1 such that A x = b
// ==============================================
template<typename F>
Int BasisPursuit
( const Matrix<F>& A, const Matrix<F>& b,
  Matrix<F>& x, Matrix<F>& z, Matrix<F>& u, 
  Base<F> rho=1., Base<F> alpha=1.2, Int maxIter=500, Base<F> absTol=1e-6, 
  Base<F> relTol=1e-4, bool usePinv=false, Base<F> pinvTol=0, 
  bool progress=true );
template<typename F>
Int BasisPursuit
( const DistMatrix<F>& A, const DistMatrix<F>& b,
  DistMatrix<F>& x, DistMatrix<F>& z, DistMatrix<F>& u,
  Base<F> rho=1., Base<F> alpha=1.2, Int maxIter=500, Base<F> absTol=1e-6, 
  Base<F> relTol=1e-4, bool usePinv=false, Base<F> pinvTol=0,
  bool progress=true );

// Clipping
// ========
template<typename Real>
void LowerClip( Matrix<Real>& X, Real lowerBound=0 );
template<typename Real,Dist U,Dist V>
void LowerClip( DistMatrix<Real,U,V>& X, Real lowerBound=0 );

template<typename Real>
void UpperClip( Matrix<Real>& X, Real upperBound=0 );
template<typename Real,Dist U,Dist V>
void UpperClip( DistMatrix<Real,U,V>& X, Real upperBound=0 );

template<typename Real>
void Clip( Matrix<Real>& X, Real lowerBound=0, Real upperBound=1 );
template<typename Real,Dist U,Dist V>
void Clip( DistMatrix<Real,U,V>& X, Real lowerBound=0, Real upperBound=1 );

// Covariance
// ==========
template<typename F>
void Covariance( const Matrix<F>& D, Matrix<F>& S );
template<typename F>
void Covariance( const DistMatrix<F>& D, DistMatrix<F>& S );

// Linear program
// ==============
template<typename Real>
Int LinearProgram
( const Matrix<Real>& A, const Matrix<Real>& b, const Matrix<Real>& c,
  Matrix<Real>& x, Matrix<Real>& z, Matrix<Real>& u,
  Real rho=1., Real alpha=1.2, Int maxIter=500,
  Real absTol=1e-6, Real relTol=1e-4, bool inv=false, bool progress=true );
template<typename Real>
Int LinearProgram
( const DistMatrix<Real>& A, const DistMatrix<Real>& b,
  const DistMatrix<Real>& c, DistMatrix<Real>& x, DistMatrix<Real>& z,
  DistMatrix<Real>& u,
  Real rho=1., Real alpha=1.2, Int maxIter=500, Real absTol=1e-6,
  Real relTol=1e-4, bool inv=true, bool progress=true );

// Log barrier
// ===========
template<typename F>
Base<F> LogBarrier( UpperOrLower uplo, const Matrix<F>& A );
template<typename F>
Base<F> LogBarrier( UpperOrLower uplo, const DistMatrix<F>& A );

template<typename F>
Base<F> LogBarrier
( UpperOrLower uplo, Matrix<F>& A, bool canOverwrite=false );
template<typename F>
Base<F> LogBarrier
( UpperOrLower uplo, DistMatrix<F>& A, bool canOverwrite=false );

// Log-det divergence
// ==================
template<typename F>
Base<F> LogDetDiv
( UpperOrLower uplo, const Matrix<F>& A, const Matrix<F>& B );
template<typename F>
Base<F> LogDetDiv
( UpperOrLower uplo, const DistMatrix<F>& A, const DistMatrix<F>& B );

// Non-negative matrix factorization
// =================================
// TODO: Generalize to complex
template<typename Real>
void NMF( const Matrix<Real>& A, Matrix<Real>& X, Matrix<Real>& Y );
template<typename Real>
void NMF( const DistMatrix<Real>& A, DistMatrix<Real>& X, DistMatrix<Real>& Y );

// Non-negative least squares
// ==========================
// TODO: Generalize to complex
template<typename Real>
Int NonNegativeLeastSquares
( const Matrix<Real>& A, const Matrix<Real>& Y, Matrix<Real>& Z,
  Real rho=1., Real alpha=1.2, Int maxIter=500, Real absTol=1e-6,
  Real relTol=1e-4, bool inv=true, bool progress=true );
template<typename Real>
Int NonNegativeLeastSquares
( const DistMatrix<Real>& A, const DistMatrix<Real>& Y, DistMatrix<Real>& Z, 
  Real rho=1., Real alpha=1.2, Int maxIter=500, Real absTol=1e-6,
  Real relTol=1e-4, bool inv=true, bool progress=true );

// Quadratic program
// =================
// TODO: Generalize to complex
template<typename Real>
Int QuadraticProgram
( const Matrix<Real>& P, const Matrix<Real>& S, Real lb, Real ub,
  Matrix<Real>& X, Matrix<Real>& Z, Matrix<Real>& U,
  Real rho=1., Real alpha=1.2, Int maxIter=500, Real absTol=1e-6,
  Real relTol=1e-4, bool inv=true, bool progress=true );
template<typename Real>
Int QuadraticProgram
( const DistMatrix<Real>& P, const DistMatrix<Real>& S, Real lb, Real ub,
  DistMatrix<Real>& X, DistMatrix<Real>& Z, DistMatrix<Real>& U,
  Real rho=1., Real alpha=1.2, Int maxIter=500, Real absTol=1e-6,
  Real relTol=1e-4, bool inv=true, bool progress=true );

// Soft-thresholding
// =================
template<typename F>
F SoftThreshold( F alpha, Base<F> tau );
template<typename F>
void SoftThreshold( Matrix<F>& A, Base<F> tau, bool relative=false );
template<typename F,Dist U,Dist V>
void SoftThreshold( DistMatrix<F,U,V>& A, Base<F> tau, bool relative=false );

// Sparse inverse covariance selection
// ===================================
template<typename F>
Int SparseInvCov
( const Matrix<F>& D, Matrix<F>& X, Matrix<F>& Z, Matrix<F>& U,
  Base<F> lambda, Base<F> rho=1., Base<F> alpha=1.2, Int maxIter=500,
  Base<F> absTol=1e-6, Base<F> relTol=1e-4, bool progress=true );
template<typename F>
Int SparseInvCov
( const DistMatrix<F>& D, DistMatrix<F>& X, DistMatrix<F>& Z, DistMatrix<F>& U,
  Base<F> lambda, Base<F> rho=1., Base<F> alpha=1.2, Int maxIter=500,
  Base<F> absTol=1e-6, Base<F> relTol=1e-4, bool progress=true );

// Singular-value soft thresholding
// ================================
template<typename F>
Int SVT( Matrix<F>& A, Base<F> tau, bool relative=false );
template<typename F>
Int SVT( DistMatrix<F>& A, Base<F> tau, bool relative=false );
template<typename F>
Int SVT( Matrix<F>& A, Base<F> tau, Int relaxedRank, bool relative=false );
template<typename F>
Int SVT( DistMatrix<F>& A, Base<F> tau, Int relaxedRank, bool relative=false );
template<typename F,Dist U>
Int SVT( DistMatrix<F,U,STAR>& A, Base<F> tau, bool relative=false );

namespace svt {

template<typename F>
Int Cross( Matrix<F>& A, Base<F> tau, bool relative=false );
template<typename F>
Int Cross( DistMatrix<F>& A, Base<F> tau, bool relative=false );
template<typename F>
Int TallCross( DistMatrix<F,VC,STAR>& A, Base<F> tau, bool relative=false );

template<typename F>
Int Normal( Matrix<F>& A, Base<F> tau, bool relative=false );
template<typename F>
Int Normal( DistMatrix<F>& A, Base<F> tau, bool relative=false );

template<typename F>
Int PivotedQR
( Matrix<F>& A, Base<F> tau, Int numSteps, bool relative=false );
template<typename F>
Int PivotedQR
( DistMatrix<F>& A, Base<F> tau, Int numSteps, bool relative=false );

template<typename F,Dist U>
Int TSQR( DistMatrix<F,U,STAR>& A, Base<F> tau, bool relative=false );

} // namespace svt

} // namespace El

#endif // ifndef EL_OPTIMIZATION_HPP
