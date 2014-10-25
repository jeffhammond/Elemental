/*
   Copyright (c) 2009-2014, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_BLAS1_HPP
#define EL_BLAS1_HPP

namespace El {

// Adjoint
// =======
template<typename T>
void Adjoint( const Matrix<T>& A, Matrix<T>& B );
template<typename T>
void Adjoint( const AbstractDistMatrix<T>& A, AbstractDistMatrix<T>& B );
template<typename T>
void Adjoint
( const AbstractBlockDistMatrix<T>& A, AbstractBlockDistMatrix<T>& B );

// Axpy
// ====
template<typename T,typename S>
void Axpy( S alpha, const Matrix<T>& X, Matrix<T>& Y );
template<typename T,typename S>
void Axpy( S alpha, const AbstractDistMatrix<T>& X, AbstractDistMatrix<T>& Y );
template<typename T,typename S>
void Axpy( S alpha, const DistMultiVec<T>& X, DistMultiVec<T>& Y );

// AxpyTriangle
// ============
template<typename T,typename S>
void AxpyTriangle
( UpperOrLower uplo, S alpha, const Matrix<T>& X, Matrix<T>& Y );
template<typename T,typename S>
void AxpyTriangle
( UpperOrLower uplo, S alpha,
  const AbstractDistMatrix<T>& X, AbstractDistMatrix<T>& Y );

// Conjugate
// =========
template<typename Real>
void Conjugate( Matrix<Real>& A );
template<typename Real>
void Conjugate( Matrix<Complex<Real>>& A );

template<typename T>
void Conjugate( const Matrix<T>& A, Matrix<T>& B );

template<typename T>
void Conjugate( AbstractDistMatrix<T>& A );
template<typename T>
void Conjugate( const AbstractDistMatrix<T>& A, AbstractDistMatrix<T>& B );

// Copy
// ====
template<typename T>
void Copy( const Matrix<T>& A, Matrix<T>& B );
// TODO: A detailed description of which conversions are instantiated 
template<typename S,typename T>
void Copy( const Matrix<S>& A, Matrix<T>& B );
template<typename S,typename T>
void Copy( const AbstractDistMatrix<S>& A, AbstractDistMatrix<T>& B );
template<typename S,typename T>
void Copy( const AbstractBlockDistMatrix<S>& A, AbstractBlockDistMatrix<T>& B );

void Copy( const Graph& A, Graph& B );
void Copy( const DistGraph& A, DistGraph& B );
void CopyFromRoot( const DistGraph& distGraph, Graph& graph );
void CopyFromNonRoot( const DistGraph& distGraph, Int root=0 );

template<typename T>
void Copy( const SparseMatrix<T>& A, SparseMatrix<T>& B );
template<typename T>
void Copy( const DistSparseMatrix<T>& A, DistSparseMatrix<T>& B );
template<typename T>
void CopyFromRoot( const DistSparseMatrix<T>& ADist, SparseMatrix<T>& A );
template<typename T>
void CopyFromNonRoot( const DistSparseMatrix<T>& ADist, Int root=0 );

template<typename T>
void Copy( const DistMultiVec<T>& A, DistMultiVec<T>& B );
template<typename T>
void CopyFromRoot( const DistMultiVec<T>& XDist, Matrix<T>& X );
template<typename T>
void CopyFromNonRoot( const DistMultiVec<T>& XDist, Int root=0 );

// DiagonalScale
// =============
template<typename TDiag,typename T>
void DiagonalScale
( LeftOrRight side, Orientation orientation,
  const Matrix<TDiag>& d, Matrix<T>& X );

template<typename TDiag,typename T,Dist U,Dist V>
void DiagonalScale
( LeftOrRight side, Orientation orientation,
  const AbstractDistMatrix<TDiag>& d, DistMatrix<T,U,V>& X );

template<typename T>
void DiagonalScale
( LeftOrRight side, Orientation orientation,
  const AbstractDistMatrix<T>& d, AbstractDistMatrix<T>& X );

template<typename Real>
void DiagonalScale
( LeftOrRight side, Orientation orientation,
  const AbstractDistMatrix<Real>& d, AbstractDistMatrix<Complex<Real>>& X );

// DiagonalScaleTrapezoid
// ======================
template<typename TDiag,typename T>
void DiagonalScaleTrapezoid
( LeftOrRight side, UpperOrLower uplo, Orientation orientation,
  const Matrix<TDiag>& d, Matrix<T>& A, Int offset=0 );

template<typename TDiag,typename T,Dist U,Dist V>
void DiagonalScaleTrapezoid
( LeftOrRight side, UpperOrLower uplo, Orientation orientation,
  const AbstractDistMatrix<TDiag>& d, DistMatrix<T,U,V>& A, Int offset=0 );

template<typename T>
void DiagonalScaleTrapezoid
( LeftOrRight side, UpperOrLower uplo, Orientation orientation,
  const AbstractDistMatrix<T>& d, AbstractDistMatrix<T>& A, Int offset=0 );

template<typename Real>
void DiagonalScaleTrapezoid
( LeftOrRight side, UpperOrLower uplo, Orientation orientation,
  const AbstractDistMatrix<Real>& d,
        AbstractDistMatrix<Complex<Real>>& A, Int offset=0 );

// DiagonalSolve
// =============
template<typename FDiag,typename F>
void DiagonalSolve
( LeftOrRight side, Orientation orientation,
  const Matrix<FDiag>& d, Matrix<F>& X, bool checkIfSingular=true );

template<typename FDiag,typename F,Dist U,Dist V>
void DiagonalSolve
( LeftOrRight side, Orientation orientation,
  const AbstractDistMatrix<FDiag>& d, DistMatrix<F,U,V>& X,
  bool checkIfSingular=true );

template<typename F>
void DiagonalSolve
( LeftOrRight side, Orientation orientation,
  const AbstractDistMatrix<F>& d, AbstractDistMatrix<F>& X,
  bool checkIfSingular=true );

template<typename F>
void DiagonalSolve
( LeftOrRight side, Orientation orientation,
  const AbstractDistMatrix<F>& d, AbstractDistMatrix<Complex<F>>& X,
  bool checkIfSingular=true );

// Dot
// ===
template<typename T>
T Dot( const Matrix<T>& A, const Matrix<T>& B );
template<typename T>
T Dot( const AbstractDistMatrix<T>& A, const AbstractDistMatrix<T>& B );

// Dotu
// ====
template<typename T>
T Dotu( const Matrix<T>& A, const Matrix<T>& B );
template<typename T>
T Dotu( const AbstractDistMatrix<T>& A, const AbstractDistMatrix<T>& B );

// EntrywiseFill
// =============
template<typename T>
void EntrywiseFill( Matrix<T>& A, std::function<T(void)> func );
template<typename T>
void EntrywiseFill( AbstractDistMatrix<T>& A, std::function<T(void)> func );
template<typename T>
void EntrywiseFill
( AbstractBlockDistMatrix<T>& A, std::function<T(void)> func );

// EntrywiseMap
// ============
template<typename T>
void EntrywiseMap( Matrix<T>& A, std::function<T(T)> func );
template<typename T>
void EntrywiseMap( AbstractDistMatrix<T>& A, std::function<T(T)> func );
template<typename T>
void EntrywiseMap( AbstractBlockDistMatrix<T>& A, std::function<T(T)> func );

template<typename S,typename T>
void EntrywiseMap( const Matrix<S>& A, Matrix<T>& B, std::function<T(S)> func );
template<typename S,typename T>
void EntrywiseMap
( const AbstractDistMatrix<S>& A, AbstractDistMatrix<T>& B, 
  std::function<T(S)> func );
template<typename S,typename T>
void EntrywiseMap
( const AbstractBlockDistMatrix<S>& A, AbstractBlockDistMatrix<T>& B, 
  std::function<T(S)> func );

// Fill
// ====
template<typename T>
void Fill( Matrix<T>& A, T alpha );
template<typename T>
void Fill( AbstractDistMatrix<T>& A, T alpha );
template<typename T>
void Fill( AbstractBlockDistMatrix<T>& A, T alpha );

// Hadamard
// ========
template<typename T>
void Hadamard( const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C );
template<typename T>
void Hadamard
( const AbstractDistMatrix<T>& A,
  const AbstractDistMatrix<T>& B,
        AbstractDistMatrix<T>& C );

// HilbertSchmidt
// ==============
template<typename T>
T HilbertSchmidt( const Matrix<T>& A, const Matrix<T>& B );
template<typename T>
T HilbertSchmidt
( const AbstractDistMatrix<T>& A, const AbstractDistMatrix<T>& C );

// Imaginary part
// ==============
template<typename T>
void ImagPart
( const Matrix<T>& A, Matrix<Base<T>>& AImag );
template<typename T>
void ImagPart
( const AbstractDistMatrix<T>& A, AbstractDistMatrix<Base<T>>& AImag );

// IndexDependentFill
// ==================
template<typename T>
void IndexDependentFill( Matrix<T>& A, std::function<T(Int,Int)> func );
template<typename T>
void IndexDependentFill
( AbstractDistMatrix<T>& A, std::function<T(Int,Int)> func );
template<typename T>
void IndexDependentFill
( AbstractBlockDistMatrix<T>& A, std::function<T(Int,Int)> func );

// IndexDependentMap
// =================
template<typename T>
void IndexDependentMap( Matrix<T>& A, std::function<T(Int,Int,T)> func );
template<typename T>
void IndexDependentMap
( AbstractDistMatrix<T>& A, std::function<T(Int,Int,T)> func );
template<typename T>
void IndexDependentMap
( AbstractBlockDistMatrix<T>& A, std::function<T(Int,Int,T)> func );

template<typename S,typename T>
void IndexDependentMap
( const Matrix<S>& A, Matrix<T>& B, std::function<T(Int,Int,S)> func );
template<typename S,typename T>
void IndexDependentMap
( const AbstractDistMatrix<S>& A, AbstractDistMatrix<T>& B,
  std::function<T(Int,Int,S)> func );
template<typename S,typename T>
void IndexDependentMap
( const AbstractBlockDistMatrix<S>& A, AbstractBlockDistMatrix<T>& B,
  std::function<T(Int,Int,S)> func );

// MakeHermitian
// =============
template<typename T>
void MakeHermitian( UpperOrLower uplo, Matrix<T>& A );
template<typename T>
void MakeHermitian( UpperOrLower uplo, AbstractDistMatrix<T>& A );

// MakeReal
// ========
template<typename Real>
void MakeReal( Matrix<Real>& A );
template<typename Real>
void MakeReal( Matrix<Complex<Real>>& A );
template<typename T>
void MakeReal( AbstractDistMatrix<T>& A );

// MakeSymmetric
// =============
template<typename T>
void MakeSymmetric( UpperOrLower uplo, Matrix<T>& A, bool conjugate=false );
template<typename F>
void MakeSymmetric
( UpperOrLower uplo, AbstractDistMatrix<F>& A, bool conjugate=false );

// MakeTrapezoidal
// ===============
template<typename T>
void MakeTrapezoidal( UpperOrLower uplo, Matrix<T>& A, Int offset=0 );
template<typename T>
void MakeTrapezoidal
( UpperOrLower uplo, AbstractDistMatrix<T>& A, Int offset=0 );
template<typename T>
void MakeTrapezoidal
( UpperOrLower uplo, AbstractBlockDistMatrix<T>& A, Int offset=0 );

// MakeTriangular
// ==============
template<typename T>
void MakeTriangular( UpperOrLower uplo, Matrix<T>& A );
template<typename T>
void MakeTriangular( UpperOrLower uplo, AbstractDistMatrix<T>& A );
template<typename T>
void MakeTriangular( UpperOrLower uplo, AbstractBlockDistMatrix<T>& A );

// Max
// ===
template<typename Real>
ValueIntPair<Real> Max( const Matrix<Real>& A );
template<typename Real>
ValueIntPair<Real> Max( const AbstractDistMatrix<Real>& A );

template<typename Real>
ValueIntPair<Real> SymmetricMax( UpperOrLower uplo, const Matrix<Real>& A );
template<typename Real>
ValueIntPair<Real>
SymmetricMax( UpperOrLower uplo, const AbstractDistMatrix<Real>& A );

template<typename Real>
ValueInt<Real> VectorMax( const Matrix<Real>& x );
template<typename Real>
ValueInt<Real> VectorMax( const AbstractDistMatrix<Real>& x );

// MaxAbs
// ======
template<typename F>
ValueInt<Base<F>> VectorMaxAbs( const Matrix<F>& x );
template<typename F>
ValueInt<Base<F>> VectorMaxAbs( const AbstractDistMatrix<F>& x );

template<typename F>
ValueIntPair<Base<F>> MaxAbs( const Matrix<F>& A );
template<typename F>
ValueIntPair<Base<F>> MaxAbs( const AbstractDistMatrix<F>& A );

template<typename F>
ValueIntPair<Base<F>> SymmetricMaxAbs( UpperOrLower uplo, const Matrix<F>& A );
template<typename F>
ValueIntPair<Base<F>> SymmetricMaxAbs
( UpperOrLower uplo, const AbstractDistMatrix<F>& A );

// Min
// ===
template<typename Real>
ValueInt<Real> VectorMin( const Matrix<Real>& x );
template<typename Real>
ValueInt<Real> VectorMin( const AbstractDistMatrix<Real>& x );

template<typename Real>
ValueIntPair<Real> Min( const Matrix<Real>& A );
template<typename Real>
ValueIntPair<Real> Min( const AbstractDistMatrix<Real>& A );

template<typename Real>
ValueIntPair<Real> SymmetricMin( UpperOrLower uplo, const Matrix<Real>& A );
template<typename Real>
ValueIntPair<Real>
SymmetricMin( UpperOrLower uplo, const AbstractDistMatrix<Real>& A );

// MinAbs
// ======
template<typename F>
ValueInt<Base<F>> VectorMinAbs( const Matrix<F>& x );
template<typename F>
ValueInt<Base<F>> VectorMinAbs( const AbstractDistMatrix<F>& x );

template<typename F>
ValueIntPair<Base<F>> MinAbs( const Matrix<F>& A );
template<typename F>
ValueIntPair<Base<F>> MinAbs( const AbstractDistMatrix<F>& A );

template<typename F>
ValueIntPair<Base<F>> SymmetricMinAbs( UpperOrLower uplo, const Matrix<F>& A );
template<typename F>
ValueIntPair<Base<F>>
SymmetricMinAbs( UpperOrLower uplo, const AbstractDistMatrix<F>& A );

// Nrm2
// ====
template<typename F>
Base<F> Nrm2( const Matrix<F>& x );
template<typename F>
Base<F> Nrm2( const AbstractDistMatrix<F>& x );

// QuasiDiagonalScale
// ==================
template<typename F,typename FMain>
void QuasiDiagonalScale
( LeftOrRight side, UpperOrLower uplo,
  const Matrix<FMain>& d, const Matrix<F>& dSub,
  Matrix<F>& X, bool conjugated=false );
// TODO: Switch to full AbstractDistMatrix interface
template<typename F,typename FMain,Dist U,Dist V>
void QuasiDiagonalScale
( LeftOrRight side, UpperOrLower uplo,
  const AbstractDistMatrix<FMain>& d, const AbstractDistMatrix<F>& dSub,
  DistMatrix<F,U,V>& X, bool conjugated=false );

template<typename F,typename FMain,Dist U,Dist V>
void LeftQuasiDiagonalScale
( UpperOrLower uplo,
  const DistMatrix<FMain,U,STAR>& d,
  const DistMatrix<FMain,U,STAR>& dPrev,
  const DistMatrix<FMain,U,STAR>& dNext,
  const DistMatrix<F,    U,STAR>& dSub,
  const DistMatrix<F,    U,STAR>& dSubPrev,
  const DistMatrix<F,    U,STAR>& dSubNext,
        DistMatrix<F,U,V>& X,
  const DistMatrix<F,U,V>& XPrev,
  const DistMatrix<F,U,V>& XNext,
  bool conjugated=false );

template<typename F,typename FMain,Dist U,Dist V>
void RightQuasiDiagonalScale
( UpperOrLower uplo,
  const DistMatrix<FMain,V,STAR>& d,
  const DistMatrix<FMain,V,STAR>& dPrev,
  const DistMatrix<FMain,V,STAR>& dNext,
  const DistMatrix<F,    V,STAR>& dSub,
  const DistMatrix<F,    V,STAR>& dSubPrev,
  const DistMatrix<F,    V,STAR>& dSubNext,
        DistMatrix<F,U,V>& X,
  const DistMatrix<F,U,V>& XPrev,
  const DistMatrix<F,U,V>& XNext,
  bool conjugated=false );

// QuasiDiagonalSolve
// ==================
template<typename F,typename FMain>
void
QuasiDiagonalSolve
( LeftOrRight side, UpperOrLower uplo,
  const Matrix<FMain>& d, const Matrix<F>& dSub,
  Matrix<F>& X, bool conjugated=false );
// TODO: Switch to full AbstractDistMatrix interface
template<typename F,typename FMain,Dist U,Dist V>
void
QuasiDiagonalSolve
( LeftOrRight side, UpperOrLower uplo,
  const AbstractDistMatrix<FMain>& d, const AbstractDistMatrix<F>& dSub,
  DistMatrix<F,U,V>& X, bool conjugated=false );

template<typename F,typename FMain,Dist U,Dist V>
void
LeftQuasiDiagonalSolve
( UpperOrLower uplo,
  const DistMatrix<FMain,U,STAR>& d,
  const DistMatrix<FMain,U,STAR>& dPrev,
  const DistMatrix<FMain,U,STAR>& dNext,
  const DistMatrix<F,    U,STAR>& dSub,
  const DistMatrix<F,    U,STAR>& dSubPrev,
  const DistMatrix<F,    U,STAR>& dSubNext,
        DistMatrix<F,U,V>& X,
  const DistMatrix<F,U,V>& XPrev,
  const DistMatrix<F,U,V>& XNext,
  bool conjugated=false );

template<typename F,typename FMain,Dist U,Dist V>
void
RightQuasiDiagonalSolve
( UpperOrLower uplo,
  const DistMatrix<FMain,V,STAR>& d,
  const DistMatrix<FMain,V,STAR>& dPrev,
  const DistMatrix<FMain,V,STAR>& dNext,
  const DistMatrix<F,    V,STAR>& dSub,
  const DistMatrix<F,    V,STAR>& dSubPrev,
  const DistMatrix<F,    V,STAR>& dSubNext,
        DistMatrix<F,U,V>& X,
  const DistMatrix<F,U,V>& XPrev,
  const DistMatrix<F,U,V>& XNext,
  bool conjugated=false );

// Real part
// =========
template<typename T>
void RealPart
( const Matrix<T>& A, Matrix<Base<T>>& AReal );
template<typename T>
void RealPart
( const AbstractDistMatrix<T>& A, AbstractDistMatrix<Base<T>>& AReal );

// Scale
// =====
template<typename T,typename S>
void Scale( S alpha, Matrix<T>& A );
template<typename T,typename S>
void Scale( S alpha, AbstractDistMatrix<T>& A );
template<typename T,typename S>
void Scale( S alpha, AbstractBlockDistMatrix<T>& A );

template<typename Real,typename S>
void Scale( S alpha, Matrix<Real>& AReal, Matrix<Real>& AImag );
template<typename Real,typename S>
void Scale
( S alpha, AbstractDistMatrix<Real>& AReal, AbstractDistMatrix<Real>& AImag );
template<typename Real,typename S>
void Scale
( S alpha, AbstractBlockDistMatrix<Real>& AReal,
           AbstractBlockDistMatrix<Real>& AImag );

// ScaleTrapezoid
// ==============
template<typename T,typename S>
void ScaleTrapezoid( S alpha, UpperOrLower uplo, Matrix<T>& A, Int offset=0 );
template<typename T,typename S>
void ScaleTrapezoid
( S alpha, UpperOrLower uplo, AbstractDistMatrix<T>& A, Int offset=0 );

// SetDiagonal
// ===========
template<typename T,typename S>
void SetDiagonal( Matrix<T>& A, S alpha, Int offset=0 );
template<typename T,typename S>
void SetDiagonal( AbstractDistMatrix<T>& A, S alpha, Int offset=0 );
template<typename T,typename S>
void SetDiagonal( AbstractBlockDistMatrix<T>& A, S alpha, Int offset=0 );

// Swap
// ====
template<typename T>
void Swap( Orientation orientation, Matrix<T>& X, Matrix<T>& Y );
template<typename T>
void Swap
( Orientation orientation, AbstractDistMatrix<T>& X, AbstractDistMatrix<T>& Y );

template<typename T>
void RowSwap( Matrix<T>& A, Int to, Int from );
template<typename T>
void RowSwap( AbstractDistMatrix<T>& A, Int to, Int from );

template<typename T>
void ColSwap( Matrix<T>& A, Int to, Int from );
template<typename T>
void ColSwap( AbstractDistMatrix<T>& A, Int to, Int from );

template<typename T>
void SymmetricSwap
( UpperOrLower uplo, Matrix<T>& A, Int to, Int from, bool conjugate=false );
template<typename T>
void SymmetricSwap
( UpperOrLower uplo, AbstractDistMatrix<T>& A, Int to, Int from,
  bool conjugate=false );

template<typename T>
void HermitianSwap( UpperOrLower uplo, Matrix<T>& A, Int to, Int from );
template<typename T>
void HermitianSwap
( UpperOrLower uplo, AbstractDistMatrix<T>& A, Int to, Int from );

// Symmetric2x2Inv
// ===============
template<typename F>
void Symmetric2x2Inv( UpperOrLower uplo, Matrix<F>& D, bool conjugate=false );

// Symmetric2x2Scale
// =================
template<typename F>
void
Symmetric2x2Scale
( LeftOrRight side, UpperOrLower uplo,
  const Matrix<F>& D, Matrix<F>& A, bool conjugate=false );
template<typename F>
void
Symmetric2x2Scale
( LeftOrRight side, UpperOrLower uplo,
  const AbstractDistMatrix<F>& D, AbstractDistMatrix<F>& A,
  bool conjugate=false );

template<typename F>
void
FirstHalfOfSymmetric2x2Scale
( LeftOrRight side, UpperOrLower uplo,
  const Matrix<F>& D, Matrix<F>& a1, const Matrix<F>& a2,
  bool conjugate=false );

template<typename F>
void
SecondHalfOfSymmetric2x2Scale
( LeftOrRight side, UpperOrLower uplo,
  const Matrix<F>& D, const Matrix<F>& a1, Matrix<F>& a2,
  bool conjugate=false );

// Symmetric2x2Solve
// =================
template<typename F>
void
Symmetric2x2Solve
( LeftOrRight side, UpperOrLower uplo,
  const Matrix<F>& D, Matrix<F>& A, bool conjugate=false );
template<typename F>
void
Symmetric2x2Solve
( LeftOrRight side, UpperOrLower uplo,
  const AbstractDistMatrix<F>& D, AbstractDistMatrix<F>& A,
  bool conjugate=false );

template<typename F>
void
FirstHalfOfSymmetric2x2Solve
( LeftOrRight side, UpperOrLower uplo,
  const Matrix<F>& D, Matrix<F>& a1, const Matrix<F>& a2,
  bool conjugate=false );

template<typename F>
void
SecondHalfOfSymmetric2x2Solve
( LeftOrRight side, UpperOrLower uplo,
  const Matrix<F>& D, const Matrix<F>& a1, Matrix<F>& a2,
  bool conjugate=false );

// Transpose
// =========
template<typename T>
void Transpose( const Matrix<T>& A, Matrix<T>& B, bool conjugate=false );
template<typename T>
void Transpose
( const AbstractDistMatrix<T>& A, AbstractDistMatrix<T>& B,
  bool conjugate=false );
template<typename T>
void Transpose
( const AbstractBlockDistMatrix<T>& A, AbstractBlockDistMatrix<T>& B,
  bool conjugate=false );

// UpdateDiagonal
// ==============
template<typename T,typename S>
void UpdateDiagonal( Matrix<T>& A, S alpha, Int offset=0 );
template<typename T,typename S>
void UpdateDiagonal( AbstractDistMatrix<T>& A, S alpha, Int offset=0 );
template<typename T,typename S>
void UpdateDiagonal( AbstractBlockDistMatrix<T>& A, S alpha, Int offset=0 );

// Zero
// ====
template<typename T>
void Zero( Matrix<T>& A );
template<typename T>
void Zero( AbstractDistMatrix<T>& A );
template<typename T>
void Zero( AbstractBlockDistMatrix<T>& A );
template<typename T>
void Zero( DistMultiVec<T>& A );

} // namespace El

#endif // ifndef EL_BLAS1_HPP
