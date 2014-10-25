/*
   Copyright (c) 2009-2014, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_IMPORTS_MPI_C_H
#define EL_IMPORTS_MPI_C_H

#ifdef __cplusplus
extern "C" {
#endif

EL_EXPORT ElError ElMPICommIsVoidPointer( bool* isVoidP );
EL_EXPORT ElError ElMPIGroupIsVoidPointer( bool* isVoidP );
/* These may seem trivial but is useful when calling C from another language */
EL_EXPORT ElError ElMPICommWorld( MPI_Comm* commWorld );
EL_EXPORT ElError ElMPICommSelf( MPI_Comm* commSelf );

EL_EXPORT ElError ElMPICommRank( MPI_Comm comm, int* rank );
EL_EXPORT ElError ElMPIGroupRank( MPI_Group group, int* rank );

EL_EXPORT ElError ElMPICommSize( MPI_Comm comm, int* size );
EL_EXPORT ElError ElMPIGroupSize( MPI_Group group, int* size );

EL_EXPORT ElError ElMPICommFree( MPI_Comm* comm );
EL_EXPORT ElError ElMPIGroupFree( MPI_Group* group );

EL_EXPORT ElError ElMPIWorldRank( int* rank );

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* ifndef EL_IMPORTS_MPI_C_H */
