/*
   Copyright (c) 2009-2014, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

namespace El {

// Public section
// ##############

// Constructors and destructors
// ============================

template<typename T,Dist U,Dist V>
GeneralDistMatrix<T,U,V>::GeneralDistMatrix( const El::Grid& grid, Int root )
: AbstractDistMatrix<T>(grid,root)
{ }

template<typename T,Dist U,Dist V>
GeneralDistMatrix<T,U,V>::GeneralDistMatrix( GeneralDistMatrix<T,U,V>&& A ) 
EL_NOEXCEPT
: AbstractDistMatrix<T>(std::move(A))
{ }

// Assignment and reconfiguration
// ==============================

template<typename T,Dist U,Dist V>
GeneralDistMatrix<T,U,V>& 
GeneralDistMatrix<T,U,V>::operator=( GeneralDistMatrix<T,U,V>&& A )
{
    AbstractDistMatrix<T>::operator=( std::move(A) );
    return *this;
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::AlignColsWith
( const El::DistData& data, bool constrain, bool allowMismatch )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::AlignColsWith")) 
    this->SetGrid( *data.grid );
    this->SetRoot( data.root );
    if( data.colDist == U || data.colDist == UPart )
        this->AlignCols( data.colAlign, constrain );
    else if( data.rowDist == U || data.rowDist == UPart )
        this->AlignCols( data.rowAlign, constrain );
    else if( data.colDist == UScat )
        this->AlignCols( data.colAlign % this->ColStride(), constrain );
    else if( data.rowDist == UScat )
        this->AlignCols( data.rowAlign % this->ColStride(), constrain );
    else if( U != UGath && data.colDist != UGath && data.rowDist != UGath &&
            !allowMismatch ) 
        LogicError("Nonsensical alignment");
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::AlignRowsWith
( const El::DistData& data, bool constrain, bool allowMismatch )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::AlignRowsWith")) 
    this->SetGrid( *data.grid );
    this->SetRoot( data.root );
    if( data.colDist == V || data.colDist == VPart )
        this->AlignRows( data.colAlign, constrain );
    else if( data.rowDist == V || data.rowDist == VPart )
        this->AlignRows( data.rowAlign, constrain );
    else if( data.colDist == VScat )
        this->AlignRows( data.colAlign % this->RowStride(), constrain );
    else if( data.rowDist == VScat )
        this->AlignRows( data.rowAlign % this->RowStride(), constrain );
    else if( V != VGath && data.colDist != VGath && data.rowDist != VGath &&
             !allowMismatch ) 
        LogicError("Nonsensical alignment");
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::PartialColAllGather( DistMatrix<T,UPart,V>& A ) const
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::PartialColAllGather");
        AssertSameGrids( *this, A );
    )
    const Int height = this->Height();
    const Int width = this->Width();
#ifdef EL_VECTOR_WARNINGS
    if( width == 1 && this->Grid().Rank() == 0 )
    {
        std::cerr <<
          "The vector version of PartialColAllGather is not yet written but "
          "would only require modifying the vector version of "
          "PartialRowAllGather" << std::endl;
    }
#endif
#ifdef EL_CACHE_WARNINGS
    if( width && this->Grid().Rank() == 0 )
    {
        std::cerr <<
          "PartialColAllGather potentially causes a large amount of cache-"
          "thrashing. If possible, avoid it by performing the redistribution"
          "on the (conjugate-)transpose" << std::endl;
    }
#endif
    A.AlignColsAndResize
    ( this->ColAlign()%A.ColStride(), height, width, false, false );
    if( !this->Participating() )
        return;

    DEBUG_ONLY(
        if( this->LocalWidth() != this->Width() )
            LogicError("This routine assumes rows are not distributed");
    )
    const T* thisBuf = this->LockedBuffer();
    const Int ldim = this->LDim();
    T* ABuf = A.Buffer();
    const Int ALDim = A.LDim();

    const Int colAlign = this->ColAlign();
    const Int colAlignA = A.ColAlign();
    const Int colStride = this->ColStride();
    const Int colStrideUnion = this->PartialUnionColStride();
    const Int colStridePart = this->PartialColStride();
    const Int colRankPart = this->PartialColRank();
    const Int colShiftA = A.ColShift();

    const Int thisLocalHeight = this->LocalHeight();
    const Int maxLocalHeight = MaxLength(height,colStride);
    const Int portionSize = mpi::Pad( maxLocalHeight*width );
    T* buffer = A.auxMemory_.Require( (colStrideUnion+1)*portionSize );
    T* firstBuf = &buffer[0];
    T* secondBuf = &buffer[portionSize];

    if( colAlignA == colAlign % colStridePart ) 
    {
        // Pack
        EL_PARALLEL_FOR
        for( Int j=0; j<width; ++j )
        {
            const T* thisCol = &thisBuf[j*ldim];
            T* firstBufCol = &firstBuf[j*thisLocalHeight];
            MemCopy( firstBufCol, thisCol, thisLocalHeight );
        }

        // Communicate
        mpi::AllGather
        ( firstBuf, portionSize, secondBuf, portionSize, 
          this->PartialUnionColComm() );

        // Unpack
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<colStrideUnion; ++k )
        {
            const T* data = &secondBuf[k*portionSize];
            const Int colShift = 
                Shift_( colRankPart+k*colStridePart, colAlign, colStride );
            const Int colOffset = (colShift-colShiftA) / colStridePart;
            const Int localHeight = Length_( height, colShift, colStride );
            EL_INNER_PARALLEL_FOR
            for( Int j=0; j<width; ++j )
            {
                const T* dataCol = &data[j*localHeight];
                T* ACol = &ABuf[colOffset+j*ALDim];
                for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                    ACol[iLoc*colStrideUnion] = dataCol[iLoc];
            }
        }
    }
    else
    {
#ifdef EL_UNALIGNED_WARNINGS
        if( this->Grid().Rank() == 0 )
            std::cerr << "Unaligned PartialColAllGather" << std::endl;
#endif
        // Perform a SendRecv to match the row alignments
        const Int colRank = this->ColRank();
        const Int sendColRank = 
            (colRank+colStride+colAlignA-colAlign) % colStride;
        const Int recvColRank = 
            (colRank+colStride+colAlign-colAlignA) % colStride;
        EL_PARALLEL_FOR
        for( Int j=0; j<width; ++j ) 
        {
            const T* thisCol = &thisBuf[j*ldim];
            T* secondBufCol = &secondBuf[j*thisLocalHeight];
            MemCopy( secondBufCol, thisCol, thisLocalHeight );
        }
        mpi::SendRecv
        ( secondBuf, portionSize, sendColRank,
          firstBuf,  portionSize, recvColRank, this->ColComm() );

        // Use the SendRecv as an input to the partial union AllGather
        mpi::AllGather
        ( firstBuf,  portionSize, 
          secondBuf, portionSize, this->PartialUnionColComm() );

        // Unpack
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<colStrideUnion; ++k )
        {
            const T* data = &secondBuf[k*portionSize];
            const Int colShift = 
                Shift_( colRankPart+colStridePart*k, colAlignA, colStride );
            const Int colOffset = (colShift-colShiftA) / colStridePart;
            const Int localHeight = Length_( height, colShift, colStride );
            EL_INNER_PARALLEL_FOR
            for( Int j=0; j<width; ++j )
            {
                const T* dataCol = &data[j*localHeight];
                T* ACol = &ABuf[colOffset+j*ALDim];
                for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                    ACol[iLoc*colStrideUnion] = dataCol[iLoc];
            }
        }
    }
    A.auxMemory_.Release();
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::PartialRowAllGather( DistMatrix<T,U,VPart>& A ) const
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::PartialRowAllGather");
        AssertSameGrids( *this, A );
    )
    const Int height = this->Height();
    const Int width = this->Width();
    A.AlignRowsAndResize
    ( this->RowAlign()%A.RowStride(), height, width, false, false );
    if( !this->Participating() )
        return;

    DEBUG_ONLY(
        if( this->LocalHeight() != this->Height() )
            LogicError("This routine assumes columns are not distributed");
    )
    const T* thisBuf = this->LockedBuffer();
    const Int ldim = this->LDim();
    T* ABuf = A.Buffer();
    const Int ALDim = A.LDim();

    const Int rowAlign = this->RowAlign();
    const Int rowAlignA = A.RowAlign();
    const Int rowStride = this->RowStride();
    const Int rowStrideUnion = this->PartialUnionRowStride();
    const Int rowStridePart = this->PartialRowStride();
    const Int rowRankPart = this->PartialRowRank();
    const Int rowShiftA = A.RowShift();

    const Int thisLocalWidth = this->LocalWidth();
    const Int maxLocalWidth = MaxLength(width,rowStride);
    const Int portionSize = mpi::Pad( height*maxLocalWidth );
    T* buffer = A.auxMemory_.Require( (rowStrideUnion+1)*portionSize );
    T* firstBuf = &buffer[0];
    T* secondBuf = &buffer[portionSize];

    if( rowAlignA == rowAlign % rowStridePart ) 
    {
        // Pack
        EL_PARALLEL_FOR
        for( Int jLoc=0; jLoc<thisLocalWidth; ++jLoc )
        {
            const T* thisCol = &thisBuf[jLoc*ldim];
            T* firstBufCol = &firstBuf[jLoc*height];
            MemCopy( firstBufCol, thisCol, height );
        }

        // Communicate
        mpi::AllGather
        ( firstBuf, portionSize, secondBuf, portionSize, 
          this->PartialUnionRowComm() );

        // Unpack
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<rowStrideUnion; ++k )
        {
            const T* data = &secondBuf[k*portionSize];
            const Int rowShift = 
                Shift_( rowRankPart+k*rowStridePart, rowAlign, rowStride );
            const Int rowOffset = (rowShift-rowShiftA) / rowStridePart;
            const Int localWidth = Length_( width, rowShift, rowStride );
            EL_INNER_PARALLEL_FOR
            for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            {
                const T* dataCol = &data[jLoc*height];
                T* ACol = &ABuf[(rowOffset+jLoc*rowStrideUnion)*ALDim];
                MemCopy( ACol, dataCol, height );
            }
        }
    }
    else
    {
#ifdef EL_UNALIGNED_WARNINGS
        if( this->Grid().Rank() == 0 )
            std::cerr << "Unaligned PartialRowAllGather" << std::endl;
#endif
        // Perform a SendRecv to match the row alignments
        const Int rowRank = this->RowRank();
        const Int sendRowRank = 
            (rowRank+rowStride+rowAlignA-rowAlign) % rowStride;
        const Int recvRowRank = 
            (rowRank+rowStride+rowAlign-rowAlignA) % rowStride;
        EL_PARALLEL_FOR
        for( Int jLoc=0; jLoc<thisLocalWidth; ++jLoc ) 
        {
            const T* thisCol = &thisBuf[jLoc*ldim];
            T* secondBufCol = &secondBuf[jLoc*height];
            MemCopy( secondBufCol, thisCol, height );
        }
        mpi::SendRecv
        ( secondBuf, portionSize, sendRowRank,
          firstBuf,  portionSize, recvRowRank, this->RowComm() );

        // Use the SendRecv as an input to the partial union AllGather
        mpi::AllGather
        ( firstBuf,  portionSize, 
          secondBuf, portionSize, this->PartialUnionRowComm() );

        // Unpack
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<rowStrideUnion; ++k )
        {
            const T* data = &secondBuf[k*portionSize];
            const Int rowShift = 
                Shift_( rowRankPart+rowStridePart*k, rowAlignA, rowStride );
            const Int rowOffset = (rowShift-rowShiftA) / rowStridePart;
            const Int localWidth = Length_( width, rowShift, rowStride );
            EL_INNER_PARALLEL_FOR
            for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            {
                const T* dataCol = &data[jLoc*height];
                T* ACol = &ABuf[(rowOffset+jLoc*rowStrideUnion)*ALDim];
                MemCopy( ACol, dataCol, height );
            }
        }
    }
    A.auxMemory_.Release();
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::FilterFrom( const DistMatrix<T,UGath,VGath>& A )
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::FilterFrom");
        AssertSameGrids( *this, A );
    )
    const Int height = A.Height();
    const Int width = A.Width();
    this->Resize( height, width );
    if( !this->Participating() )
        return;

    const Int colStride = this->ColStride();
    const Int rowStride = this->RowStride();
    const Int colShift = this->ColShift();
    const Int rowShift = this->RowShift();

    const Int localHeight = this->LocalHeight();
    const Int localWidth = this->LocalWidth();
    
    T* thisBuf = this->Buffer();
    const Int ldim = this->LDim();
    const T* ABuf = A.LockedBuffer();
    const Int ALDim = A.LDim();
    EL_PARALLEL_FOR
    for( Int jLoc=0; jLoc<localWidth; ++jLoc )
    {
        T* thisCol = &thisBuf[jLoc*ldim];
        const T* ACol = &ABuf[colShift+(rowShift+jLoc*rowStride)*ALDim];
        for( Int iLoc=0; iLoc<localHeight; ++iLoc )
            thisCol[iLoc] = ACol[iLoc*colStride];
    }
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::ColFilterFrom( const DistMatrix<T,UGath,V>& A )
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::ColFilterFrom");
        AssertSameGrids( *this, A );
    )
    const Int height = A.Height();
    const Int width = A.Width();
    this->AlignRowsAndResize( A.RowAlign(), height, width, false, false );
    if( !this->Participating() )
        return;

    const Int colStride = this->ColStride();
    const Int colShift = this->ColShift();
    const Int rowAlign = this->RowAlign();
    const Int rowAlignA = A.RowAlign();

    const Int localHeight = this->LocalHeight();
    const Int localWidth = this->LocalWidth();
    
    T* thisBuf = this->Buffer();
    const Int ldim = this->LDim();
    const T* ABuf = A.LockedBuffer();
    const Int ALDim = A.LDim();

    if( rowAlign == rowAlignA )
    {
        EL_PARALLEL_FOR
        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
        {
            T* thisCol = &thisBuf[jLoc*ldim];
            const T* ACol = &ABuf[colShift+jLoc*ALDim];
            for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                thisCol[iLoc] = ACol[iLoc*colStride];
        }
    }
    else
    {
#ifdef EL_UNALIGNED_WARNINGS
        if( this->Grid().Rank() == 0 )
            std::cerr << "Unaligned ColFilterFrom" << std::endl;
#endif
        const Int rowStride = this->RowStride();
        const Int rowRank = this->RowRank();
        const Int sendRowRank = 
            (rowRank+rowStride+rowAlign-rowAlignA) % rowStride;
        const Int recvRowRank = 
            (rowRank+rowStride+rowAlignA-rowAlign) % rowStride;
        const Int localWidthA = A.LocalWidth();
        const Int sendSize = localHeight*localWidthA;
        const Int recvSize = localHeight*localWidth;
        T* buffer = this->auxMemory_.Require( sendSize+recvSize );
        T* sendBuf = &buffer[0];
        T* recvBuf = &buffer[sendSize];
        
        // Pack
        EL_PARALLEL_FOR
        for( Int jLoc=0; jLoc<localWidthA; ++jLoc )
        {
            T* sendCol = &sendBuf[jLoc*localHeight];
            const T* ACol = &ABuf[colShift+jLoc*ALDim];
            for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                sendCol[iLoc] = ACol[iLoc*colStride];
        }

        // Realign
        mpi::SendRecv
        ( sendBuf, sendSize, sendRowRank,
          recvBuf, recvSize, recvRowRank, this->RowComm() );

        // Unpack
        EL_PARALLEL_FOR
        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            MemCopy
            ( &thisBuf[jLoc*ldim], &recvBuf[jLoc*localHeight], localHeight );
        this->auxMemory_.Release();
    }
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::RowFilterFrom( const DistMatrix<T,U,VGath>& A )
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::RowFilterFrom");
        AssertSameGrids( *this, A );
    )
    const Int height = A.Height();
    const Int width = A.Width();
    this->AlignColsAndResize( A.ColAlign(), height, width, false, false );
    if( !this->Participating() )
        return;

    const Int colAlign = this->ColAlign();
    const Int colAlignA = A.ColAlign();
    const Int rowStride = this->RowStride();
    const Int rowShift = this->RowShift();

    const Int localHeight = this->LocalHeight();
    const Int localWidth = this->LocalWidth();
    
    T* thisBuf = this->Buffer();
    const Int ldim = this->LDim();
    const T* ABuf = A.LockedBuffer();
    const Int ALDim = A.LDim();
    
    if( colAlign == colAlignA )
    {
        EL_PARALLEL_FOR
        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
        {
            T* thisCol = &thisBuf[jLoc*ldim];
            const T* ACol = &ABuf[(rowShift+jLoc*rowStride)*ALDim];
            MemCopy( thisCol, ACol, localHeight );
        }
    }
    else
    {
#ifdef EL_UNALIGNED_WARNINGS
        if( this->Grid().Rank() == 0 )
            std::cerr << "Unaligned RowFilterFrom" << std::endl;
#endif
        const Int colRank = this->ColRank();
        const Int colStride = this->ColStride();
        const Int sendColRank = 
            (colRank+colStride+colAlign-colAlignA) % colStride;
        const Int recvColRank = 
            (colRank+colStride+colAlignA-colAlign) % colStride;
        const Int localHeightA = A.LocalHeight();
        const Int sendSize = localHeightA*localWidth;
        const Int recvSize = localHeight *localWidth;

        T* buffer = this->auxMemory_.Require( sendSize+recvSize );
        T* sendBuf = &buffer[0];
        T* recvBuf = &buffer[sendSize];

        // Pack
        EL_PARALLEL_FOR
        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            MemCopy
            ( &sendBuf[jLoc*localHeightA],
              &ABuf[(rowShift+jLoc*rowStride)*ALDim], localHeightA );

        // Realign
        mpi::SendRecv
        ( sendBuf, sendSize, sendColRank, 
          recvBuf, recvSize, recvColRank, this->ColComm() );

        // Unpack
        EL_PARALLEL_FOR
        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            MemCopy
            ( &thisBuf[jLoc*ldim], &recvBuf[jLoc*localHeight], localHeight );
        this->auxMemory_.Release();
    }
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::PartialColFilterFrom( const DistMatrix<T,UPart,V>& A )
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::PartialColFilterFrom");
        AssertSameGrids( *this, A );
    )
    const Int height = A.Height();
    const Int width = A.Width();
    this->AlignColsAndResize( A.ColAlign(), height, width, false, false );
    if( !this->Participating() )
        return;

    const Int colAlign = this->ColAlign();
    const Int colAlignA = A.ColAlign();
    const Int colStride = this->ColStride();
    const Int colStridePart = this->PartialColStride();
    const Int colStrideUnion = this->PartialUnionColStride();
    const Int colShiftA = A.ColShift();

    const Int localHeight = this->LocalHeight();

    T* thisBuf = this->Buffer();
    const Int ldim = this->LDim();
    const T* ABuf = A.LockedBuffer();
    const Int ALDim = A.LDim();
    if( colAlign % colStridePart == colAlignA )
    {
        const Int colShift = this->ColShift();
        const Int colOffset = (colShift-colShiftA) / colStridePart;
        EL_PARALLEL_FOR
        for( Int j=0; j<width; ++j )
        {
            T* thisCol = &thisBuf[j*ldim];
            const T* ACol = &ABuf[colOffset+j*ALDim];
            for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                thisCol[iLoc] = ACol[iLoc*colStrideUnion];
        }
    }
    else
    {
#ifdef EL_UNALIGNED_WARNINGS
        if( this->Grid().Rank() == 0 )
            std::cerr << "Unaligned PartialColFilterFrom" << std::endl;
#endif
        const Int colRankPart = this->PartialColRank();
        const Int colRankUnion = this->PartialUnionColRank();
        const Int colShiftA = A.ColShift();

        // Realign
        // -------
        const Int sendColRankPart = 
            (colRankPart+colStridePart+(colAlign%colStridePart)-colAlignA) % 
            colStridePart;
        const Int recvColRankPart =
            (colRankPart+colStridePart+colAlignA-(colAlign%colStridePart)) %
            colStridePart;
        const Int sendColRank = sendColRankPart + colStridePart*colRankUnion;
        const Int sendColShift = Shift( sendColRank, colAlign, colStride );
        const Int sendColOffset = (sendColShift-colShiftA) / colStridePart;
        const Int localHeightSend = Length( height, sendColShift, colStride );
        const Int sendSize = localHeightSend*width;
        const Int recvSize = localHeight    *width;
        T* buffer = this->auxMemory_.Require( sendSize+recvSize );
        T* sendBuf = &buffer[0];
        T* recvBuf = &buffer[sendSize];
        // Pack
        EL_PARALLEL_FOR
        for( Int j=0; j<width; ++j )
        {
            T* sendCol = &sendBuf[j*localHeightSend];
            const T* ACol = &ABuf[sendColOffset+j*ALDim];
            for( Int iLoc=0; iLoc<localHeightSend; ++iLoc )
                sendCol[iLoc] = ACol[iLoc*colStrideUnion];
        }
        // Change the column alignment
        mpi::SendRecv
        ( sendBuf, sendSize, sendColRankPart,
          recvBuf, recvSize, recvColRankPart, this->PartialColComm() );

        // Unpack
        // ------
        EL_PARALLEL_FOR
        for( Int j=0; j<width; ++j )
        {
            const T* recvCol = &recvBuf[j*localHeight];
            T* thisCol = &thisBuf[j*ldim];
            MemCopy( thisCol, recvCol, localHeight );
        }
        this->auxMemory_.Release();
    }
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::PartialRowFilterFrom( const DistMatrix<T,U,VPart>& A )
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::PartialRowFilterFrom");
        AssertSameGrids( *this, A );
    )
    const Int height = A.Height();
    const Int width = A.Width();
    this->AlignRowsAndResize( A.RowAlign(), height, width, false, false );
    if( !this->Participating() )
        return;

    const Int rowAlign = this->RowAlign();
    const Int rowAlignA = A.RowAlign();
    const Int rowStride = this->RowStride();
    const Int rowStridePart = this->PartialRowStride();
    const Int rowStrideUnion = this->PartialUnionRowStride();
    const Int rowShiftA = A.RowShift();

    const Int localWidth = this->LocalWidth();

    T* thisBuf = this->Buffer();
    const Int ldim = this->LDim();
    const T* ABuf = A.LockedBuffer();
    const Int ALDim = A.LDim();
    if( rowAlign % rowStridePart == rowAlignA )
    {
        const Int rowShift = this->RowShift();
        const Int rowOffset = (rowShift-rowShiftA) / rowStridePart;
        EL_PARALLEL_FOR
        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
        {
            T* thisCol = &thisBuf[jLoc*ldim];
            const T* ACol = &ABuf[(rowOffset+jLoc*rowStrideUnion)*ALDim];
            MemCopy( thisCol, ACol, height );
        }
    }
    else
    {
#ifdef EL_UNALIGNED_WARNINGS
        if( this->Grid().Rank() == 0 )
            std::cerr << "Unaligned PartialRowFilterFrom" << std::endl;
#endif
        const Int rowRankPart = this->PartialRowRank();
        const Int rowRankUnion = this->PartialUnionRowRank();
        const Int rowShiftA = A.RowShift();

        // Realign
        // -------
        const Int sendRowRankPart = 
            (rowRankPart+rowStridePart+(rowAlign%rowStridePart)-rowAlignA) % 
            rowStridePart;
        const Int recvRowRankPart =
            (rowRankPart+rowStridePart+rowAlignA-(rowAlign%rowStridePart)) %
            rowStridePart;
        const Int sendRowRank = sendRowRankPart + rowStridePart*rowRankUnion;
        const Int sendRowShift = Shift( sendRowRank, rowAlign, rowStride );
        const Int sendRowOffset = (sendRowShift-rowShiftA) / rowStridePart;
        const Int localWidthSend = Length( width, sendRowShift, rowStride );
        const Int sendSize = height*localWidthSend;
        const Int recvSize = height*localWidth;
        T* buffer = this->auxMemory_.Require( sendSize+recvSize );
        T* sendBuf = &buffer[0];
        T* recvBuf = &buffer[sendSize];
        // Pack
        EL_PARALLEL_FOR
        for( Int jLoc=0; jLoc<localWidthSend; ++jLoc )
        {
            T* sendCol = &sendBuf[jLoc*height];
            const T* ACol = &ABuf[(sendRowOffset+jLoc*rowStrideUnion)*ALDim];
            MemCopy( sendCol, ACol, height );
        }
        // Change the column alignment
        mpi::SendRecv
        ( sendBuf, sendSize, sendRowRankPart,
          recvBuf, recvSize, recvRowRankPart, this->PartialRowComm() );

        // Unpack
        // ------
        EL_PARALLEL_FOR
        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
        {
            const T* recvCol = &recvBuf[jLoc*height];
            T* thisCol = &thisBuf[jLoc*ldim];
            MemCopy( thisCol, recvCol, height );
        }
        this->auxMemory_.Release();
    }
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::PartialColAllToAllFrom
( const DistMatrix<T,UPart,VScat>& A )
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::PartialColAllToAllFrom");
        AssertSameGrids( *this, A );
    )
    const Int height = A.Height();
    const Int width = A.Width();
    this->AlignColsAndResize( A.ColAlign(), height, width, false, false );
    if( !this->Participating() )
        return;

    const Int colAlign = this->ColAlign();
    const Int colAlignA = A.ColAlign();
    const Int rowAlignA = A.RowAlign();

    const Int colStride = this->ColStride();
    const Int colStridePart = this->PartialColStride();
    const Int colStrideUnion = this->PartialUnionColStride();
    const Int colRankPart = this->PartialColRank();

    const Int colShiftA = A.ColShift();

    const Int thisLocalHeight = this->LocalHeight();
    const Int localWidthA = A.LocalWidth();
    const Int maxLocalHeight = MaxLength(height,colStride);
    const Int maxLocalWidth = MaxLength(width,colStrideUnion);
    const Int portionSize = mpi::Pad( maxLocalHeight*maxLocalWidth );

    T* thisBuf = this->Buffer();
    const Int ldim = this->LDim();
    const T* ABuf = A.LockedBuffer();
    const Int ALDim = A.LDim();

    T* buffer = this->auxMemory_.Require( 2*colStrideUnion*portionSize );
    T* firstBuf  = &buffer[0];
    T* secondBuf = &buffer[colStrideUnion*portionSize];

    if( colAlign % colStridePart == colAlignA )
    {
        // Pack            
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<colStrideUnion; ++k )
        {
            T* data = &firstBuf[k*portionSize];
            const Int colRank = colRankPart + k*colStridePart;
            const Int colShift = Shift_( colRank, colAlign, colStride );
            const Int colOffset = (colShift-colShiftA) / colStridePart;
            const Int localHeight = Length_( height, colShift, colStride );
            EL_INNER_PARALLEL_FOR
            for( Int jLoc=0; jLoc<localWidthA; ++jLoc )
            {
                T* dataCol = &data[jLoc*localHeight];
                const T* ACol = &ABuf[colOffset+jLoc*ALDim];
                for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                    dataCol[iLoc] = ACol[iLoc*colStrideUnion];
            }
        }

        // Simultaneously Scatter in columns and Gather in rows
        mpi::AllToAll
        ( firstBuf,  portionSize, 
          secondBuf, portionSize, this->PartialUnionColComm() );

        // Unpack
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<colStrideUnion; ++k )
        {
            const T* data = &secondBuf[k*portionSize];
            const Int rowShift = Shift_( k, rowAlignA, colStrideUnion );
            const Int localWidth = Length_( width, rowShift, colStrideUnion );
            EL_INNER_PARALLEL_FOR
            for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            {
                const T* dataCol = &data[jLoc*thisLocalHeight];
                T* thisCol = &thisBuf[(rowShift+jLoc*colStrideUnion)*ldim]; 
                MemCopy( thisCol, dataCol, thisLocalHeight );
            }
        }
    }
    else
    {
#ifdef EL_UNALIGNED_WARNINGS
        if( this->Grid().Rank() == 0 )
            std::cerr << "Unaligned PartialColAllToAllFrom" << std::endl;
#endif
        const Int sendColRankPart = 
            (colRankPart+colStridePart+(colAlign%colStridePart)-colAlignA) % 
            colStridePart;
        const Int recvColRankPart =
            (colRankPart+colStridePart+colAlignA-(colAlign%colStridePart)) %
            colStridePart; 

        // Pack
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<colStrideUnion; ++k )
        {
            T* data = &secondBuf[k*portionSize];
            const Int colRank = sendColRankPart + k*colStridePart;
            const Int colShift = Shift_( colRank, colAlign, colStride );
            const Int colOffset = (colShift-colShiftA) / colStridePart;
            const Int localHeight = Length_( height, colShift, colStride );
            EL_INNER_PARALLEL_FOR
            for( Int jLoc=0; jLoc<localWidthA; ++jLoc )
            {
                T* dataCol = &data[jLoc*localHeight];
                const T* ACol = &ABuf[colOffset+jLoc*ALDim];
                for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                    dataCol[iLoc] = ACol[iLoc*colStrideUnion];
            }
        }

        // Simultaneously Scatter in columns and Gather in rows
        mpi::AllToAll
        ( secondBuf, portionSize, 
          firstBuf,  portionSize, this->PartialUnionColComm() );

        // Realign the result
        mpi::SendRecv 
        ( firstBuf,  colStrideUnion*portionSize, sendColRankPart,
          secondBuf, colStrideUnion*portionSize, recvColRankPart, 
          this->PartialColComm() );

        // Unpack
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<colStrideUnion; ++k )
        {
            const T* data = &secondBuf[k*portionSize];
            const Int rowShift = Shift_( k, rowAlignA, colStrideUnion );
            const Int localWidth = Length_( width, rowShift, colStrideUnion );
            EL_INNER_PARALLEL_FOR
            for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            {
                const T* dataCol = &data[jLoc*thisLocalHeight];
                T* thisCol = &thisBuf[(rowShift+jLoc*colStrideUnion)*ldim]; 
                MemCopy( thisCol, dataCol, thisLocalHeight );
            }
        }
    }
    this->auxMemory_.Release();
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::PartialRowAllToAllFrom
( const DistMatrix<T,UScat,VPart>& A )
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::PartialRowAllToAllFrom");
        AssertSameGrids( *this, A );
    )
    const Int height = A.Height();
    const Int width = A.Width();
    this->AlignRowsAndResize( A.RowAlign(), height, width, false, false );
    if( !this->Participating() )
        return;

    const Int rowAlign = this->RowAlign();
    const Int rowAlignA = A.RowAlign();
    const Int colAlignA = A.ColAlign();

    const Int rowStride = this->RowStride();
    const Int rowStridePart = this->PartialRowStride();
    const Int rowStrideUnion = this->PartialUnionRowStride();
    const Int rowRankPart = this->PartialRowRank();

    const Int rowShiftA = A.RowShift();

    const Int thisLocalWidth = this->LocalWidth();
    const Int localHeightA = A.LocalHeight();
    const Int maxLocalHeight = MaxLength(height,rowStrideUnion);
    const Int maxLocalWidth = MaxLength(width,rowStride);
    const Int portionSize = mpi::Pad( maxLocalHeight*maxLocalWidth );

    T* thisBuf = this->Buffer();
    const Int ldim = this->LDim();
    const T* ABuf = A.LockedBuffer();
    const Int ALDim = A.LDim();

    T* buffer = this->auxMemory_.Require( 2*rowStrideUnion*portionSize );
    T* firstBuf  = &buffer[0];
    T* secondBuf = &buffer[rowStrideUnion*portionSize];

    if( rowAlign % rowStridePart == rowAlignA )
    {
        // Pack            
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<rowStrideUnion; ++k )
        {
            T* data = &firstBuf[k*portionSize];
            const Int rowRank = rowRankPart + k*rowStridePart;
            const Int rowShift = Shift_( rowRank, rowAlign, rowStride );
            const Int rowOffset = (rowShift-rowShiftA) / rowStridePart;
            const Int localWidth = Length_( width, rowShift, rowStride );
            EL_INNER_PARALLEL_FOR
            for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            {
                T* dataCol = &data[jLoc*localHeightA];
                const T* ACol = &ABuf[(rowOffset+jLoc*rowStrideUnion)*ALDim];
                MemCopy( dataCol, ACol, localHeightA );
            }
        }

        // Simultaneously Scatter in rows and Gather in columns
        mpi::AllToAll
        ( firstBuf,  portionSize, 
          secondBuf, portionSize, this->PartialUnionRowComm() );

        // Unpack
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<rowStrideUnion; ++k )
        {
            const T* data = &secondBuf[k*portionSize];
            const Int colShift = Shift_( k, colAlignA, rowStrideUnion );
            const Int localHeight = Length_( height, colShift, rowStrideUnion );
            EL_INNER_PARALLEL_FOR
            for( Int jLoc=0; jLoc<thisLocalWidth; ++jLoc )
            {
                const T* dataCol = &data[jLoc*localHeight];
                T* thisCol = &thisBuf[colShift+jLoc*ldim]; 
                for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                    thisCol[iLoc*rowStrideUnion] = dataCol[iLoc];
            }
        }
    }
    else
    {
#ifdef EL_UNALIGNED_WARNINGS
        if( this->Grid().Rank() == 0 )
            std::cerr << "Unaligned PartialRowAllToAllFrom" << std::endl;
#endif
        const Int sendRowRankPart = 
            (rowRankPart+rowStridePart+(rowAlign%rowStridePart)-rowAlignA) % 
            rowStridePart;
        const Int recvRowRankPart =
            (rowRankPart+rowStridePart+rowAlignA-(rowAlign%rowStridePart)) %
            rowStridePart; 

        // Pack
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<rowStrideUnion; ++k )
        {
            T* data = &secondBuf[k*portionSize];
            const Int rowRank = sendRowRankPart + k*rowStridePart;
            const Int rowShift = Shift_( rowRank, rowAlign, rowStride );
            const Int rowOffset = (rowShift-rowShiftA) / rowStridePart;
            const Int localWidth = Length_( width, rowShift, rowStride );
            EL_INNER_PARALLEL_FOR
            for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            {
                T* dataCol = &data[jLoc*localHeightA];
                const T* ACol = &ABuf[(rowOffset+jLoc*rowStrideUnion)*ALDim];
                MemCopy( dataCol, ACol, localHeightA );
            }
        }

        // Simultaneously Scatter in rows and Gather in columns
        mpi::AllToAll
        ( secondBuf, portionSize, 
          firstBuf,  portionSize, this->PartialUnionRowComm() );

        // Realign the result
        mpi::SendRecv 
        ( firstBuf,  rowStrideUnion*portionSize, sendRowRankPart,
          secondBuf, rowStrideUnion*portionSize, recvRowRankPart, 
          this->PartialRowComm() );

        // Unpack
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<rowStrideUnion; ++k )
        {
            const T* data = &secondBuf[k*portionSize];
            const Int colShift = Shift_( k, colAlignA, rowStrideUnion );
            const Int localHeight = Length_( height, colShift, rowStrideUnion );
            EL_INNER_PARALLEL_FOR
            for( Int jLoc=0; jLoc<thisLocalWidth; ++jLoc )
            {
                const T* dataCol = &data[jLoc*localHeight];
                T* thisCol = &thisBuf[colShift+jLoc*ldim]; 
                for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                    thisCol[iLoc*rowStrideUnion] = dataCol[iLoc];
            }
        }
    }
    this->auxMemory_.Release();
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::PartialColAllToAll
( DistMatrix<T,UPart,VScat>& A ) const
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::PartialColAllToAll");
        AssertSameGrids( *this, A );
    )
    const Int height = this->Height();
    const Int width = this->Width();
    A.AlignColsAndResize
    ( this->ColAlign()%A.ColStride(), height, width, false, false );
    if( !A.Participating() )
        return;

    const Int colAlign = this->ColAlign();
    const Int colAlignA = A.ColAlign();
    const Int rowAlignA = A.RowAlign();

    const Int colStride = this->ColStride();
    const Int colStridePart = this->PartialColStride();
    const Int colStrideUnion = this->PartialUnionColStride();
    const Int colRankPart = this->PartialColRank();

    const Int colShiftA = A.ColShift();

    const Int thisLocalHeight = this->LocalHeight();
    const Int localWidthA = A.LocalWidth();
    const Int maxLocalHeight = MaxLength(height,colStride);
    const Int maxLocalWidth = MaxLength(width,colStrideUnion);
    const Int portionSize = mpi::Pad( maxLocalHeight*maxLocalWidth );

    const T* thisBuf = this->LockedBuffer();
    const Int ldim = this->LDim();
    T* ABuf = A.Buffer();
    const Int ALDim = A.LDim();

    T* buffer = A.auxMemory_.Require( 2*colStrideUnion*portionSize );
    T* firstBuf  = &buffer[0];
    T* secondBuf = &buffer[colStrideUnion*portionSize];

    if( colAlignA == colAlign % colStridePart )
    {
        // Pack            
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<colStrideUnion; ++k )
        {
            T* data = &firstBuf[k*portionSize];
            const Int rowShift = Shift_( k, rowAlignA, colStrideUnion );
            const Int localWidth = Length_( width, rowShift, colStrideUnion );
            EL_INNER_PARALLEL_FOR
            for( Int jLoc=0; jLoc<localWidth; ++jLoc )
                MemCopy
                ( &data[jLoc*thisLocalHeight],
                  &thisBuf[(rowShift+jLoc*colStrideUnion)*ldim], 
                  thisLocalHeight );
        }

        // Simultaneously Gather in columns and Scatter in rows
        mpi::AllToAll
        ( firstBuf,  portionSize, 
          secondBuf, portionSize, this->PartialUnionColComm() );

        // Unpack
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<colStrideUnion; ++k )
        {
            const T* data = &secondBuf[k*portionSize];
            const Int colRank = colRankPart + k*colStridePart;
            const Int colShift = Shift_( colRank, colAlign, colStride );
            const Int colOffset = (colShift-colShiftA) / colStridePart;
            const Int localHeight = Length_( height, colShift, colStride );
            EL_INNER_PARALLEL_FOR
            for( Int jLoc=0; jLoc<localWidthA; ++jLoc )
            {
                T* ACol = &ABuf[colOffset+jLoc*ALDim];
                const T* dataCol = &data[jLoc*localHeight];
                for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                    ACol[iLoc*colStrideUnion] = dataCol[iLoc];
            }
        }
    }
    else
    {
#ifdef EL_UNALIGNED_WARNINGS
        if( this->Grid().Rank() == 0 )
            std::cerr << "Unaligned PartialColAllToAll" << std::endl;
#endif
        const Int colAlignDiff = colAlignA - (colAlign%colStridePart);
        const Int sendColRankPart = 
            (colRankPart+colStridePart+colAlignDiff) % colStridePart;
        const Int recvColRankPart =
            (colRankPart+colStridePart-colAlignDiff) % colStridePart;

        // Pack
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<colStrideUnion; ++k )
        {
            T* data = &secondBuf[k*portionSize];    
            const Int rowShift = Shift_( k, rowAlignA, colStrideUnion );
            const Int localWidth = Length_( width, rowShift, colStrideUnion );
            EL_INNER_PARALLEL_FOR
            for( Int jLoc=0; jLoc<localWidth; ++jLoc )
                MemCopy
                ( &data[jLoc*thisLocalHeight],
                  &thisBuf[(rowShift+jLoc*colStrideUnion)*ldim], 
                  thisLocalHeight );
        }

        // Realign the input
        mpi::SendRecv 
        ( secondBuf, colStrideUnion*portionSize, sendColRankPart,
          firstBuf,  colStrideUnion*portionSize, recvColRankPart, 
          this->PartialColComm() );

        // Simultaneously Scatter in columns and Gather in rows
        mpi::AllToAll
        ( firstBuf,  portionSize, 
          secondBuf, portionSize, this->PartialUnionColComm() );

        // Unpack
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<colStrideUnion; ++k )
        {
            const T* data = &secondBuf[k*portionSize];
            const Int colRank = recvColRankPart + k*colStridePart;
            const Int colShift = Shift_( colRank, colAlign, colStride );
            const Int colOffset = (colShift-colShiftA) / colStridePart;
            const Int localHeight = Length_( height, colShift, colStride );
            EL_INNER_PARALLEL_FOR
            for( Int jLoc=0; jLoc<localWidthA; ++jLoc )
            {
                T* ACol = &ABuf[colOffset+jLoc*ALDim];
                const T* dataCol = &data[jLoc*localHeight];
                for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                    ACol[iLoc*colStrideUnion] = dataCol[iLoc];
            }
        }
    }
    A.auxMemory_.Release();
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::PartialRowAllToAll
( DistMatrix<T,UScat,VPart>& A ) const
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::PartialRowAllToAll");
        AssertSameGrids( *this, A );
    )
    const Int height = this->Height();
    const Int width = this->Width();
    A.AlignRowsAndResize
    ( this->RowAlign()%A.RowStride(), height, width, false, false );
    if( !A.Participating() )
        return;

    const Int colAlignA = A.ColAlign();
    const Int rowAlign = this->RowAlign();
    const Int rowAlignA = A.RowAlign();

    const Int rowStride = this->RowStride();
    const Int rowStridePart = this->PartialRowStride();
    const Int rowStrideUnion = this->PartialUnionRowStride();
    const Int rowRankPart = this->PartialRowRank();

    const Int rowShiftA = A.RowShift();

    const Int thisLocalWidth = this->LocalWidth();
    const Int localHeightA = A.LocalHeight();
    const Int maxLocalWidth = MaxLength(width,rowStride);
    const Int maxLocalHeight = MaxLength(height,rowStrideUnion);
    const Int portionSize = mpi::Pad( maxLocalHeight*maxLocalWidth );

    const T* thisBuf = this->LockedBuffer();
    const Int ldim = this->LDim();
    T* ABuf = A.Buffer();
    const Int ALDim = A.LDim();

    T* buffer = A.auxMemory_.Require( 2*rowStrideUnion*portionSize );
    T* firstBuf  = &buffer[0];
    T* secondBuf = &buffer[rowStrideUnion*portionSize];

    if( rowAlignA == rowAlign % rowStridePart )
    {
        // Pack            
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<rowStrideUnion; ++k )
        {
            T* data = &firstBuf[k*portionSize];
            const Int colShift = Shift_( k, colAlignA, rowStrideUnion );
            const Int localHeight = Length_( height, colShift, rowStrideUnion );
            EL_INNER_PARALLEL_FOR
            for( Int jLoc=0; jLoc<thisLocalWidth; ++jLoc )
            {
                T* dataCol = &data[jLoc*localHeight];
                const T* thisCol = &thisBuf[colShift+jLoc*ldim];
                for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                    dataCol[iLoc] = thisCol[iLoc*rowStrideUnion];
            }
        }

        // Simultaneously Gather in rows and Scatter in columns
        mpi::AllToAll
        ( firstBuf,  portionSize, 
          secondBuf, portionSize, this->PartialUnionRowComm() );

        // Unpack
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<rowStrideUnion; ++k )
        {
            const T* data = &secondBuf[k*portionSize];
            const Int rowRank = rowRankPart + k*rowStridePart;
            const Int rowShift = Shift_( rowRank, rowAlign, rowStride );
            const Int rowOffset = (rowShift-rowShiftA) / rowStridePart;
            const Int localWidth = Length_( width, rowShift, rowStride );
            EL_INNER_PARALLEL_FOR
            for( Int jLoc=0; jLoc<localWidth; ++jLoc )
                MemCopy
                ( &ABuf[(rowOffset+jLoc*rowStrideUnion)*ALDim],
                  &data[jLoc*localHeightA], localHeightA );
        }
    }
    else
    {
#ifdef EL_UNALIGNED_WARNINGS
        if( this->Grid().Rank() == 0 )
            std::cerr << "Unaligned PartialRowAllToAll" << std::endl;
#endif
        const Int rowAlignDiff = rowAlignA - (rowAlign%rowStridePart);
        const Int sendRowRankPart = 
            (rowRankPart+rowStridePart+rowAlignDiff) % rowStridePart;
        const Int recvRowRankPart =
            (rowRankPart+rowStridePart-rowAlignDiff) % rowStridePart;

        // Pack
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<rowStrideUnion; ++k )
        {
            T* data = &secondBuf[k*portionSize];    
            const Int colShift = Shift_( k, colAlignA, rowStrideUnion );
            const Int localHeight = Length_( height, colShift, rowStrideUnion );
            EL_INNER_PARALLEL_FOR
            for( Int jLoc=0; jLoc<thisLocalWidth; ++jLoc )
            {
                T* dataCol = &data[jLoc*localHeight];
                const T* sourceCol = &thisBuf[colShift+jLoc*ldim];
                for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                    dataCol[iLoc] = sourceCol[iLoc*rowStrideUnion]; 
            }
        }

        // Realign the input
        mpi::SendRecv 
        ( secondBuf, rowStrideUnion*portionSize, sendRowRankPart,
          firstBuf,  rowStrideUnion*portionSize, recvRowRankPart, 
          this->PartialRowComm() );

        // Simultaneously Scatter in rows and Gather in columns
        mpi::AllToAll
        ( firstBuf,  portionSize, 
          secondBuf, portionSize, this->PartialUnionRowComm() );

        // Unpack
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<rowStrideUnion; ++k )
        {
            const T* data = &secondBuf[k*portionSize];
            const Int rowRank = recvRowRankPart + k*rowStridePart;
            const Int rowShift = Shift_( rowRank, rowAlign, rowStride );
            const Int rowOffset = (rowShift-rowShiftA) / rowStridePart;
            const Int localWidth = Length_( width, rowShift, rowStride );
            EL_INNER_PARALLEL_FOR
            for( Int jLoc=0; jLoc<localWidth; ++jLoc )
                MemCopy
                ( &ABuf[(rowOffset+jLoc*rowStrideUnion)*ALDim],
                  &data[jLoc*localHeightA], localHeightA );
        }
    }
    A.auxMemory_.Release();
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::RowSumScatterFrom( const DistMatrix<T,U,VGath>& A )
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::RowSumScatterFrom");
        AssertSameGrids( *this, A );
    )
    this->AlignColsAndResize
    ( A.ColAlign(), A.Height(), A.Width(), false, false );
    // NOTE: This will be *slightly* slower than necessary due to the result
    //       of the MPI operations being added rather than just copied
    Zeros( this->Matrix(), this->LocalHeight(), this->LocalWidth() );
    this->RowSumScatterUpdate( T(1), A );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::ColSumScatterFrom( const DistMatrix<T,UGath,V>& A )
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::ColSumScatterFrom");
        AssertSameGrids( *this, A );
    )
    this->AlignRowsAndResize
    ( A.RowAlign(), A.Height(), A.Width(), false, false );
    // NOTE: This will be *slightly* slower than necessary due to the result
    //       of the MPI operations being added rather than just copied
    Zeros( this->Matrix(), this->LocalHeight(), this->LocalWidth() );
    this->ColSumScatterUpdate( T(1), A );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::SumScatterFrom( const DistMatrix<T,UGath,VGath>& A )
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::SumScatterFrom");
        AssertSameGrids( *this, A );
    )
    this->Resize( A.Height(), A.Width() );
    // NOTE: This will be *slightly* slower than necessary due to the result
    //       of the MPI operations being added rather than just copied
    Zeros( this->Matrix(), this->LocalHeight(), this->LocalWidth() );
    this->SumScatterUpdate( T(1), A );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::PartialRowSumScatterFrom
( const DistMatrix<T,U,VPart>& A )
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::PartialRowSumScatterFrom");
        AssertSameGrids( *this, A );
    )
    this->AlignAndResize
    ( A.ColAlign(), A.RowAlign(), A.Height(), A.Width(), false, false );
    // NOTE: This will be *slightly* slower than necessary due to the result
    //       of the MPI operations being added rather than just copied
    Zeros( this->Matrix(), this->LocalHeight(), this->LocalWidth() );
    this->PartialRowSumScatterUpdate( T(1), A );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::PartialColSumScatterFrom
( const DistMatrix<T,UPart,V>& A )
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::PartialColSumScatterFrom");
        AssertSameGrids( *this, A );
    )
    this->AlignAndResize
    ( A.ColAlign(), A.RowAlign(), A.Height(), A.Width(), false, false );
    // NOTE: This will be *slightly* slower than necessary due to the result
    //       of the MPI operations being added rather than just copied
    Zeros( this->Matrix(), this->LocalHeight(), this->LocalWidth() );
    this->PartialColSumScatterUpdate( T(1), A );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::RowSumScatterUpdate
( T alpha, const DistMatrix<T,U,VGath>& A )
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::RowSumScatterUpdate");
        AssertSameGrids( *this, A );
        this->AssertNotLocked();
        this->AssertSameSize( A.Height(), A.Width() );
    )
    if( !this->Participating() )
        return;

    if( this->ColAlign() == A.ColAlign() )
    {
        if( this->Width() == 1 )
        {
            const Int rowAlign = this->RowAlign();
            const Int rowRank = this->RowRank();

            const Int localHeight = this->LocalHeight();
            const Int portionSize = mpi::Pad( localHeight );
            T* buffer = this->auxMemory_.Require( 2*portionSize );
            T* sendBuf = &buffer[0];
            T* recvBuf = &buffer[portionSize];

            // Pack 
            const T* ACol = A.LockedBuffer();
            MemCopy( sendBuf, ACol, localHeight );

            // Reduce to rowAlign
            mpi::Reduce
            ( sendBuf, recvBuf, portionSize, rowAlign, this->RowComm() );

            if( rowRank == rowAlign )
            {
                T* thisCol = this->Buffer();
                EL_FMA_PARALLEL_FOR
                for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                    thisCol[iLoc] += alpha*recvBuf[iLoc];
            }

            this->auxMemory_.Release();
        }
        else
        {
            const Int rowStride = this->RowStride();
            const Int rowAlign = this->RowAlign();

            const Int width = this->Width();
            const Int localHeight = this->LocalHeight();
            const Int localWidth = this->LocalWidth();
            const Int maxLocalWidth = MaxLength(width,rowStride);

            const Int portionSize = mpi::Pad( localHeight*maxLocalWidth );
            const Int sendSize = rowStride*portionSize;

            // Pack 
            const Int ALDim = A.LDim();
            const T* ABuffer = A.LockedBuffer();
            T* buffer = this->auxMemory_.Require( sendSize );
            EL_OUTER_PARALLEL_FOR
            for( Int k=0; k<rowStride; ++k )
            {
                T* data = &buffer[k*portionSize];
                const Int thisRowShift = Shift_( k, rowAlign, rowStride );
                const Int thisLocalWidth = 
                    Length_(width,thisRowShift,rowStride);
                EL_INNER_PARALLEL_FOR
                for( Int jLoc=0; jLoc<thisLocalWidth; ++jLoc )
                {
                    const T* ACol = 
                        &ABuffer[(thisRowShift+jLoc*rowStride)*ALDim];
                    T* dataCol = &data[jLoc*localHeight];
                    MemCopy( dataCol, ACol, localHeight );
                }
            }
            // Communicate
            mpi::ReduceScatter( buffer, portionSize, this->RowComm() );

            // Update with our received data
            T* thisBuffer = this->Buffer();
            const Int thisLDim = this->LDim();
            EL_PARALLEL_FOR
            for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            {
                const T* bufferCol = &buffer[jLoc*localHeight];
                T* thisCol = &thisBuffer[jLoc*thisLDim];
                blas::Axpy( localHeight, alpha, bufferCol, 1, thisCol, 1 );
            }
            this->auxMemory_.Release();
        }
    }
    else
    {
#ifdef EL_UNALIGNED_WARNINGS
        if( this->Grid().Rank() == 0 )
            std::cerr << "Unaligned RowSumScatterUpdate" << std::endl;
#endif
        if( this->Width() == 1 )
        {
            const Int colStride = this->ColStride();
            const Int rowAlign = this->RowAlign();
            const Int colRank = this->ColRank();
            const Int rowRank = this->RowRank();

            const Int height = this->Height();
            const Int localHeight = this->LocalHeight();
            const Int localHeightA = A.LocalHeight();
            const Int maxLocalHeight = MaxLength(height,colStride);
            const Int portionSize = mpi::Pad( maxLocalHeight );

            const Int colAlign = this->ColAlign();
            const Int colAlignA = A.ColAlign();
            const Int sendRow = 
                (colRank+colStride+colAlign-colAlignA) % colStride;
            const Int recvRow = 
                (colRank+colStride+colAlignA-colAlign) % colStride;

            T* buffer = this->auxMemory_.Require( 2*portionSize );
            T* sendBuf = &buffer[0];
            T* recvBuf = &buffer[portionSize];

            // Pack 
            // TODO: This pack is silly. Remove it
            const T* ACol = A.LockedBuffer();
            MemCopy( sendBuf, ACol, localHeightA );

            // Reduce to rowAlign
            mpi::Reduce
            ( sendBuf, recvBuf, portionSize, rowAlign, this->RowComm() );

            if( rowRank == rowAlign )
            {
                // Perform the realignment
                mpi::SendRecv
                ( recvBuf, portionSize, sendRow,
                  sendBuf, portionSize, recvRow, this->ColComm() );

                T* thisCol = this->Buffer();
                EL_FMA_PARALLEL_FOR
                for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                    thisCol[iLoc] += alpha*sendBuf[iLoc];
            }
            this->auxMemory_.Release();
        }
        else
        {
            const Int colStride = this->ColStride();
            const Int rowStride = this->RowStride();
            const Int colRank = this->ColRank();

            const Int colAlign = this->ColAlign();
            const Int rowAlign = this->RowAlign();
            const Int colAlignA = A.ColAlign();
            const Int sendRow = 
                (colRank+colStride+colAlign-colAlignA) % colStride;
            const Int recvRow = 
                (colRank+colStride+colAlignA-colAlign) % colStride;

            const Int width = this->Width();
            const Int localHeight = this->LocalHeight();
            const Int localWidth = this->LocalWidth();
            const Int localHeightA = A.LocalHeight();
            const Int maxLocalWidth = MaxLength(width,rowStride);

            const Int recvSize_RS = mpi::Pad( localHeightA*maxLocalWidth );
            const Int sendSize_RS = rowStride * recvSize_RS;
            const Int recvSize_SR = localHeight * localWidth;

            T* buffer = this->auxMemory_.Require
                ( recvSize_RS + std::max(sendSize_RS,recvSize_SR) );
            T* firstBuf = &buffer[0];
            T* secondBuf = &buffer[recvSize_RS];

            // Pack 
            EL_OUTER_PARALLEL_FOR
            for( Int k=0; k<rowStride; ++k )
            {
                const Int thisRowShift = Shift_( k, rowAlign, rowStride );
                const Int thisLocalWidth = 
                    Length_(width,thisRowShift,rowStride);
                InterleaveMatrix
                ( localHeightA, thisLocalWidth,
                  A.LockedBuffer(0,thisRowShift), 1, rowStride*A.LDim(),
                  &secondBuf[k*recvSize_RS],      1, localHeightA );
            }

            // Reduce-scatter over each process row
            mpi::ReduceScatter
            ( secondBuf, firstBuf, recvSize_RS, this->RowComm() );

            // Trade reduced data with the appropriate process row
            mpi::SendRecv
            ( firstBuf,  localHeightA*localWidth, sendRow,
              secondBuf, localHeight*localWidth,  recvRow, this->ColComm() );

            // Update with our received data
            InterleaveMatrixUpdate
            ( alpha, localHeight, localWidth,
              secondBuf,      1, localHeight,
              this->Buffer(), 1, this->LDim() );
            this->auxMemory_.Release();
        }
    }
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::ColSumScatterUpdate
( T alpha, const DistMatrix<T,UGath,V>& A )
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::ColSumScatterUpdate");
        AssertSameGrids( *this, A );
        this->AssertNotLocked();
        this->AssertSameSize( A.Height(), A.Width() );
    )
#ifdef EL_VECTOR_WARNINGS
    if( A.Width() == 1 && this->Grid().Rank() == 0 )
    {
        std::cerr <<
          "The vector version of ColSumScatterUpdate does not"
          " yet have a vector version implemented, but it would only "
          "require a modification of the vector version of RowSumScatterUpdate"
          << std::endl;
    }
#endif
#ifdef EL_CACHE_WARNINGS
    if( A.Width() != 1 && this->Grid().Rank() == 0 )
    {
        std::cerr <<
          "ColSumScatterUpdate potentially causes a large "
          "amount of cache-thrashing. If possible, avoid it by forming the "
          "(conjugate-)transpose of the [* ,V] matrix instead." << std::endl;
    }
#endif
    if( !this->Participating() )
        return;

    if( this->RowAlign() == A.RowAlign() )
    {
        const Int colStride = this->ColStride();
        const Int colAlign = this->ColAlign();
        const Int height = this->Height();
        const Int localHeight = this->LocalHeight();
        const Int localWidth = this->LocalWidth();
        const Int maxLocalHeight = MaxLength(height,colStride);

        const Int recvSize = mpi::Pad( maxLocalHeight*localWidth );
        const Int sendSize = colStride*recvSize;

        // Pack 
        T* buffer = this->auxMemory_.Require( sendSize );
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<colStride; ++k )
        {
            const Int thisColShift = Shift_( k, colAlign, colStride );
            const Int thisLocalHeight = Length_(height,thisColShift,colStride);
            InterleaveMatrix
            ( thisLocalHeight, localWidth,
              A.LockedBuffer(thisColShift,0), colStride, A.LDim(),
              &buffer[k*recvSize],            1,         thisLocalHeight );
        }

        // Communicate
        mpi::ReduceScatter( buffer, recvSize, this->ColComm() );

        // Update with our received data
        InterleaveMatrixUpdate
        ( alpha, localHeight, localWidth,
          buffer,         1, localHeight,
          this->Buffer(), 1, this->LDim() );
        this->auxMemory_.Release();
    }
    else
    {
#ifdef EL_UNALIGNED_WARNINGS
        if( this->Grid().Rank() == 0 )
            std::cerr << "Unaligned ColSumScatterUpdate" << std::endl;
#endif
        const Int colStride = this->ColStride();
        const Int rowStride = this->RowStride();
        const Int rowRank = this->RowRank();

        const Int colAlign = this->ColAlign();
        const Int rowAlign = this->RowAlign();
        const Int rowAlignA = A.RowAlign();
        const Int sendCol = (rowRank+rowStride+rowAlign-rowAlignA) % rowStride;
        const Int recvCol = (rowRank+rowStride+rowAlignA-rowAlign) % rowStride;

        const Int height = this->Height();
        const Int localHeight = this->LocalHeight();
        const Int localWidth = this->LocalWidth();
        const Int localWidthA = A.LocalWidth();
        const Int maxLocalHeight = MaxLength(height,colStride);

        const Int recvSize_RS = mpi::Pad( maxLocalHeight*localWidthA );
        const Int sendSize_RS = colStride * recvSize_RS;
        const Int recvSize_SR = localHeight * localWidth;

        T* buffer = this->auxMemory_.Require
            ( recvSize_RS + std::max(sendSize_RS,recvSize_SR) );
        T* firstBuf = &buffer[0];
        T* secondBuf = &buffer[recvSize_RS];

        // Pack
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<colStride; ++k )
        {
            const Int thisColShift = Shift_( k, colAlign, colStride );
            const Int thisLocalHeight = Length_(height,thisColShift,colStride);
            InterleaveMatrix
            ( thisLocalHeight, localWidthA,
              A.LockedBuffer(thisColShift,0), colStride, A.LDim(),
              &secondBuf[k*recvSize_RS],      1,         thisLocalHeight );
        }

        // Reduce-scatter over each col
        mpi::ReduceScatter( secondBuf, firstBuf, recvSize_RS, this->ColComm() );

        // Trade reduced data with the appropriate col
        mpi::SendRecv
        ( firstBuf,  localHeight*localWidthA, sendCol,
          secondBuf, localHeight*localWidth,  recvCol, this->RowComm() );

        // Update with our received data
        InterleaveMatrixUpdate
        ( alpha, localHeight, localWidth,
          secondBuf,      1, localHeight,
          this->Buffer(), 1, this->LDim() );
        this->auxMemory_.Release();
    }
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::SumScatterUpdate
( T alpha, const DistMatrix<T,UGath,VGath>& A )
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::SumScatterUpdate");
        AssertSameGrids( *this, A );
        this->AssertNotLocked();
        this->AssertSameSize( A.Height(), A.Width() );
    )
    if( !this->Participating() )
        return;

    const Int colStride = this->ColStride();
    const Int rowStride = this->RowStride();
    const Int colAlign = this->ColAlign();
    const Int rowAlign = this->RowAlign();

    const Int height = this->Height();
    const Int width = this->Width();
    const Int localHeight = this->LocalHeight();
    const Int localWidth = this->LocalWidth();
    const Int maxLocalHeight = MaxLength(height,colStride);
    const Int maxLocalWidth = MaxLength(width,rowStride);

    const Int recvSize = mpi::Pad( maxLocalHeight*maxLocalWidth );
    const Int sendSize = colStride*rowStride*recvSize;

    // Pack 
    T* buffer = this->auxMemory_.Require( sendSize );
    EL_OUTER_PARALLEL_FOR
    for( Int l=0; l<rowStride; ++l )
    {
        const Int thisRowShift = Shift_( l, rowAlign, rowStride );
        const Int thisLocalWidth = Length_( width, thisRowShift, rowStride );
        for( Int k=0; k<colStride; ++k )
        {
            T* data = &buffer[(k+l*colStride)*recvSize];
            const Int thisColShift = Shift_( k, colAlign, colStride );
            const Int thisLocalHeight = Length_(height,thisColShift,colStride);
            InterleaveMatrix
            ( thisLocalHeight, thisLocalWidth,
              A.LockedBuffer(thisColShift,thisRowShift), colStride, A.LDim(),
              data, 1, thisLocalHeight );
        }
    }

    // Communicate
    mpi::ReduceScatter( buffer, recvSize, this->DistComm() );

    // Unpack our received data
    InterleaveMatrixUpdate
    ( alpha, localHeight, localWidth,
      buffer,         1, localHeight,
      this->Buffer(), 1, this->LDim() );
    this->auxMemory_.Release();
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::PartialRowSumScatterUpdate
( T alpha, const DistMatrix<T,U,VPart>& A )
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::PartialRowSumScatterUpdate");
        AssertSameGrids( *this, A );
        this->AssertNotLocked();
        this->AssertSameSize( A.Height(), A.Width() );
    )
    if( !this->Participating() )
        return;

    if( this->RowAlign() % A.RowStride() == A.RowAlign() )
    {
        const Int rowStride = this->RowStride();
        const Int rowStridePart = this->PartialRowStride();
        const Int rowStrideUnion = this->PartialUnionRowStride();
        const Int rowRankPart = this->PartialRowRank();
        const Int rowAlign = this->RowAlign();
        const Int rowShiftOfA = A.RowShift();

        const Int height = this->Height();
        const Int width = this->Width();
        const Int localWidth = this->LocalWidth();
        const Int maxLocalWidth = MaxLength( width, rowStride );
        const Int recvSize = mpi::Pad( height*maxLocalWidth );
        const Int sendSize = rowStrideUnion*recvSize;

        // Pack
        T* buffer = this->auxMemory_.Require( sendSize );
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<rowStrideUnion; ++k )
        {
            T* data = &buffer[k*recvSize];
            const Int thisRank = rowRankPart+k*rowStridePart;
            const Int thisRowShift = Shift_( thisRank, rowAlign, rowStride );
            const Int thisRowOffset = 
                (thisRowShift-rowShiftOfA) / rowStridePart;
            const Int thisLocalWidth = 
                Length_( width, thisRowShift, rowStride );
            InterleaveMatrix
            ( height, thisLocalWidth,
              A.LockedBuffer(0,thisRowOffset), 1, rowStrideUnion*A.LDim(),
              data,                            1, height );
        }
    
        // Communicate
        mpi::ReduceScatter( buffer, recvSize, this->PartialUnionRowComm() );

        // Unpack our received data
        InterleaveMatrixUpdate
        ( alpha, height, localWidth,
          buffer,         1, height,
          this->Buffer(), 1, this->LDim() );
        this->auxMemory_.Release();
    }
    else
        LogicError("Unaligned PartialRowSumScatterUpdate not implemented");
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::PartialColSumScatterUpdate
( T alpha, const DistMatrix<T,UPart,V>& A )
{
    DEBUG_ONLY(
        CallStackEntry cse("GDM::PartialColSumScatterUpdate");
        AssertSameGrids( *this, A );
        this->AssertNotLocked();
        this->AssertSameSize( A.Height(), A.Width() );
    )
    if( !this->Participating() )
        return;

#ifdef EL_CACHE_WARNINGS
    if( A.Width() != 1 && A.Grid().Rank() == 0 )
    {
        std::cerr <<
          "PartialColSumScatterUpdate potentially causes a large amount"
          " of cache-thrashing. If possible, avoid it by forming the "
          "(conjugate-)transpose of the [UGath,* ] matrix instead." 
          << std::endl;
    }
#endif
    if( this->ColAlign() % A.ColStride() == A.ColAlign() )
    {
        const Int colStride = this->ColStride();
        const Int colStridePart = this->PartialColStride();
        const Int colStrideUnion = this->PartialUnionColStride();
        const Int colRankPart = this->PartialColRank();
        const Int colAlign = this->ColAlign();
        const Int colShiftOfA = A.ColShift();

        const Int height = this->Height();
        const Int width = this->Width();
        const Int localHeight = this->LocalHeight();
        const Int maxLocalHeight = MaxLength( height, colStride );
        const Int recvSize = mpi::Pad( maxLocalHeight*width );
        const Int sendSize = colStrideUnion*recvSize;

        T* buffer = this->auxMemory_.Require( sendSize );

        // Pack
        const Int ALDim = A.LDim();
        const T* ABuf = A.LockedBuffer();
        EL_OUTER_PARALLEL_FOR
        for( Int k=0; k<colStrideUnion; ++k )
        {
            T* data = &buffer[k*recvSize];
            const Int thisRank = colRankPart+k*colStridePart;
            const Int thisColShift = Shift_( thisRank, colAlign, colStride );
            const Int thisColOffset = 
                (thisColShift-colShiftOfA) / colStridePart;
            const Int thisLocalHeight = 
                Length_( height, thisColShift, colStride );
            InterleaveMatrix
            ( thisLocalHeight, width,
              A.LockedBuffer(thisColOffset,0), colStrideUnion, A.LDim(),
              data, 1, thisLocalHeight );
        }

        // Communicate
        mpi::ReduceScatter( buffer, recvSize, this->PartialUnionColComm() );

        // Unpack our received data
        InterleaveMatrixUpdate
        ( alpha, localHeight, width,
          buffer,         1, localHeight,
          this->Buffer(), 1, this->LDim() );
        this->auxMemory_.Release();
    }
    else
    {
        LogicError("Unaligned PartialColSumScatterUpdate not implemented");
    }
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::TransposeColAllGather
( DistMatrix<T,V,UGath>& A, bool conjugate ) const
{
    DEBUG_ONLY(CallStackEntry cse("GDM::TransposeColAllGather"))
    DistMatrix<T,V,U> ATrans( this->Grid() );
    ATrans.AlignWith( *this );
    ATrans.Resize( this->Width(), this->Height() );
    Transpose( this->LockedMatrix(), ATrans.Matrix(), conjugate );
    copy::RowAllGather( ATrans, A );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::TransposePartialColAllGather
( DistMatrix<T,V,UPart>& A, bool conjugate ) const
{
    DEBUG_ONLY(CallStackEntry cse("GDM::TransposePartialColAllGather"))
    DistMatrix<T,V,U> ATrans( this->Grid() );
    ATrans.AlignWith( *this );
    ATrans.Resize( this->Width(), this->Height() );
    Transpose( this->LockedMatrix(), ATrans.Matrix(), conjugate );
    ATrans.PartialRowAllGather( A );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::AdjointColAllGather( DistMatrix<T,V,UGath>& A ) const
{
    DEBUG_ONLY(CallStackEntry cse("GDM::AdjointRowAllGather"))
    this->TransposeColAllGather( A, true );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::AdjointPartialColAllGather
( DistMatrix<T,V,UPart>& A ) const
{
    DEBUG_ONLY(CallStackEntry cse("GDM::AdjointPartialColAllGather"))
    this->TransposePartialColAllGather( A, true );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::TransposeColFilterFrom
( const DistMatrix<T,V,UGath>& A, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::TransposeColFilterFrom"))
    DistMatrix<T,V,U> AFilt( A.Grid() );
    if( this->ColConstrained() )
        AFilt.AlignRowsWith( *this, false );
    if( this->RowConstrained() )
        AFilt.AlignColsWith( *this, false );
    AFilt.RowFilterFrom( A );
    if( !this->ColConstrained() )
        this->AlignColsWith( AFilt, false );
    if( !this->RowConstrained() )
        this->AlignRowsWith( AFilt, false );
    this->Resize( A.Width(), A.Height() );
    Transpose( AFilt.LockedMatrix(), this->Matrix(), conjugate );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::TransposeRowFilterFrom
( const DistMatrix<T,VGath,U>& A, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::TransposeRowFilterFrom"))
    DistMatrix<T,V,U> AFilt( A.Grid() );
    if( this->ColConstrained() )
        AFilt.AlignRowsWith( *this, false );
    if( this->RowConstrained() )
        AFilt.AlignColsWith( *this, false );
    AFilt.ColFilterFrom( A );
    if( !this->ColConstrained() )
        this->AlignColsWith( AFilt, false );
    if( !this->RowConstrained() )
        this->AlignRowsWith( AFilt, false );
    this->Resize( A.Width(), A.Height() );
    Transpose( AFilt.LockedMatrix(), this->Matrix(), conjugate );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::TransposePartialColFilterFrom
( const DistMatrix<T,V,UPart>& A, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::TransposePartialColFilterFrom"))
    DistMatrix<T,V,U> AFilt( A.Grid() );
    if( this->ColConstrained() )
        AFilt.AlignRowsWith( *this, false );
    if( this->RowConstrained() )
        AFilt.AlignColsWith( *this, false );
    AFilt.PartialRowFilterFrom( A );
    if( !this->ColConstrained() )
        this->AlignColsWith( AFilt, false );
    if( !this->RowConstrained() )
        this->AlignRowsWith( AFilt, false );
    this->Resize( A.Width(), A.Height() );
    Transpose( AFilt.LockedMatrix(), this->Matrix(), conjugate );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::TransposePartialRowFilterFrom
( const DistMatrix<T,VPart,U>& A, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::TransposePartialRowFilterFrom"))
    DistMatrix<T,V,U> AFilt( A.Grid() );
    if( this->ColConstrained() )
        AFilt.AlignRowsWith( *this, false );
    if( this->RowConstrained() )
        AFilt.AlignColsWith( *this, false );
    AFilt.PartialColFilterFrom( A );
    if( !this->ColConstrained() )
        this->AlignColsWith( AFilt, false );
    if( !this->RowConstrained() )
        this->AlignRowsWith( AFilt, false );
    this->Resize( A.Width(), A.Height() );
    Transpose( AFilt.LockedMatrix(), this->Matrix(), conjugate );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::AdjointColFilterFrom( const DistMatrix<T,V,UGath>& A )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::AdjointColFilterFrom"))
    this->TransposeColFilterFrom( A, true );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::AdjointRowFilterFrom( const DistMatrix<T,VGath,U>& A )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::AdjointRowFilterFrom"))
    this->TransposeRowFilterFrom( A, true );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::AdjointPartialColFilterFrom
( const DistMatrix<T,V,UPart>& A )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::AdjointPartialColFilterFrom"))
    this->TransposePartialColFilterFrom( A, true );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::AdjointPartialRowFilterFrom
( const DistMatrix<T,VPart,U>& A )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::AdjointPartialRowFilterFrom"))
    this->TransposePartialRowFilterFrom( A, true );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::TransposeColSumScatterFrom
( const DistMatrix<T,V,UGath>& A, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::TransposeColSumScatterFrom"))
    DistMatrix<T,V,U> ASumFilt( A.Grid() );
    if( this->ColConstrained() )
        ASumFilt.AlignRowsWith( *this, false );
    if( this->RowConstrained() )
        ASumFilt.AlignColsWith( *this, false );
    ASumFilt.RowSumScatterFrom( A );
    if( !this->ColConstrained() )
        this->AlignColsWith( ASumFilt, false );
    if( !this->RowConstrained() )
        this->AlignRowsWith( ASumFilt, false );
    this->Resize( A.Width(), A.Height() );
    Transpose( ASumFilt.LockedMatrix(), this->Matrix(), conjugate );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::TransposePartialColSumScatterFrom
( const DistMatrix<T,V,UPart>& A, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::TransposePartialColSumScatterFrom"))
    DistMatrix<T,V,U> ASumFilt( A.Grid() );
    if( this->ColConstrained() )
        ASumFilt.AlignRowsWith( *this, false );
    if( this->RowConstrained() )
        ASumFilt.AlignColsWith( *this, false );
    ASumFilt.PartialRowSumScatterFrom( A );
    if( !this->ColConstrained() )
        this->AlignColsWith( ASumFilt, false );
    if( !this->RowConstrained() )
        this->AlignRowsWith( ASumFilt, false );
    this->Resize( A.Width(), A.Height() );
    Transpose( ASumFilt.LockedMatrix(), this->Matrix(), conjugate );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::AdjointColSumScatterFrom
( const DistMatrix<T,V,UGath>& A )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::AdjointColSumScatterFrom"))
    this->TransposeColSumScatterFrom( A, true );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::AdjointPartialColSumScatterFrom
( const DistMatrix<T,V,UPart>& A )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::AdjointPartialColSumScatterFrom"))
    this->TransposePartialColSumScatterFrom( A, true );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::TransposeColSumScatterUpdate
( T alpha, const DistMatrix<T,V,UGath>& A, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::TransposeColSumScatterUpdate"))
    DistMatrix<T,V,U> ASumFilt( A.Grid() );
    if( this->ColConstrained() )
        ASumFilt.AlignRowsWith( *this, false );
    if( this->RowConstrained() )
        ASumFilt.AlignColsWith( *this, false );
    ASumFilt.RowSumScatterFrom( A );
    if( !this->ColConstrained() )
        this->AlignColsWith( ASumFilt, false );
    if( !this->RowConstrained() )
        this->AlignRowsWith( ASumFilt, false );
    // ALoc += alpha ASumFiltLoc'
    El::Matrix<T>& ALoc = this->Matrix();
    const El::Matrix<T>& BLoc = ASumFilt.LockedMatrix();
    const Int localHeight = ALoc.Height();
    const Int localWidth = ALoc.Width();
    if( conjugate )
    {
        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                ALoc.Update( iLoc, jLoc, alpha*Conj(BLoc.Get(jLoc,iLoc)) );
    }
    else
    {
        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                ALoc.Update( iLoc, jLoc, alpha*BLoc.Get(jLoc,iLoc) );
    }
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::TransposePartialColSumScatterUpdate
( T alpha, const DistMatrix<T,V,UPart>& A, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::TransposePartialColSumScatterUpdate"))
    DistMatrix<T,V,U> ASumFilt( A.Grid() );
    if( this->ColConstrained() )
        ASumFilt.AlignRowsWith( *this, false );
    if( this->RowConstrained() )
        ASumFilt.AlignColsWith( *this, false );
    ASumFilt.PartialRowSumScatterFrom( A );
    if( !this->ColConstrained() )
        this->AlignColsWith( ASumFilt, false );
    if( !this->RowConstrained() )
        this->AlignRowsWith( ASumFilt, false );
    // ALoc += alpha ASumFiltLoc'
    El::Matrix<T>& ALoc = this->Matrix();
    const El::Matrix<T>& BLoc = ASumFilt.LockedMatrix();
    const Int localHeight = ALoc.Height();
    const Int localWidth = ALoc.Width();
    if( conjugate )
    {
        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                ALoc.Update( iLoc, jLoc, alpha*Conj(BLoc.Get(jLoc,iLoc)) );
    }
    else
    {
        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
            for( Int iLoc=0; iLoc<localHeight; ++iLoc )
                ALoc.Update( iLoc, jLoc, alpha*BLoc.Get(jLoc,iLoc) );
    }
}

template<typename T,Dist U,Dist V>
void GeneralDistMatrix<T,U,V>::AdjointColSumScatterUpdate
( T alpha, const DistMatrix<T,V,UGath>& A )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::AdjointColSumScatterUpdate"))
    this->TransposeColSumScatterUpdate( alpha, A, true );
}

template<typename T,Dist U,Dist V>
void GeneralDistMatrix<T,U,V>::AdjointPartialColSumScatterUpdate
( T alpha, const DistMatrix<T,V,UPart>& A )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::AdjointPartialColSumScatterUpdate"))
    this->TransposePartialColSumScatterUpdate( alpha, A, true );
}

// Basic queries
// =============
// Distribution information
// ------------------------
template<typename T,Dist U,Dist V>
Dist GeneralDistMatrix<T,U,V>::ColDist() const { return U; }
template<typename T,Dist U,Dist V>
Dist GeneralDistMatrix<T,U,V>::RowDist() const { return V; }
template<typename T,Dist U,Dist V>
Dist GeneralDistMatrix<T,U,V>::PartialColDist() const 
{ return Partial<U>(); }
template<typename T,Dist U,Dist V>
Dist GeneralDistMatrix<T,U,V>::PartialRowDist() const 
{ return Partial<V>(); }
template<typename T,Dist U,Dist V>
Dist GeneralDistMatrix<T,U,V>::PartialUnionColDist() const 
{ return PartialUnionCol<U,V>(); }
template<typename T,Dist U,Dist V>
Dist GeneralDistMatrix<T,U,V>::PartialUnionRowDist() const 
{ return PartialUnionRow<U,V>(); }

// Diagonal manipulation
// =====================
template<typename T,Dist U,Dist V>
void GeneralDistMatrix<T,U,V>::GetDiagonal
( AbstractDistMatrix<T>& d, Int offset ) const
{
    DEBUG_ONLY(CallStackEntry cse("GDM::GetDiagonal"))
    this->GetDiagonalHelper
    ( d, offset, []( T& alpha, T beta ) { alpha = beta; } );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::GetRealPartOfDiagonal
( AbstractDistMatrix<Base<T>>& d, Int offset ) const
{
    DEBUG_ONLY(CallStackEntry cse("GDM::GetRealPartOfDiagonal"))
    this->GetDiagonalHelper
    ( d, offset, []( Base<T>& alpha, T beta ) { alpha = RealPart(beta); } );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::GetImagPartOfDiagonal
( AbstractDistMatrix<Base<T>>& d, Int offset ) const
{
    DEBUG_ONLY(CallStackEntry cse("GDM::GetImagPartOfDiagonal"))
    this->GetDiagonalHelper
    ( d, offset, []( Base<T>& alpha, T beta ) { alpha = ImagPart(beta); } );
}

template<typename T,Dist U,Dist V>
auto
GeneralDistMatrix<T,U,V>::GetDiagonal( Int offset ) const
-> DistMatrix<T,UDiag,VDiag>
{
    DistMatrix<T,UDiag,VDiag> d( this->Grid() );
    GetDiagonal( d, offset );
    return d;
}

template<typename T,Dist U,Dist V>
auto
GeneralDistMatrix<T,U,V>::GetRealPartOfDiagonal( Int offset ) const
-> DistMatrix<Base<T>,UDiag,VDiag>
{
    DistMatrix<Base<T>,UDiag,VDiag> d( this->Grid() );
    GetRealPartOfDiagonal( d, offset );
    return d;
}

template<typename T,Dist U,Dist V>
auto
GeneralDistMatrix<T,U,V>::GetImagPartOfDiagonal( Int offset ) const
-> DistMatrix<Base<T>,UDiag,VDiag>
{
    DistMatrix<Base<T>,UDiag,VDiag> d( this->Grid() );
    GetImagPartOfDiagonal( d, offset );
    return d;
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::SetDiagonal
( const AbstractDistMatrix<T>& d, Int offset )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::SetDiagonal"))
    this->SetDiagonalHelper
    ( d, offset, []( T& alpha, T beta ) { alpha = beta; } );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::SetRealPartOfDiagonal
( const AbstractDistMatrix<Base<T>>& d, Int offset )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::SetRealPartOfDiagonal"))
    this->SetDiagonalHelper
    ( d, offset, 
      []( T& alpha, Base<T> beta ) { El::SetRealPart(alpha,beta); } );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::SetImagPartOfDiagonal
( const AbstractDistMatrix<Base<T>>& d, Int offset )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::SetImagPartOfDiagonal"))
    this->SetDiagonalHelper
    ( d, offset, 
      []( T& alpha, Base<T> beta ) { El::SetImagPart(alpha,beta); } );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::UpdateDiagonal
( T gamma, const AbstractDistMatrix<T>& d, Int offset )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::UpdateDiagonal"))
    this->SetDiagonalHelper
    ( d, offset, [gamma]( T& alpha, T beta ) { alpha += gamma*beta; } );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::UpdateRealPartOfDiagonal
( Base<T> gamma, const AbstractDistMatrix<Base<T>>& d, Int offset )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::UpdateRealPartOfDiagonal"))
    this->SetDiagonalHelper
    ( d, offset, 
      [gamma]( T& alpha, Base<T> beta ) 
      { El::UpdateRealPart(alpha,gamma*beta); } );
}

template<typename T,Dist U,Dist V>
void
GeneralDistMatrix<T,U,V>::UpdateImagPartOfDiagonal
( Base<T> gamma, const AbstractDistMatrix<Base<T>>& d, Int offset )
{
    DEBUG_ONLY(CallStackEntry cse("GDM::UpdateImagPartOfDiagonal"))
    this->SetDiagonalHelper
    ( d, offset, 
      [gamma]( T& alpha, Base<T> beta ) 
      { El::UpdateImagPart(alpha,gamma*beta); } );
}

// Private section
// ###############

// Diagonal helper functions
// =========================
template<typename T,Dist U,Dist V>
template<typename S,class Function>
void
GeneralDistMatrix<T,U,V>::GetDiagonalHelper
( AbstractDistMatrix<S>& dPre, Int offset, Function func ) const
{
    DEBUG_ONLY(
      CallStackEntry cse("GDM::GetDiagonalHelper");
      AssertSameGrids( *this, dPre );
    )
    ProxyCtrl ctrl;
    ctrl.colConstrain = true;
    ctrl.colAlign = this->DiagonalAlign(offset);
    ctrl.rootConstrain = true;
    ctrl.root = this->DiagonalRoot(offset);
    auto dPtr = WriteProxy<S,UDiag,VDiag>(&dPre,ctrl);
    auto& d = *dPtr;

    d.Resize( this->DiagonalLength(offset), 1 );
    if( d.Participating() )
    {
        const Int diagShift = d.ColShift();
        const Int diagStride = d.ColStride();
        const Int iStart = ( offset>=0 ? diagShift        : diagShift-offset );
        const Int jStart = ( offset>=0 ? diagShift+offset : diagShift        );

        const Int colStride = this->ColStride();
        const Int rowStride = this->RowStride();
        const Int iLocStart = (iStart-this->ColShift()) / colStride;
        const Int jLocStart = (jStart-this->RowShift()) / rowStride;

        const Int localDiagLength = d.LocalHeight();
        S* dBuf = d.Buffer();
        const T* buffer = this->LockedBuffer();
        const Int ldim = this->LDim();

        EL_PARALLEL_FOR
        for( Int k=0; k<localDiagLength; ++k )
        {
            const Int iLoc = iLocStart + k*(diagStride/colStride);
            const Int jLoc = jLocStart + k*(diagStride/rowStride);
            func( dBuf[k], buffer[iLoc+jLoc*ldim] );
        }
    }
}

template<typename T,Dist U,Dist V>
template<typename S,class Function>
void
GeneralDistMatrix<T,U,V>::SetDiagonalHelper
( const AbstractDistMatrix<S>& dPre, Int offset, Function func ) 
{
    DEBUG_ONLY(
      CallStackEntry cse("GDM::SetDiagonalHelper");
      AssertSameGrids( *this, dPre );
    )
    ProxyCtrl ctrl;
    ctrl.colConstrain = true;
    ctrl.colAlign = this->DiagonalAlign(offset);
    ctrl.rootConstrain = true;
    ctrl.root = this->DiagonalRoot(offset);
    auto dPtr = ReadProxy<S,UDiag,VDiag>(&dPre,ctrl); 
    const auto& d = *dPtr;

    if( !d.Participating() )
        return;

    const Int diagShift = d.ColShift();
    const Int diagStride = d.ColStride();
    const Int iStart = ( offset>=0 ? diagShift        : diagShift-offset );
    const Int jStart = ( offset>=0 ? diagShift+offset : diagShift        );

    const Int colStride = this->ColStride();
    const Int rowStride = this->RowStride();
    const Int iLocStart = (iStart-this->ColShift()) / colStride;
    const Int jLocStart = (jStart-this->RowShift()) / rowStride;

    const Int localDiagLength = d.LocalHeight();
    const S* dBuf = d.LockedBuffer();
    T* buffer = this->Buffer();
    const Int ldim = this->LDim();

    EL_PARALLEL_FOR
    for( Int k=0; k<localDiagLength; ++k )
    {
        const Int iLoc = iLocStart + k*(diagStride/colStride);
        const Int jLoc = jLocStart + k*(diagStride/rowStride);
        func( buffer[iLoc+jLoc*ldim], dBuf[k] );
    }
}

// Instantiations for {Int,Real,Complex<Real>} for each Real in {float,double}
// ###########################################################################

#define DISTPROTO(T,U,V) template class GeneralDistMatrix<T,U,V>
  
#define PROTO(T)\
  DISTPROTO(T,CIRC,CIRC);\
  DISTPROTO(T,MC,  MR  );\
  DISTPROTO(T,MC,  STAR);\
  DISTPROTO(T,MD,  STAR);\
  DISTPROTO(T,MR,  MC  );\
  DISTPROTO(T,MR,  STAR);\
  DISTPROTO(T,STAR,MC  );\
  DISTPROTO(T,STAR,MD  );\
  DISTPROTO(T,STAR,MR  );\
  DISTPROTO(T,STAR,STAR);\
  DISTPROTO(T,STAR,VC  );\
  DISTPROTO(T,STAR,VR  );\
  DISTPROTO(T,VC,  STAR);\
  DISTPROTO(T,VR,  STAR);

#include "El/macros/Instantiate.h"

} // namespace El
