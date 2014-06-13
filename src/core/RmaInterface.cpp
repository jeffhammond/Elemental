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
#include "El-lite.hpp"

namespace El {

template<typename T>
RmaInterface<T>::RmaInterface()
: attachedForLocalToGlobal_(false), attachedForGlobalToLocal_(false), 
  localToGlobalMat_(0), globalToLocalMat_(0),
  sendDummy_(0), recvDummy_(0)
{ }

template<typename T>
RmaInterface<T>::RmaInterface( RmaType type, DistMatrix<T>& Z )
: sendDummy_(0), recvDummy_(0)
{
    DEBUG_ONLY(CallStackEntry cse("RmaInterface::RmaInterface"))
    if( type == LOCAL_TO_GLOBAL )
    {
        attachedForLocalToGlobal_ = true;
        attachedForGlobalToLocal_ = false;
    }
    else
    {
        attachedForLocalToGlobal_ = false;
        attachedForGlobalToLocal_ = true;
    }
}

template<typename T>
RmaInterface<T>::RmaInterface( RmaType type, const DistMatrix<T>& X )
: sendDummy_(0), recvDummy_(0)
{
    DEBUG_ONLY(CallStackEntry cse("RmaInterface::RmaInterface"))
    if( type == LOCAL_TO_GLOBAL )
    {
        LogicError("Cannot update a constant matrix");
    }
    else
    {
        attachedForLocalToGlobal_ = false;
        attachedForGlobalToLocal_ = true;
    }
}

template<typename T>
RmaInterface<T>::~RmaInterface()
{
    if( attachedForLocalToGlobal_ || attachedForGlobalToLocal_ )
    {
        if( std::uncaught_exception() )
        {
           const Grid& g = ( attachedForLocalToGlobal_ ?
                             localToGlobalMat_->Grid() :
                             globalToLocalMat_->Grid() );
           std::ostringstream os;
           os << g.Rank()
              << "Uncaught exception detected during RmaInterface destructor "
                 "that required a call to Detach. Instead of allowing for the "
                 "possibility of Detach throwing another exception and "
                 "resulting in a 'terminate', we instead immediately dump the "
                 "call stack (if not in RELEASE mode) since the program will "
                 "likely hang:" << std::endl;
           std::cerr << os.str();
           DEBUG_ONLY(DumpCallStack())
        }
        else
        {
            Detach();
        }
    }
}

template<typename T>
void RmaInterface<T>::Attach( RmaType type, DistMatrix<T>& Z )
{
    DEBUG_ONLY(CallStackEntry cse("RmaInterface::Attach"))
    if( attachedForLocalToGlobal_ || attachedForGlobalToLocal_ )
        LogicError("Must detach before reattaching.");

    if( type == LOCAL_TO_GLOBAL )
    {
        attachedForLocalToGlobal_ = true;
    }
    else
    {
        attachedForGlobalToLocal_ = true;
    }
}

template<typename T>
void RmaInterface<T>::Attach( RmaType type, const DistMatrix<T>& X )
{
    DEBUG_ONLY(CallStackEntry cse("RmaInterface::Attach"))
    if( attachedForLocalToGlobal_ || attachedForGlobalToLocal_ )
        LogicError("Must detach before reattaching.");

    if( type == LOCAL_TO_GLOBAL )
    {
        LogicError("Cannot update a constant matrix");
    }
    else
    {
        attachedForGlobalToLocal_ = true;
        globalToLocalMat_ = &X;
    }
}

template<typename T>
void RmaInterface<T>::Axpy( T alpha, Matrix<T>& Z, Int i, Int j )
{
    DEBUG_ONLY(CallStackEntry cse("RmaInterface::Axpy"))
    if( attachedForLocalToGlobal_ )
        RmaLocalToGlobal( alpha, Z, i, j );
    else if( attachedForGlobalToLocal_ )
        RmaGlobalToLocal( alpha, Z, i, j );
    else
        LogicError("Cannot RmaAxpy before attaching.");
}

template<typename T>
void RmaInterface<T>::Axpy( T alpha, const Matrix<T>& Z, Int i, Int j )
{
    DEBUG_ONLY(CallStackEntry cse("RmaInterface::Axpy"))
    if( attachedForLocalToGlobal_ )
        RmaLocalToGlobal( alpha, Z, i, j );
    else if( attachedForGlobalToLocal_ )
        LogicError("Cannot update a constant matrix.");
    else
        LogicError("Cannot RmaAxpy before attaching.");
}

// Update Y(i:i+height-1,j:j+width-1) += alpha X, where X is height x width
template<typename T>
void RmaInterface<T>::RmaLocalToGlobal
( T alpha, const Matrix<T>& X, Int i, Int j )
{
    DEBUG_ONLY(CallStackEntry cse("RmaInterface::RmaLocalToGlobal"))
#if 0
    DistMatrix<T>& Y = *localToGlobalMat_;
    if( i < 0 || j < 0 )
        LogicError("Submatrix offsets must be non-negative");
    if( i+X.Height() > Y.Height() || j+X.Width() > Y.Width() )
        LogicError("Submatrix out of bounds of global matrix");

    const Grid& g = Y.Grid();
    const Int r = g.Height();
    const Int c = g.Width();
    const Int p = g.Size();
    const Int myProcessRow = g.Row();
    const Int myProcessCol = g.Col();
    const Int colAlign = (Y.ColAlign() + i) % r;
    const Int rowAlign = (Y.RowAlign() + j) % c;

    const Int height = X.Height();
    const Int width = X.Width();

    Int receivingRow = myProcessRow;
    Int receivingCol = myProcessCol;
    for( Int step=0; step<p; ++step )
    {
        const Int colShift = Shift( receivingRow, colAlign, r );
        const Int rowShift = Shift( receivingCol, rowAlign, c );
        const Int localHeight = Length( height, colShift, r );
        const Int localWidth = Length( width, rowShift, c );
        const Int numEntries = localHeight*localWidth;

        if( numEntries != 0 )
        {
            const Int destination = receivingRow + r*receivingCol;
            const Int bufferSize = 4*sizeof(Int) + (numEntries+1)*sizeof(T);

            const Int index = 
                ReadyForSend
                ( bufferSize, dataVectors_[destination], 
                  dataSendRequests_[destination], sendingData_[destination] );
            DEBUG_ONLY(
                if( Int(dataVectors_[destination][index].size()) != bufferSize )
                    LogicError("Error in ReadyForSend");
            )

            // Pack the header
            byte* sendBuffer = dataVectors_[destination][index].data();
            byte* head = sendBuffer;
            *reinterpret_cast<Int*>(head) = i; head += sizeof(Int);
            *reinterpret_cast<Int*>(head) = j; head += sizeof(Int);
            *reinterpret_cast<Int*>(head) = height; head += sizeof(Int);
            *reinterpret_cast<Int*>(head) = width; head += sizeof(Int);
            *reinterpret_cast<T*>(head) = alpha; head += sizeof(T);

            // Pack the payload
            T* sendData = reinterpret_cast<T*>(head);
            const T* XBuffer = X.LockedBuffer();
            const Int XLDim = X.LDim();
            for( Int t=0; t<localWidth; ++t )
            {
                T* thisSendCol = &sendData[t*localHeight];
                const T* thisXCol = &XBuffer[(rowShift+t*c)*XLDim];
                for( Int s=0; s<localHeight; ++s )
                    thisSendCol[s] = thisXCol[colShift+s*r];
            }

            // Fire off the non-blocking send
            mpi::TaggedISSend
            ( sendBuffer, bufferSize, destination, DATA_TAG, g.VCComm(), 
              dataSendRequests_[destination][index] );
        }

        receivingRow = (receivingRow + 1) % r;
        if( receivingRow == 0 )
            receivingCol = (receivingCol + 1) % c;
    }
#endif
}

// Update Y += alpha X(i:i+height-1,j:j+width-1), where X is the dist-matrix
template<typename T>
void RmaInterface<T>::RmaGlobalToLocal( T alpha, Matrix<T>& Y, Int i, Int j )
{
    DEBUG_ONLY(CallStackEntry cse("RmaInterface::RmaGlobalToLocal"))
#if 0
    const DistMatrix<T>& X = *globalToLocalMat_;

    const Int height = Y.Height();
    const Int width = Y.Width();
    if( i+height > X.Height() || j+width > X.Width() )
        LogicError("Invalid RmaGlobalToLocal submatrix");

    const Grid& g = X.Grid();
    const Int r = g.Height();
    const Int c = g.Width();
    const Int p = g.Size();

    // Send out the requests to all processes in the grid
    for( Int rank=0; rank<p; ++rank )
    {
        const Int bufferSize = 4*sizeof(Int);
        const Int index = 
            ReadyForSend
            ( bufferSize, requestVectors_[rank], 
              requestSendRequests_[rank], sendingRequest_[rank] );

        // Copy the request header into the send buffer
        byte* sendBuffer = requestVectors_[rank][index].data();
        byte* head = sendBuffer;
        *reinterpret_cast<Int*>(head) = i; head += sizeof(Int);
        *reinterpret_cast<Int*>(head) = j; head += sizeof(Int);
        *reinterpret_cast<Int*>(head) = height; head += sizeof(Int);
        *reinterpret_cast<Int*>(head) = width; head += sizeof(Int);

        // Begin the non-blocking send
        mpi::TaggedISSend
        ( sendBuffer, bufferSize, rank, DATA_REQUEST_TAG, g.VCComm(), 
          requestSendRequests_[rank][index] );
    }

    // Receive all of the replies
    Int numReplies = 0;
    while( numReplies < p )
    {
        HandleGlobalToLocalRequest();

        mpi::Status status;
        if( mpi::IProbe( mpi::ANY_SOURCE, DATA_REPLY_TAG, g.VCComm(), status ) )
        {
            const Int source = status.MPI_SOURCE;

            // Ensure that we have a recv buffer
            const Int count = mpi::GetCount<byte>( status );
            recvVector_.resize( count );
            byte* recvBuffer = recvVector_.data();

            // Receive the data
            mpi::TaggedRecv
            ( recvBuffer, count, source, DATA_REPLY_TAG, g.VCComm() );

            // Unpack the reply header
            const byte* head = recvBuffer;
            const Int row = *reinterpret_cast<const Int*>(head); 
            head += sizeof(Int);
            const Int col = *reinterpret_cast<const Int*>(head); 
            head += sizeof(Int);
            const T* recvData = reinterpret_cast<const T*>(head);

            // Compute the local heights and offsets
            const Int colAlign = (X.ColAlign()+i) % r;
            const Int rowAlign = (X.RowAlign()+j) % c;
            const Int colShift = Shift( row, colAlign, r );
            const Int rowShift = Shift( col, rowAlign, c );
            const Int localHeight = Length( height, colShift, r );
            const Int localWidth = Length( width, rowShift, c );

            // Unpack the local matrix
            for( Int t=0; t<localWidth; ++t )
            {
                T* YCol = Y.Buffer(0,rowShift+t*c);
                const T* XCol = &recvData[t*localHeight];
                for( Int s=0; s<localHeight; ++s )
                    YCol[colShift+s*r] += alpha*XCol[s];
            }

            ++numReplies;
        }
    }
#endif
}

template<typename T>
void RmaInterface<T>::Detach()
{
    DEBUG_ONLY(CallStackEntry cse("RmaInterface::Detach"))
    if( !attachedForLocalToGlobal_ && !attachedForGlobalToLocal_ )
        LogicError("Must attach before detaching.");

    const Grid& g = ( attachedForLocalToGlobal_ ? 
                      localToGlobalMat_->Grid() : 
                      globalToLocalMat_->Grid() );

    while( !Finished() )
    {
        if( attachedForLocalToGlobal_ )
            HandleLocalToGlobalData();
        else
            HandleGlobalToLocalRequest();
        HandleEoms();
    }

    mpi::Barrier( g.VCComm() );

    attachedForLocalToGlobal_ = false;
    attachedForGlobalToLocal_ = false;
}

template class RmaInterface<Int>;
template class RmaInterface<float>;
template class RmaInterface<double>;
template class RmaInterface<Complex<float>>;
template class RmaInterface<Complex<double>>;

} // namespace El
