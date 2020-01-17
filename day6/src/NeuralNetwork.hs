-- |= Convolutional Neural Network Building Blocks
--
-- Note that some functions have been updated w.r.t massiv-0.4.4.0,
-- most notably changed Data.Massiv.Array.Numeric
--
-- This work was largely inspired by
-- https://github.com/mstksg/backprop/blob/master/samples/backprop-mnist.lhs

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
{-# OPTIONS_GHC -fno-warn-partial-type-signatures #-}

module NeuralNetwork
  ( BNN
  , Vector
  , Matrix
  , Volume
  , Volume4
  , sigmoid
  , sign
  , sign_
  , relu
  , relu_
  , conv2d
  , conv2d_
  , conv2d'
  , conv2d''
  , maxpool
  , maxpool_
  , softmax_
  , flatten
  , linear
  , forward
  , binaryNet
  , noPad2
  , (~>)

  -- * Training
  , sgd

  -- * Inference
  , accuracy
  , avgAccuracy
  , winnerTakesAll

  -- * Helpers
  , rows
  , cols
  , sumRows
  , sumCols
  , computeMap
  , randLinear
  , randConv2d
  , randNetwork
  , rand
  , randn
  , iterN
  , br
  ) where

import           Control.Applicative ( liftA2 )
import           Control.DeepSeq ( NFData )
import           Control.Monad ( foldM )
import           Data.List ( foldl', maximumBy )
import           Data.Massiv.Array hiding ( map, zip, zipWith, flatten )
import qualified Data.Massiv.Array as A
import           Data.Ord
import           GHC.Generics ( Generic )
-- import           Lens.Micro
import           Lens.Micro.TH
import           Numeric.Backprop
import           Numeric.Backprop.Class ( addNum )
import           Numeric.OneLiner
import           Streamly
import qualified Streamly.Prelude as S
import qualified System.Random.MWC as MWC
import           System.Random.MWC ( createSystemRandom )
import           System.Random.MWC.Distributions ( standard )

-- Note that images are volumes of channels x width x height, whereas
-- mini-batches are volumes-4 of batch size x channels x width x height.
-- Similarly, convolutional filter weights are volumes-4 of
-- out channels x in channels x kernel width x kernel height.
type Vector a = Array U Ix1 a
type Matrix a = Array U Ix2 a
type Volume a = Array U Ix3 a
type Volume4 a = Array U Ix4 a

-- | Learnable neural network parameters.
-- Fully-connected layer weights.
data Linear a = Linear { _weights :: !(Matrix a)
                       , _biases :: !(Vector a)
                       }
  deriving (Show, Generic)

newtype LinearB a = LinearB { _weights' :: Matrix a }
  deriving (Show, Generic)

data BatchNorm1d a = BatchNorm1d { _gamma :: Vector a  -- TODO: add running mean and variance
                                 , _beta :: Vector a
                                 }
  deriving (Show, Generic)

-- | Convolutional layer weights. We omit biases for simplicity.
newtype Conv2d a = Conv2d { _kernels :: Volume4 a }
  deriving (Show, Generic)

instance NFData (Linear a)
-- makeLenses ''Linear

instance NFData (Conv2d a)
-- makeLenses ''Conv2d

data BNN a =
    BNN { _fc1 :: !(LinearB a)
        , _bn1 :: !(BatchNorm1d a)
        , _fc2 :: !(LinearB a)
        , _bn2 :: !(BatchNorm1d a)
        , _fc3 :: !(LinearB a)
        , _bn3 :: !(BatchNorm1d a)
        }
  deriving (Show, Generic)

makeLenses ''BNN

sameConv2d :: Reifies s W
    => BVar s (Conv2d Float)
    -> BVar s (Volume4 Float)
    -> BVar s (Volume4 Float)
sameConv2d = conv2d (Padding (Sz2 2 2) (Sz2 2 2) (Fill 0.0))

validConv2d :: Reifies s W
    => BVar s (Conv2d Float)
    -> BVar s (Volume4 Float)
    -> BVar s (Volume4 Float)
validConv2d = conv2d noPad2

-- Inputs are NOT binary. Binary weights
linear' = undefined

-- Both inputs and weights are binary
linearB = undefined

binaryNet
    :: (Reifies s W)
    => BVar s (BNN Float)
    -> Volume4 Float  -- ^ Batch of MNIST images
    -> BVar s (Matrix Float)
binaryNet l = undefined
-- binaryNet l = constVar
--             ~> flatten
--
--             -- Classifier
--             -- Layer #3
--             ~> linear' (l ^^. fc1)
--             ~> sign
--             -- Layer #4
--             ~> linearB (l ^^. fc2)
--             ~> sign
--             -- Layer #5
--             ~> linearB (l ^^. fc3)
{-# INLINE binaryNet #-}

infixl 9 ~>
(~>) :: (a -> b) -> (b -> c) -> a -> c
f ~> g = g. f
{-# INLINE (~>) #-}

noPad2 :: Padding Ix2 Float
noPad2 = Padding (Sz2 0 0) (Sz2 0 0) (Fill 0.0)

type Net = BNN

-- We would like to be able to perform arithmetic
-- operations over parameters, e.g. in SDG implementation.
-- Therefore, we define the Num instance.
instance (Num a, Unbox a, Index ix) => Num (Array U ix a) where
    x + y       = maybe (error $ "Dimension mismatch " ++ show (size x, size y)) compute (delay x .+. delay y)
    x - y       = maybe (error $ "Dimension mismatch " ++ show (size x, size y)) compute (delay x .-. delay y)
    x * y       = maybe (error $ "Dimension mismatch " ++ show (size x, size y)) compute (delay x .*. delay y)
    negate      = computeMap negate
    -- Maybe define later, when we will actually need those
    abs         = error "Please define abs"
    signum      = error "Please define signum"
    fromInteger = error "Please define me"

instance (Num a, Unbox a) => Num (Conv2d a) where
    (+)         = gPlus
    (-)         = gMinus
    (*)         = gTimes
    negate      = gNegate
    abs         = gAbs
    signum      = gSignum
    fromInteger = gFromInteger

instance (Num a, Unbox a) => Num (Linear a) where
    (+)         = gPlus
    (-)         = gMinus
    (*)         = gTimes
    negate      = gNegate
    abs         = gAbs
    signum      = gSignum
    fromInteger = gFromInteger

instance (Num a, Unbox a) => Num (LinearB a) where
    (+)         = gPlus
    (-)         = gMinus
    (*)         = gTimes
    negate      = gNegate
    abs         = gAbs
    signum      = gSignum
    fromInteger = gFromInteger

instance (Num a, Unbox a) => Num (BatchNorm1d a) where
    (+)         = gPlus
    (-)         = gMinus
    (*)         = gTimes
    negate      = gNegate
    abs         = gAbs
    signum      = gSignum
    fromInteger = gFromInteger

instance ( Unbox a
         , Num a
         ) => Num (Net a) where
    (+)         = gPlus
    (-)         = gMinus
    (*)         = gTimes
    negate      = gNegate
    abs         = gAbs
    signum      = gSignum
    fromInteger = gFromInteger

-- instance (Num a, Unbox a) => Fractional (Conv2d a) where
--     (/)          = error "Please define Conv2d (/)"
--     recip        = error "Please define Conv2d recip"
--     fromRational = error "Please define Conv2d fromRational (introduce Conv2d a i o)"
--
-- instance (Num a, Unbox a) => Fractional (Linear a) where
--     (/)          = error "Please define Linear (/)"
--     recip        = error "Please define Linear recip"
--     fromRational = error "Please define Linear fromRational (introduce Linear a i o)"
--
-- instance ( Num a
--          , Unbox a
--          ) => Fractional (Net a) where
--     (/)          = gDivide
--     recip        = gRecip
--     fromRational = gFromRational

instance (Num a, Unbox a) => Backprop (Conv2d a)
instance (Num a, Unbox a) => Backprop (Linear a)
instance (Num a, Unbox a) => Backprop (LinearB a)
instance (Num a, Unbox a) => Backprop (BatchNorm1d a)
instance (Num a, Unbox a) => Backprop (Net a)

-- | 2D convolution that operates on a batch.
--
-- Padding is Ix2 because it is performed only w.r.t. width and height.
--
-- The stride is assumed to be 1, but this can be extended
-- in a straightforward manner with `computeWithStride`.
-- Do not forget to use strides 1 in batch and channel dimensions (Dim4, Dim3).
conv2d_
       :: Padding Ix2 Float  -- ^ Image plane padding (p1 :. p2) (q1 :. q2)
       -> Volume4 Float  -- ^ Weights (cout :> cin :> w1 :> w2)
       -> Volume4 Float  -- ^ Batch of input features (bs :> cin :> x1 :. x2)
       -> Volume4 Float  -- ^ Output features
                         -- (bs :> cout :> x1 - w1 + p1 + q1 + 1 :> x2 - w2 + p2 + q2 + 1)
conv2d_ (Padding (Sz szp1) (Sz szp2) be) w x = res
  where
    (Sz (cout :> cin :> w1 :. w2)) = size w
    -- Extract weights, add fake Dim4, and make stencil
    sten = makeCorrelationStencilFromKernel. resize' (Sz4 1 cin w1 w2). (w !>)
    {-# INLINE sten #-}
    -- Add zeroes in batch and channel dimensions
    pad4 = Padding (Sz (0 :> 0 :> szp1)) (Sz (0 :> 0 :> szp2)) be
    -- Note: we apply stencils on zero channel of *all* images in the batch
    base = computeAs U $ applyStencil pad4 (sten 0) x
    -- Again, stencils are applied simultaneously on all images for a given
    -- channel
    res = foldl' (\prev ch -> let conv = computeAs U $ applyStencil pad4 (sten ch) x
                              in computeAs U $ append' 3 prev conv) base [1..cout - 1]

-- | Input gradients
--
-- \[ dX = \delta (*) W_{flip}, \]
--
-- where (*) is cross-correlation. We also have to perform inner transpose
-- over the kernel volume since we are propagating errors in the backward
-- direction.
conv2d'
  :: Padding Ix2 Float -> Volume4 Float -> Volume4 Float -> Volume4 Float
conv2d' p w dz = conv2d_ p w' dz
  where
    w' = (compute. rot180. transposeInner) w

    -- Rotate image kernels by 180 degrees
    rot180 = reverse' 1. reverse' 2
    {-# INLINE rot180 #-}

-- | Kernel gradients
--
-- \[ dW = X (*) delta \]
conv2d''
  :: Padding Ix2 Float -> Volume4 Float -> Volume4 Float -> Volume4 Float
-- Iterate over images in a batch
conv2d'' p x dz = res  -- computeMap (/fromIntegral bs) res
  where
    -- Assume bs1 == bs2
    Sz (bs :> _) = size x
    -- (sum. zipWith) in 5D
    dzd = delay dz
    xd = delay x
    base = _conv2d'' p (compute $ dzd !> 0) (compute $ xd !> 0)
    -- Accumulate gradients over the batch
    res = foldl' (\acc im ->
      let cur = _conv2d'' p (compute $ dzd !> im) (compute $ xd !> im)
       in acc + cur) base [1..bs-1]

-- Iterate over out channels, i.e. dW has the same outer dimension as dZ (after
-- image indexing)
_conv2d'' :: Padding Ix2 Float
          -> Volume Float
          -> Volume Float
          -> Volume4 Float
_conv2d'' (Padding (Sz szp1) (Sz szp2) pb) dz x = res
  where
    (Sz (cout :> dz1 :. dz2)) = size dz
    sten i =
      let kernel2d = dz !> i
          kernel3d = computeAs U $ resize' (Sz3 1 dz1 dz2) kernel2d :: Volume Float
       in makeCorrelationStencilFromKernel kernel3d
    {-# INLINE sten #-}
    pad3 = Padding (Sz (0 :> szp1)) (Sz (0 :> szp2)) pb
    base0 = applyStencil pad3 (sten 0) x

    Sz szW0 = size base0
    szW = Sz (1 :> szW0)
    rsz = resize' szW. computeAs U

    base = rsz base0
    res = foldl' (\prev ch -> let conv = rsz. applyStencil pad3 (sten ch) $ x
                              in computeAs U $ append' 4 prev conv) base [1..cout - 1]
{-# INLINE _conv2d'' #-}

-- | Differentiable 2D convolutional layer
conv2d :: Reifies s W
       => Padding Ix2 Float
       -> BVar s (Conv2d Float)
       -> BVar s (Volume4 Float)
       -> BVar s (Volume4 Float)
conv2d p@(Padding (Sz (p11 :. p12)) (Sz (p21 :. p22)) be) = liftOp2. op2 $ \(Conv2d w) x ->
    let conv = conv2d_ p w x
     in (conv, \dz -> let dw = conv2d'' p x dz
                          Sz (_ :> _ :> h0 :. w0) = size x
                          Sz (_ :> _ :> h1 :. w1) = size conv
                          ch = h0 - h1
                          cw = w0 - w1
                          szp1' = (p11 + ch) :. (p12 + cw)
                          szp2' = (p21 + ch) :. (p22 + cw)
                          p1 = Padding (Sz szp1') (Sz szp2') be
                          dx = conv2d' p1 w dz
                       in (Conv2d dw, dx) )

instance (Index ix, Num e, Unbox e) => Backprop (Array U ix e) where
    zero x = A.replicate Par (size x) 0
    add = addNum  -- Making use of Num Array instance
    one x = A.replicate Par (size x) 1

-- | Linear layer
linear :: Reifies s W
       => BVar s (Linear Float)
       -> BVar s (Matrix Float)
       -> BVar s (Matrix Float)
linear = liftOp2. op2 $ \(Linear w b) x ->
  let prod = maybe (error $ "Dimension mismatch " ++ show (size x, size w)) id (x |*| w)
      lin = maybe (error "Dimension mismatch") compute (delay prod .+. (b `rowsLike` x))
  in (lin, \dZ -> let dW = linearW' x dZ
                      dB = bias' dZ
                      dX = linearX' w dZ
                  in (Linear dW dB, dX)
     )

-- | Sign activation
sign_ :: (Index ix, Unbox e, Ord e, Num e) => Array U ix e -> Array U ix e
sign_ = computeMap f
  where
    f x = if x <= 0
             then -1
             else 1

-- | Sign gradient approximation
sign' :: (Index ix, Unbox e, Ord e, Num e)
      => Array U ix e
      -> Array U ix e
      -> Array U ix e
sign' x = compute. A.zipWith f x
  where
    f x0 dy0 = if (x0 > (-1)) && (x0 < 1)
                  then dy0
                  else 0

sign :: (Reifies s W, Index ix)
     => BVar s (Array U ix Float)
     -> BVar s (Array U ix Float)
sign = liftOp1. op1 $ \x ->
  (sign_ x, sign' x)

relu_ :: (Index ix, Unbox e, Ord e, Num e) => Array U ix e -> Array U ix e
relu_ = computeMap (max 0)

relu :: (Reifies s W, Index ix)
     => BVar s (Array U ix Float)
     -> BVar s (Array U ix Float)
relu = liftOp1. op1 $ \x ->
  (relu_ x, \dY ->
    let f x0 dy0 = if x0 <= 0
                      then 0
                      else dy0
     in compute $ A.zipWith f x dY)

-- | Elementwise sigmoid with gradients
sigmoid :: forall s ix. (Reifies s W, Index ix)
        => BVar s (Array U ix Float)
        -> BVar s (Array U ix Float)
sigmoid = liftOp1. op1 $ \x ->
    let y = computeMap f x
    in (y, \dY ->
        let ones = delay $ one x
            y' = delay y
        in either throw compute $ do
            e1 <- ones .-. y'
            e2 <- y' .*. e1
            delay dY .*. e2
       )
  where
    f x = recip $ 1.0 + exp (-x)

maxpoolStencil2x2 :: Stencil Ix4 Float Float
maxpoolStencil2x2 = makeStencil (Sz4 1 1 2 2) 0 $ \ get -> let max4 x1 x2 x3 x4 = max (max (max x1 x2) x3) x4 in max4 <$> get (0 :> 0 :> 0 :. 0) <*> get (0 :> 0 :> 1 :. 1) <*> get (0 :> 0 :> 0 :. 1) <*> get (0 :> 0 :> 1 :. 0)

maxpool_ :: Volume4 Float -> Volume4 Float
maxpool_ = computeWithStride (Stride (1 :> 1 :> 2 :. 2)). applyStencil noPadding maxpoolStencil2x2

-- > testA = fromLists' Seq [[[[1..4],[5..8],[9..12],[13..16]]]] :: Array U Ix4 Float
--
-- > testA
-- Array U Seq (Sz (1 :> 1 :> 4 :. 4))
--   [ [ [ [ 1.0, 2.0, 3.0, 4.0 ]
--       , [ 5.0, 6.0, 7.0, 8.0 ]
--       , [ 9.0, 10.0, 11.0, 12.0 ]
--       , [ 13.0, 14.0, 15.0, 16.0 ]
--       ]
--     ]
--   ]
-- > maxpool_ testA
-- Array U Seq (Sz (1 :> 1 :> 2 :. 2))
--   [ [ [ [ 6.0, 8.0 ]
--       , [ 14.0, 16.0 ]
--       ]
--     ]
--   ]
-- > testB = resize' (Sz4 2 1 2 4) testA
-- > testB
-- Array U Seq (Sz (2 :> 1 :> 2 :. 4))
--   [ [ [ [ 1.0, 2.0, 3.0, 4.0 ]
--       , [ 5.0, 6.0, 7.0, 8.0 ]
--       ]
--     ]
--   , [ [ [ 9.0, 10.0, 11.0, 12.0 ]
--       , [ 13.0, 14.0, 15.0, 16.0 ]
--       ]
--     ]
--   ]
-- > maxpool_ testB
-- Array U Seq (Sz (2 :> 1 :> 1 :. 2))
--   [ [ [ [ 6.0, 8.0 ]
--       ]
--     ]
--   , [ [ [ 14.0, 16.0 ]
--       ]
--     ]
--   ]

maxpool :: Reifies s W
        => BVar s (Volume4 Float)
        -> BVar s (Volume4 Float)
maxpool = liftOp1. op1 $ \x ->
  let out = maxpool_ x
      s = Stride (1 :> 1 :> 2 :. 2)
      outUp = computeAs U $ zoom s out
      maxima = A.zipWith (\a b -> if a == b then 1 else 0) outUp x
  in (out, \dz -> let dzUp = computeAs U $ zoom s dz
                  in maybe (error $ "Dimensions problem " ++ show (size out, size dz, size outUp, size maxima, size dzUp)) compute (maxima .*. delay dzUp))

-- Test maxpool gradients
-- testC :: Volume4 Float
-- testC =
--   let a0 = A.fromLists' Par [[1,7],[3,4]] :: Matrix Float
--       b = resize' (Sz4 1 1 2 2) a0
--    in gradBP maxpool b
-- Array U Par (Sz (1 :> 1 :> 2 :. 2))
--   [ [ [ [ 0.0, 1.0 ]
--       , [ 0.0, 0.0 ]
--       ]
--     ]
--   ]
--
flatten :: Reifies s W
        => BVar s (Volume4 Float)
        -> BVar s (Matrix Float)
flatten = liftOp1. op1 $ \x ->
  let sz0@(Sz (bs :> ch :> h :. w)) = size x
      sz = Sz2 bs (ch * h * w)
   in (resize' sz x, resize' sz0)

-- | A neural network may work differently in training and evaluation modes
data Phase = Train | Eval deriving (Show, Eq)

-- | Uniformly-distributed random numbers Array
rand
  :: (Mutable r ix e, MWC.Variate e) =>
     (e, e) -> Sz ix -> IO (Array r ix e)
rand rng sz = do
    gens <- initWorkerStates Par (const createSystemRandom)
    randomArrayWS gens sz (MWC.uniformR rng)

-- | Random values from the Normal distribution
randn :: forall e ix. (Fractional e, Index ix, Unbox e) => Sz ix -> IO (Array U ix e)
randn sz = do
    gens <- initWorkerStates Par (const createSystemRandom)
    r <- randomArrayWS gens sz standard :: IO (Array P ix Double)
    return (compute $ A.map realToFrac r)

rows :: Matrix Float -> Int
rows m =
  let (r :. _) = unSz $ size m
  in r

cols :: Matrix Float -> Int
cols m =
  let (_ :. c) = unSz $ size m
  in c

computeMap :: (Source r2 ix e', Mutable r1 ix e) =>
  (e' -> e) -> Array r2 ix e' -> Array r1 ix e
computeMap f = A.compute. A.map f
{-# INLINE computeMap #-}

linearW' :: Matrix Float
        -> Matrix Float
        -> Matrix Float
linearW' x dy =
  let trX = compute $ transpose x :: Matrix Float
      prod = maybe (error "Inconsistent dimensions in linearW'") id (trX |*| dy)
      m = recip $ fromIntegral (rows x)
  in compute $ m *. delay prod

linearX' :: Matrix Float
        -> Matrix Float
        -> Matrix Float
linearX' w dy = maybe (error "Inconsistent dimensions in linearX'") compute (dy `multiplyTransposed` w)

-- | Bias gradient
bias' :: Matrix Float -> Vector Float
bias' dY = compute $ m *. sumRows_ dY
  where
    m = recip $ fromIntegral $ rows dY

-- | Forward pass in a neural network (inference)
forward :: Net Float -> Volume4 Float -> Matrix Float
forward net dta = evalBP (`binaryNet` dta) net

softmax_ :: Matrix Float -> Matrix Float
softmax_ x =
  let x0 = expA (delay x)
      x1 = computeAs U $ sumCols_ x0  -- Note sumCols_, not sumRows_
      x2 = x1 `colsLike` x
  in maybe (error  "Inconsistent dimensions in softmax_") compute (x0 ./. x2)

crossEntropyLoss
  :: forall s. (Reifies s W)
  => Volume4 Float
  -> Matrix Float
  -> BVar s (Net Float)
  -> BVar s (Matrix Float)
crossEntropyLoss x targ n = _ce y
  where
    zeros = A.replicate Par (size targ) 0.0 :: Matrix Float
    y = binaryNet n x :: BVar s (Matrix Float)
    _ce :: BVar s (Matrix Float) -> BVar s (Matrix Float)
    -- Gradients only
    _ce = liftOp1. op1 $ \y_ ->
      (zeros  -- Lazy to implement for now
      , \_ -> softmax_ y_ - targ  -- Gradients
      )
{-# INLINE crossEntropyLoss #-}

-- | Broadcast a vector in Dim2
rowsLike :: Manifest r Ix1 Float
         => Array r Ix1 Float -> Matrix Float -> Array D Ix2 Float
rowsLike v m = br (Sz (rows m)) v

-- | Broadcast a vector in Dim1
colsLike :: Manifest r Ix1 Float
         => Array r Ix1 Float -> Matrix Float -> Array D Ix2 Float
colsLike v m = br1 (Sz (cols m)) v

-- | Broadcast by the given number of rows
br :: Manifest r Ix1 Float
   => Sz1 -> Array r Ix1 Float -> Array D Ix2 Float
br rows' = expandWithin Dim2 rows' const

-- | Broadcast by the given number of cols
br1 :: Manifest r Ix1 Float
   => Sz1 -> Array r Ix1 Float -> Array D Ix2 Float
br1 rows' = expandWithin Dim1 rows' const

-- | Stochastic gradient descent
sgd :: Monad m
  => Float
  -- ^ Learning rate
  -> Int
  -- ^ No of iterations
  -> Net Float
  -- ^ Neural network
  -> SerialT m (Volume4 Float, Matrix Float)
  -- ^ Data stream
  -> m (Net Float)
sgd lr n net0 dataStream = iterN n epochStep net0
  where
    -- Iterate over all batches
    epochStep net = S.foldl' _trainStep net dataStream
    -- Update gradients based on a single batch
    _trainStep net (x, targ) = trainStep lr x targ net
    {-# INLINE _trainStep #-}

-- | Gradient descent step
trainStep
  :: Float  -- ^ Learning rate
  -> Volume4 Float  -- ^ Images batch
  -> Matrix Float  -- ^ Targets
  -> Net Float  -- ^ Initial network
  -> Net Float
trainStep lr !x !targ !n = n - computeMap' (lr *) (gradBP (crossEntropyLoss x targ) n)
{-# INLINE trainStep #-}

-- This could be improved:
-- The problem is that realToFrac does not know about the shape.
-- This can be solved having that information on the type level.
-- Conv2d a i o k, i = in channels, o = out channels, k = square kernel size
-- and Linear a i o, i = inputs, o = outputs.
-- trainStep lr !x !targ !n = n - realToFrac lr * gradBP (loss x targ) n
computeMap' :: (Float -> Float) -> BNN Float -> BNN Float
computeMap' f BNN { _fc1 = LinearB w1
                  , _bn1 = BatchNorm1d gamma1 beta1
                  , _fc2 = LinearB w2
                  , _bn2 = BatchNorm1d gamma2 beta2
                  , _fc3 = LinearB w3
                  , _bn3 = BatchNorm1d gamma3 beta3
                  } = BNN { _fc1 = LinearB (computeMap f w1)
                          , _bn1 = BatchNorm1d (computeMap f gamma1) (computeMap f beta1)
                          , _fc2 = LinearB (computeMap f w2)
                          , _bn2 = BatchNorm1d (computeMap f gamma2) (computeMap f beta2)
                          , _fc3 = LinearB (computeMap f w3)
                          , _bn3 = BatchNorm1d (computeMap f gamma3) (computeMap f beta3)
                          }

-- | Strict left fold
iterN :: Monad m => Int -> (a -> m a) -> a -> m a
iterN n f x0 = foldM (\x _ -> f x) x0 [1..n]

-- | Generate random weights and biases
randLinear :: Sz2 -> IO (Linear Float)
randLinear sz@(Sz2 _ nout) = do
  _w <- setComp Par <$> _genWeights sz
  _b <- setComp Par <$> _genBiases nout
  return (Linear _w _b)
    where
      _genBiases n = randn (Sz n)

_genWeights :: Index ix => Sz ix -> IO (Array U ix Float)
_genWeights sz = do
    a <- randn sz
    return $ compute (k *. delay a)
  where
    -- Weight scaling factor. Can also be dependent on `sz`.
    k = 0.01

-- | Generate random convolutional layer
randConv2d :: Sz4 -> IO (Conv2d Float)
randConv2d sz = do
  k <- setComp Par <$> _genWeights sz
  return (Conv2d k)

randLinearB = undefined

randNetwork :: IO (BNN Float)
randNetwork = do
  let [i, h1, h2, o] = [784, 1200, 400, 10]
  _fc1 <- randLinearB (Sz2 i h1)
  _fc2 <- randLinearB (Sz2 h1 h2)
  _fc3 <- randLinearB (Sz2 h2 o)
  return $
    BNN { _fc1 = _fc1
        , _bn1 = undefined
        , _fc2 = _fc2
        , _bn2 = undefined
        , _fc3 = _fc3
        , _bn3 = undefined
        }

maxIndex :: (Ord a, Num b, Enum b) => [a] -> b
maxIndex xs = snd $ maximumBy (comparing fst) (zip xs [0..])

winnerTakesAll ::
  Matrix Float  -- ^ Mini-batch of vectors
  -> [Int]  -- ^ List of maximal indices
winnerTakesAll m = map maxIndex xs
  where
    xs = toLists2 m

errors :: Eq lab => [(lab, lab)] -> [(lab, lab)]
errors = filter (uncurry (/=))
{-# SPECIALIZE errors :: [(Int, Int)] -> [(Int, Int)] #-}

accuracy :: (Eq a, Fractional acc) => [a] -> [a] -> acc
accuracy tgt pr = 100 * r
  where
    errNo = length $ errors (zip tgt pr)
    r = 1 - fromIntegral errNo / fromIntegral (length tgt)
{-# SPECIALIZE accuracy :: [Int] -> [Int] -> Float #-}

_accuracy :: Net Float
  -> (Volume4 Float, Matrix Float)
  -> Float
-- NB: better avoid double conversion to and from one-hot-encoding
_accuracy net (batch, labelsOneHot) =
  let batchResults = winnerTakesAll $ forward net batch
      expected = winnerTakesAll labelsOneHot
  in accuracy expected batchResults

avgAccuracy
  :: Monad m
  => Net Float
  -> SerialT m (Volume4 Float, Matrix Float)
  -> m Float
avgAccuracy net stream = s // len
  where
    results = S.map (_accuracy net) stream
    s = S.sum results
    len = fromIntegral <$> S.length results
    (//) = liftA2 (/)

-- | Sum values in each column and produce a delayed 1D Array
sumRows_ :: Source r Ix2 Float => Array r Ix2 Float -> Array D Ix1 Float
sumRows_ = A.foldlWithin Dim2 (+) 0.0

sumRows :: Reifies s W
    => BVar s (Matrix Float)
    -> BVar s (Vector Float)
sumRows = liftOp1. op1 $ \x ->
  (compute $ sumRows_ x, \dY -> compute $ dY `rowsLike` x)

-- | Sum values in each row and produce a delayed 1D Array
sumCols_ :: Source r Ix2 Float => Array r Ix2 Float -> Array D Ix1 Float
sumCols_ = A.foldlWithin Dim1 (+) 0.0

sumCols :: Reifies s W
    => BVar s (Matrix Float)
    -> BVar s (Vector Float)
sumCols = liftOp1. op1 $ \x ->
  (compute $ sumCols_ x, \dY -> compute $ dY `colsLike` x)
