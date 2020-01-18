-- |= Binarized Neural Network Building Blocks

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
  , softmax_
  , flatten
  , linear
  , forward
  , binaryNet
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
  , randNetwork
  , rand
  , randn
  , iterN
  , br
  ) where

import           Control.Applicative ( liftA2 )
import           Control.DeepSeq ( NFData )
import           Control.Monad ( foldM )
import           Data.List ( maximumBy )
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

-- Batchnorm adaptation for BNNs: mean normalization
data BatchNorm1d' a = BatchNorm1d' { _beta :: Vector a  -- Learnable parameter
                                   , _runningMean :: Vector a  -- Running mean
                                   }
  deriving (Show, Generic)

instance NFData (Linear a)
-- makeLenses ''Linear

instance NFData (LinearB a)

instance NFData (BatchNorm1d' a)

data BNN a =
    BNN { _fc1 :: !(LinearB a)
        , _bn1 :: !(BatchNorm1d' a)
        , _fc2 :: !(LinearB a)
        , _bn2 :: !(BatchNorm1d' a)
        , _fc3 :: !(LinearB a)
        , _bn3 :: !(BatchNorm1d' a)
        }
  deriving (Show, Generic)

makeLenses ''BNN

-- Inputs are NOT binary. Binary weights
linear' :: Reifies s W
        => BVar s (LinearB Float)
        -> BVar s (Matrix Float)
        -> BVar s (Matrix Float)
linear' = undefined

-- Both inputs and weights are binary
linearB :: Reifies s W
        => BVar s (LinearB Float)
        -> BVar s (Matrix Float)
        -> BVar s (Matrix Float)
linearB = undefined

-- Mean normalization
bn' :: Reifies s W
    => BVar s (BatchNorm1d' Float)
    -> BVar s (Matrix Float)
    -> BVar s (Matrix Float)
bn' = undefined

binaryNet
    :: (Reifies s W)
    => BVar s (BNN Float)
    -> Volume4 Float  -- ^ Batch of MNIST images
    -> BVar s (Matrix Float)
binaryNet l = constVar
            ~> flatten

            -- Layer #1
            ~> linear' (l ^^. fc1)
            ~> bn' (l ^^. bn1)
            ~> sign

            -- Layer #2
            ~> linearB (l ^^. fc2)
            ~> bn' (l ^^. bn2)
            ~> sign

            -- Layer #3
            ~> linearB (l ^^. fc3)
            ~> bn' (l ^^. bn3)
{-# INLINE binaryNet #-}

infixl 9 ~>
(~>) :: (a -> b) -> (b -> c) -> a -> c
f ~> g = g. f
{-# INLINE (~>) #-}

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

instance (Num a, Unbox a) => Num (BatchNorm1d' a) where
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

instance (Num a, Unbox a) => Backprop (Linear a)
instance (Num a, Unbox a) => Backprop (LinearB a)
instance (Num a, Unbox a) => Backprop (BatchNorm1d' a)
instance (Num a, Unbox a) => Backprop (Net a)

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
computeMap' :: (Float -> Float) -> BNN Float -> BNN Float
computeMap' f BNN { _fc1 = LinearB w1
                  , _bn1 = BatchNorm1d' beta1 runningMean1
                  , _fc2 = LinearB w2
                  , _bn2 = BatchNorm1d' beta2 runningMean2
                  , _fc3 = LinearB w3
                  , _bn3 = BatchNorm1d' beta3 runningMean3
                  } = BNN { _fc1 = LinearB (computeMap f w1)
                          , _bn1 = BatchNorm1d' (computeMap f beta1) runningMean1
                          , _fc2 = LinearB (computeMap f w2)
                          , _bn2 = BatchNorm1d' (computeMap f beta2) runningMean2
                          , _fc3 = LinearB (computeMap f w3)
                          , _bn3 = BatchNorm1d' (computeMap f beta3) runningMean3
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
