-- |= Neural Network Building Blocks
--
-- The idea of this module is to manage gradients manually.
-- That is done intentionally to illustrate neural
-- networks training.

{-# LANGUAGE FlexibleContexts #-}

module NeuralNetwork
  ( NeuralNetwork
  , Layer(..)
  , Matrix
  , Vector
  , FActivation(..)
  , sigmoid
  , sigmoid'
  , genWeights
  , forward

  -- * Training
  , sgd

  -- * Inference
  , accuracy
  , avgAccuracy
  , inferBinary
  , winnerTakesAll

  -- * Helpers
  , rows
  , cols
  , computeMap
  , rand
  , randn
  , randomishArray
  , scale
  , iterN
  , mean
  , var
  , br
  )
where

import           Control.Monad                  ( replicateM
                                                , foldM
                                                )
import           Control.Applicative            ( liftA2 )
import qualified System.Random                 as R
import           System.Random.MWC              ( createSystemRandom )
import           System.Random.MWC.Distributions
                                                ( standard )
import           Data.List                      ( maximumBy )
import           Data.Ord
import           Data.Massiv.Array       hiding ( map
                                                , zip
                                                , zipWith
                                                )
import qualified Data.Massiv.Array             as A
import           Streamly
import qualified Streamly.Prelude              as S
import           Data.Maybe                     ( fromMaybe )

type MatrixPrim r a = Array r Ix2 a
type Matrix a = Array U Ix2 a
type Vector a = Array U Ix1 a


-- Activation function symbols:
-- * Rectified linear unit (ReLU)
-- * Sigmoid
-- * Identity (no activation)
data FActivation = Relu | Sigmoid | Id

-- Neural network layers: Linear, Batchnorm, Activation
data Layer a = Linear (Matrix a) (Vector a)
               -- Same as Linear, but without biases
               | Linear' (Matrix a)
               -- Batchnorm with running mean, variance, and two
               -- learnable affine parameters
               | Batchnorm1d (Vector a) (Vector a) (Vector a) (Vector a)
               | Activation FActivation

type NeuralNetwork a = [Layer a]

data Gradients a = -- Weight and bias gradients
                   LinearGradients (Matrix a) (Vector a)
                   -- Weight gradients
                   | Linear'Gradients (Matrix a)
                   -- Batchnorm parameters and gradients
                   | BN1 (Vector a) (Vector a) (Vector a) (Vector a)
                   | NoGrad  -- No learnable parameters

-- | A neural network may work differently in training and evaluation modes
data Phase = Train | Eval deriving (Show, Eq)

-- | Lookup activation function by a symbol
getActivation :: FActivation -> (Matrix Float -> Matrix Float)
getActivation Id      = id
getActivation Sigmoid = sigmoid
getActivation Relu    = relu

-- | Lookup activation function derivative by a symbol
getActivation' :: FActivation -> (Matrix Float -> Matrix Float -> Matrix Float)
getActivation' Id      = flip const
getActivation' Sigmoid = sigmoid'
getActivation' Relu    = relu'

-- | Elementwise sigmoid computation
sigmoid :: Matrix Float -> Matrix Float
sigmoid = computeMap f where f x = recip $ 1.0 + exp (-x)

-- | Compute sigmoid gradients
sigmoid' :: Matrix Float -> Matrix Float -> Matrix Float
sigmoid' x dY =
  let sz   = size x
      ones = A.replicate Par sz 1.0 :: Matrix Float
      y    = sigmoid x
  in  compute $ delay dY * delay y * (delay ones - delay y)

relu :: Matrix Float -> Matrix Float
relu = computeMap f where f x = if x < 0 then 0 else x

relu' :: Matrix Float -> Matrix Float -> Matrix Float
relu' x = compute . A.zipWith f x where f x0 dy0 = if x0 <= 0 then 0 else dy0

randomishArray
  :: (Mutable r ix e, R.RandomGen a, R.Random e)
  => (e, e)
  -> a
  -> Sz ix
  -> Array r ix e
randomishArray rng g0 sz = compute $ unfoldlS_ sz _rand g0
  where _rand g = let (a, g') = R.randomR rng g in (g', a)

-- | Uniformly-distributed random numbers Array
rand :: (R.Random e, Mutable r ix e) => (e, e) -> Sz ix -> IO (Array r ix e)
rand rng sz = do
  g <- R.newStdGen
  return $ randomishArray rng g sz

-- | Random values from the Normal distribution
randn
  :: (Fractional e, Index ix, Resize r Ix1, Mutable r Ix1 e)
  => Sz ix
  -> IO (Array r ix e)
randn sz = do
  g  <- createSystemRandom
  xs <- _nv g (totalElem sz)
  return $ resize' sz (fromList Seq xs)
 where
  _nv gen n = replicateM n (realToFrac <$> standard gen)
  {-# INLINE _nv #-}

rows :: Matrix Float -> Int
rows m = let (r :. _) = unSz $ size m in r

cols :: Matrix Float -> Int
cols m = let (_ :. c) = unSz $ size m in c

-- Returns a delayed Array. Useful for fusion
_scale :: (Num e, Source r ix e) => e -> Array r ix e -> Array D ix e
_scale c = A.map (* c)

scale :: Index sz => Float -> Array U sz Float -> Array U sz Float
scale konst = computeMap (* konst)

computeMap
  :: (Source r2 ix e', Mutable r1 ix e)
  => (e' -> e)
  -> Array r2 ix e'
  -> Array r1 ix e
computeMap f = A.compute . A.map f

linearW' :: Matrix Float -> Matrix Float -> Matrix Float
linearW' x dy =
  let trX  = compute $ transpose x
      prod = fromMaybe (error "linearW': Out of bounds") (trX |*| dy)
      m    = recip $ fromIntegral (rows x)
  in  m `scale` prod

linearX' :: Matrix Float -> Matrix Float -> Matrix Float
linearX' w dy = compute
  $ fromMaybe (error "linearX': Out of bounds") (dy `multiplyTransposed` w)

-- | Bias gradient
bias' :: Matrix Float -> Vector Float
bias' dY = compute $ m `_scale` _sumRows dY
  where m = recip $ fromIntegral $ rows dY

-- | Forward pass in a neural network:
-- exploit Haskell lazyness to never compute the
-- gradients.
forward :: NeuralNetwork Float -> Matrix Float -> Matrix Float
forward net dta = fst $ pass Eval net (dta, undefined)

softmax :: Matrix Float -> Matrix Float
softmax x =
  let x0 = compute $ expA (delay x) :: Matrix Float
      x1 = compute (_sumCols x0) :: Vector Float  -- Sumcols in this case!
      x2 = x1 `colsLike` x
  in  (compute $ delay x0 / x2)

-- | Both forward and backward neural network passes
pass
  :: Phase
  -- ^ `Train` or `Eval`
  -> NeuralNetwork Float
  -- ^ `NeuralNetwork` `Layer`s: weights and activations
  -> (Matrix Float, Matrix Float)
  -- ^ Mini-batch with labels
  -> (Matrix Float, [Gradients Float])
  -- ^ NN computation from forward pass and weights gradients
pass phase net (x, tgt) = (pred, grads)
 where
  (_, pred, grads) = _pass x net

  -- Computes a tuple of:
  -- 1) Gradients for further backward pass
  -- 2) NN prediction
  -- 3) Gradients of learnable parameters (where applicable)
  _pass inp [] = (loss', pred, [])
   where
      -- TODO: Make softmax/loss/loss gradient a part of SGD/Adam?
    pred  = softmax inp

    -- Gradient of cross-entropy loss
    -- after softmax activation.
    loss' = compute $ delay pred - delay tgt

  _pass inp (Linear w b : layers) = (dX, pred, LinearGradients dW dB : t)
   where
      -- Forward
    lin =
      compute
        $ delay (fromMaybe (error "lin1: Out of bounds") (inp |*| w))
        + (b `rowsLike` inp)

    (dZ, pred, t) = _pass lin layers

    -- Backward
    dW            = linearW' inp dZ
    dB            = bias' dZ
    dX            = linearX' w dZ

  _pass inp (Linear' w : layers) = (dX, pred, Linear'Gradients dW : t)
   where
      -- Forward
    lin = compute $ fromMaybe (error "lin2: Out of bounds") (inp |*| w)

    (dZ, pred, t) = _pass lin layers

    -- Backward
    dW            = linearW' inp dZ
    dX            = linearX' w dZ

  -- See also https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
  _pass inp (Batchnorm1d mu variance gamma beta : layers) =
    (dX, pred, BN1 batchMu batchVariance dGamma dBeta : t)
   where
      -- Forward
    eps = 1e-12
    b   = br (Sz1 $ rows inp)  -- Broadcast (replicate) rows from 1 to batch size
    m   = recip (fromIntegral $ rows inp)

    -- Step 1: mean
    batchMu :: Vector Float
    batchMu = compute $ m `_scale` _sumRows inp

    -- Step 2: mean subtraction
    xmu :: Matrix Float
    xmu = compute $ delay inp - b batchMu

    -- Step 3
    sq  = compute $ delay xmu ^ (2 :: Int)

    -- Step 4
    batchVariance :: Vector Float
    batchVariance = compute $ m `_scale` _sumRows sq

    -- Step 5
    sqrtvar       = sqrtA $ batchVariance `addC` eps

    -- Step 6
    ivar          = compute $ A.map recip sqrtvar

    -- Step 7
    xhat          = delay xmu * b ivar

    -- Step 8: rescale
    gammax        = b gamma * xhat

    -- Step 9: translate
    out0 :: Matrix Float
    out0 = compute $ gammax + b beta

    out :: Matrix Float
    out = if phase == Train
            -- Forward (Train phase)
      then out0
            -- Forward (Eval phase)
      else out2

    (dZ, pred, t) = _pass out layers

    -- Backward

    -- Step 9
    dBeta         = compute $ _sumRows dZ

    -- Step 8
    dGamma        = compute $ _sumRows (compute $ delay dZ * xhat)
    dxhat :: Matrix Float
    dxhat    = compute $ delay dZ * b gamma

    -- Step 7
    divar    = _sumRows $ compute $ delay dxhat * delay xmu
    dxmu1    = delay dxhat * b ivar

    -- Step 6
    dsqrtvar = A.map (negate . recip) (sqrtvar .^ 2) * divar

    -- Step 5
    dvar     = 0.5 `_scale` ivar * dsqrtvar

    -- Step 4
    dsq      = compute $ m `_scale` dvar

    -- Step 3
    dxmu2    = 2 `_scale` xmu * b dsq

    -- Step 2
    dx1      = compute $ dxmu1 + dxmu2
    dmu      = A.map negate $ _sumRows dx1

    -- Step 1
    dx2      = b $ compute (m `_scale` dmu)

    dX       = compute $ delay dx1 + dx2

    -- Alternatively use running stats during Eval phase:
    out1 :: Matrix Float
    out1 = compute $ (delay inp - b mu) / b
      (compute $ sqrtA $ variance `addC` eps)

    out2 = compute $ (b gamma * delay out1) + b beta

  _pass inp (Activation symbol : layers) = (dY, pred, NoGrad : t)
   where
    y             = getActivation symbol inp  -- Forward

    (dZ, pred, t) = _pass y layers

    dY            = getActivation' symbol inp dZ  -- Backward

-- | Broadcast a vector in Dim2
rowsLike
  :: Manifest r Ix1 Float
  => Array r Ix1 Float
  -> Matrix Float
  -> MatrixPrim D Float
rowsLike v m = br (Sz1 $ rows m) v

-- | Broadcast a vector in Dim1
colsLike
  :: Manifest r Ix1 Float
  => Array r Ix1 Float
  -> Matrix Float
  -> MatrixPrim D Float
colsLike v m = br1 (Sz1 $ cols m) v

-- | Broadcast by the given number of rows
br :: Manifest r Ix1 Float => Sz1 -> Array r Ix1 Float -> MatrixPrim D Float
br rows' v = expandWithin Dim2 rows' const v

-- | Broadcast by the given number of cols
br1 :: Manifest r Ix1 Float => Sz1 -> Array r Ix1 Float -> MatrixPrim D Float
br1 rows' v = expandWithin Dim1 rows' const v

-- | Stochastic gradient descent
sgd
  :: Monad m
  => Float
  -- ^ Learning rate
  -> Int
  -- ^ No of iterations
  -> NeuralNetwork Float
  -- ^ Neural network
  -> SerialT m (Matrix Float, Matrix Float)
  -- ^ Data stream
  -> m (NeuralNetwork Float)
sgd lr n net0 dataStream = iterN n epochStep net0
 where
  epochStep net = S.foldl' g net dataStream

  g
    :: NeuralNetwork Float
    -> (Matrix Float, Matrix Float)
    -> NeuralNetwork Float
  g net dta = let (_, dW) = pass Train net dta in zipWith f net dW

  f :: Layer Float -> Gradients Float -> Layer Float

  -- Update Linear layer weights
  f (Linear w b) (LinearGradients dW dB) = Linear
    (compute $ delay w - lr `_scale` dW)
    (compute $ delay b - lr `_scale` dB)

  f (Linear' w) (Linear'Gradients dW) =
    Linear' (compute $ delay w - lr `_scale` dW)

  -- Update batchnorm parameters
  f (Batchnorm1d mu v gamma beta) (BN1 mu' v' dGamma dBeta) = Batchnorm1d
    mu''
    v''
    gamma'
    beta'
   where
    alpha  = 0.1
    -- Running mean and variance
    mu''   = compute $ (alpha `_scale` mu') + ((1 - alpha) `_scale` mu)
    v''    = compute $ (alpha `_scale` v') + ((1 - alpha) `_scale` v)

    gamma' = compute $ delay gamma - (lr `_scale` dGamma)
    beta'  = compute $ delay beta - (lr `_scale` dBeta)

  -- No parameters to change
  f layer NoGrad = layer

  f _     _      = error "Layer/gradients mismatch"

-- | Strict left fold
iterN :: Monad m => Int -> (a -> m a) -> a -> m a
iterN n f x0 = foldM (\x _ -> f x) x0 [1 .. n]

addC :: (Num e, Source r ix e) => Array r ix e -> e -> Array D ix e
addC m c = A.map (c +) m

-- | Generate random weights and biases
genWeights :: (Int, Int) -> IO (Matrix Float, Vector Float)
genWeights (nin, nout) = do
  w <- setComp Par <$> _genWeights (nin, nout)
  b <- setComp Par <$> _genBiases nout
  return (w, b)
 where
  _genWeights (nin', nout') = scale k <$> randn sz
   where
    sz = Sz (nin' :. nout')
    k  = 0.01

  _genBiases n = randn (Sz n)

-- TODO: correct-by-construction
-- net <- genNetwork $ Sequential (I 2
--                                 :> Linear 128
--                                 :> Batchnorm1d
--                                 :> Activation Relu
--                                 :> O 1)

-- | Perform a binary classification
inferBinary :: NeuralNetwork Float -> Matrix Float -> Matrix Float
inferBinary net dta =
  let prediction = forward net dta
  -- Thresholding the NN output
  in  compute $ A.map (\a -> if a < 0.5 then 0 else 1) prediction

maxIndex :: (Ord a, Num b, Enum b) => [a] -> b
maxIndex xs = snd $ maximumBy (comparing fst) (zip xs [0 ..])

winnerTakesAll
  :: Matrix Float  -- ^ Mini-batch of vectors
  -> [Int]  -- ^ List of maximal indices
winnerTakesAll m = map maxIndex xs where xs = toLists2 m

errors :: Eq lab => [(lab, lab)] -> [(lab, lab)]
errors = filter (uncurry (/=))
{-# SPECIALIZE errors :: [(Int, Int)] -> [(Int, Int)] #-}

accuracy :: (Eq a, Fractional acc) => [a] -> [a] -> acc
accuracy tgt pr = 100 * r
 where
  errNo = length $ errors (zip tgt pr)
  r     = 1 - fromIntegral errNo / fromIntegral (length tgt)
{-# SPECIALIZE accuracy :: [Int] -> [Int] -> Float #-}

_accuracy :: NeuralNetwork Float -> (Matrix Float, Matrix Float) -> Float
-- NB: better avoid double conversion to and from one-hot-encoding
_accuracy net (batch, labelsOneHot) =
  let batchResults = winnerTakesAll $ forward net batch
      expected     = winnerTakesAll labelsOneHot
  in  accuracy expected batchResults

avgAccuracy
  :: Monad m
  => NeuralNetwork Float
  -> SerialT m (Matrix Float, Matrix Float)
  -> m Float
avgAccuracy net stream = s // len
 where
  results = S.map (_accuracy net) stream
  s       = S.sum results
  len     = fromIntegral <$> S.length results
  (//)    = liftA2 (/)

-- | Average elements in each column
mean :: Matrix Float -> Vector Float
mean ar = compute $ m `_scale` _sumRows ar
  where m = recip $ fromIntegral (rows ar)

-- | Variance over each column
var :: Matrix Float -> Vector Float
var ar = compute $ m `_scale` r
 where
  mu    = br (Sz1 nRows) $ mean ar
  nRows = rows ar
  r0    = compute $ (delay ar - mu) .^ 2
  r     = _sumRows r0
  m     = recip $ fromIntegral nRows

-- | Sum values in each column and produce a delayed 1D Array
_sumRows :: Matrix Float -> Array D Ix1 Float
_sumRows = A.foldlWithin Dim2 (+) 0.0

-- | Sum values in each row and produce a delayed 1D Array
_sumCols :: Matrix Float -> Array D Ix1 Float
_sumCols = A.foldlWithin Dim1 (+) 0.0

-- TODO: another demo where only the forward pass is defined.
-- Then, use `backprop` package for automatic differentiation.
