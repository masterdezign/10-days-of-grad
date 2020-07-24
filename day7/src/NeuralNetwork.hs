-- |= Neural Network Building Blocks
--
-- The idea of this module is to manage gradients manually.
-- That is done intentionally to illustrate neural
-- networks training.

{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module NeuralNetwork
  ( NeuralNetwork
  , Layer(..)
  , FActivation(..)
  , sigmoid
  , sigmoid'
  , genWeights
  , forward

  -- * Training
  , sgd
  , adam
  , AdamParameters (..)
  , adamParams

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
  , scale
  , iterN
  , mean
  , var
  , br
  )
where

import           Control.Monad                  ( foldM
                                                )
import           Control.Applicative            ( liftA2 )
import qualified System.Random.MWC as MWC
import           System.Random.MWC              ( createSystemRandom )
import           System.Random.MWC.Distributions
                                                ( standard )
import           Data.List                      ( maximumBy )
import           Data.Ord
import           Data.Massiv.Array       hiding ( map
                                                , zip
                                                , zipWith
                                                , zipWith3
                                                )
import qualified Data.Massiv.Array             as A
import           Streamly
import qualified Streamly.Prelude              as S
import           Data.Maybe                     ( fromMaybe )

-- Activation function symbols:
-- * Rectified linear unit (ReLU)
-- * Sigmoid
-- * Identity (no activation)
data FActivation = Relu | Sigmoid | Id

-- Neural network layers: Linear, Activation
data Layer a = Linear (Matrix U a) (Vector U a)
               -- Similar to Linear, but having a random matrix for
               -- direct feedback alignment as the last argument
               | LinearDFA FActivation (Matrix U a) (Vector U a) (Matrix U a)
               | Activation FActivation

type NeuralNetwork a = [Layer a]

data Gradients a = -- Weight and bias gradients
                   LinearGradients (Matrix U a) (Vector U a)
                   | DFA (Matrix U a) (Vector U a)
                   | NoGrad  -- No learnable parameters

-- | A neural network may work differently in training and evaluation modes
data Phase = Train | Eval deriving (Show, Eq)

-- | Lookup activation function by a symbol
getActivation :: FActivation -> (Matrix U Float -> Matrix U Float)
getActivation Id      = id
getActivation Sigmoid = sigmoid
getActivation Relu    = relu

-- | Lookup activation function derivative by a symbol
getActivation' :: FActivation -> (Matrix U Float -> Matrix U Float -> Matrix U Float)
getActivation' Id      = flip const
getActivation' Sigmoid = sigmoid'
getActivation' Relu    = relu'

-- | Elementwise sigmoid computation
sigmoid :: Matrix U Float -> Matrix U Float
sigmoid = computeMap f where f x = recip $ 1.0 + exp (-x)

-- | Compute sigmoid gradients
sigmoid' :: Matrix U Float -> Matrix U Float -> Matrix U Float
sigmoid' x dY =
  let sz   = size x
      ones = compute $ A.replicate Par sz 1.0 :: Matrix U Float
      y    = sigmoid x
  in  compute $ delay dY * delay y * (delay ones - delay y)

relu :: Matrix U Float -> Matrix U Float
relu = computeMap f where f x = if x < 0 then 0 else x

relu' :: Matrix U Float -> Matrix U Float -> Matrix U Float
relu' x = compute . A.zipWith f x where f x0 dy0 = if x0 <= 0 then 0 else dy0

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

rows :: Matrix U Float -> Int
rows m = let (r :. _) = unSz $ size m in r

cols :: Matrix U Float -> Int
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

linearW' :: Matrix U Float -> Matrix U Float -> Matrix U Float
linearW' x dy =
  let trX  = compute $ transpose x
      prod = fromMaybe (error ("linearW': shape mismatch " ++ show (size trX, size dy))) (trX |*| dy)
      m    = recip $ fromIntegral (rows x)
  in  m `scale` prod

linearX' :: Matrix U Float -> Matrix U Float -> Matrix U Float
linearX' w dy = compute
  $ fromMaybe (error "linearX': Out of bounds") (dy `multiplyTransposed` w)

-- | Bias gradient
bias' :: Matrix U Float -> Vector U Float
bias' dY = compute $ m `_scale` _sumRows dY
  where m = recip $ fromIntegral $ rows dY

-- | Forward pass in a neural network:
-- exploit Haskell lazyness to never compute the
-- gradients.
forward :: NeuralNetwork Float -> Matrix U Float -> Matrix U Float
forward net dta =
  let (_, predic, _) = pass Eval net (dta, undefined)
   in predic

softmax :: Matrix U Float -> Matrix U Float
softmax x_ =
  let x' = delay x_
      x = x' `addC` (-A.maximum' x')
      x0 = compute $ expA x :: Matrix U Float
      x1 = compute (_sumCols x0) :: Vector U Float  -- Sumcols in this case!
      x2 = x1 `colsLike` x_
  in  (compute $ delay x0 / x2)

-- | Both forward and backward neural network passes
pass
  :: Phase
  -- ^ `Train` or `Eval`
  -> NeuralNetwork Float
  -- ^ `NeuralNetwork` `Layer`s: weights and activations
  -> (Matrix U Float, Matrix U Float)
  -- ^ Mini-batch with labels
  -> (Matrix U Float, Matrix U Float, [Gradients Float])
  -- ^ NN computation from forward pass and weights gradients
pass _ net (x, tgt) = _pass x net
 where
  -- Computes a tuple of:
  -- 1) Gradients for further backward pass
  -- 2) NN prediction
  -- 3) Gradients of learnable parameters (where applicable)
  _pass inp [] = (loss1, predic, [])
   where
    predic  = softmax inp

    -- Gradient of cross-entropy loss
    -- after softmax activation.
    loss1 = compute $ delay predic - delay tgt

  _pass inp (Linear w b : layers) = (dX, predic, LinearGradients dW dB : t)
   where
      -- Forward
    lin =
      compute
        $ delay (fromMaybe (error "lin1: Out of bounds") (inp |*| w))
        + (b `rowsLike` inp)

    (dZ, predic, t) = _pass lin layers

    -- Backward
    dW            = linearW' inp dZ
    dB            = bias' dZ
    dX            = linearX' w dZ

  _pass inp (LinearDFA fact w b ww : layers) = (loss1, predic, DFA dW dB : t)
   where
    -- Forward
    lin =
      compute
        $ delay (fromMaybe (error "DFA1: Out of bounds") (inp |*| w))
        + (b `rowsLike` inp)
    y = getActivation fact lin

    (loss1, predic, t) = _pass y layers

    -- Direct feedback
    dY = getActivation' fact lin loss1  -- Gradient of the activation
    dY1 = compute $ fromMaybe (error "DFA2: Out of bounds") (dY |*| ww)

    dW = linearW' inp dY1
    dB = bias' dY1

  _pass inp (Activation symbol : layers) = (dY, predic, NoGrad : t)
   where
    y             = getActivation symbol inp  -- Forward

    (dZ, predic, t) = _pass y layers

    dY            = getActivation' symbol inp dZ  -- Backward

-- | Broadcast a vector in Dim2
rowsLike
  :: Manifest r Ix1 Float
  => Array r Ix1 Float
  -> Matrix U Float
  -> Matrix D Float
rowsLike v m = br (Sz1 $ rows m) v

-- | Broadcast a vector in Dim1
colsLike
  :: Manifest r Ix1 Float
  => Array r Ix1 Float
  -> Matrix U Float
  -> Matrix D Float
colsLike v m = br1 (Sz1 $ cols m) v

-- | Broadcast by the given number of rows
br :: Manifest r Ix1 Float => Sz1 -> Array r Ix1 Float -> Matrix D Float
br rows' = expandWithin Dim2 rows' const

-- | Broadcast by the given number of cols
br1 :: Manifest r Ix1 Float => Sz1 -> Array r Ix1 Float -> Matrix D Float
br1 rows' = expandWithin Dim1 rows' const

-- | Stochastic gradient descend
sgd
  :: Monad m
  => Float
  -- ^ Learning rate
  -> Int
  -- ^ No of iterations
  -> NeuralNetwork Float
  -- ^ Neural network
  -> SerialT m (Matrix U Float, Matrix U Float)
  -- ^ Data stream
  -> m (NeuralNetwork Float)
sgd lr n net0 dataStream = iterN n epochStep net0
 where
  epochStep net = S.foldl' g net dataStream

  g
    :: NeuralNetwork Float
    -> (Matrix U Float, Matrix U Float)
    -> NeuralNetwork Float
  g net dta = let (_, _, dW) = pass Train net dta in zipWith f net dW

  f :: Layer Float -> Gradients Float -> Layer Float

  -- Update Linear layer weights
  f (Linear w b) (LinearGradients dW dB) = Linear
    (compute $ delay w - lr `_scale` dW)
    (compute $ delay b - lr `_scale` dB)

  f (LinearDFA fact w b ww) (DFA dW dB) = LinearDFA fact w1 b1 ww
    where
      w1 = compute $ delay w - lr `_scale` dW
      b1 = compute $ delay b - lr `_scale` dB

  -- No parameters to change
  f layer NoGrad = layer

  f _     _      = error "Layer/gradients mismatch"

data AdamParameters = AdamParameters { _beta1 :: Float
                                     , _beta2 :: Float
                                     , _epsilon :: Float
                                     , _lr :: Float
                                     }

-- | Adam optimizer parameters
adamParams :: AdamParameters
adamParams = AdamParameters { _beta1 = 0.9
                            , _beta2 = 0.999
                            , _epsilon = 1e-8
                            , _lr = 0.001  -- ^ Learning rate
                            }

type AdamP = [(Matrix U Float, Vector U Float)]

-- | Adam optimization
adam
  :: Monad m
  => AdamParameters
  -- ^ Adam parameters
  -> Int
  -- ^ No of iterations
  -> NeuralNetwork Float
  -- ^ Neural network
  -> SerialT m (Matrix U Float, Matrix U Float)
  -- ^ Data stream
  -> m (NeuralNetwork Float)
adam AdamParameters { _lr = lr
                    , _beta1 = beta1
                    , _beta2 = beta2
                    , _epsilon = epsilon
     } n net0 dataStream = do
  (net, _, _) <- iterN n epochStep (net0, s0, v0)
  return net
 where
  epochStep w = S.foldl' g w dataStream

  s0 :: AdamP
  s0 = map zf net0
  v0 :: AdamP
  v0 = map zf net0

  zf (LinearDFA _ w b _) = (zerosLike w, zerosLike b)
  zf _ = error "To be implemented; update also s0 and v0 types"
  zerosLike x = compute $ A.replicate Par (size x) 0.0

  g
    :: (NeuralNetwork Float, AdamP, AdamP)
    -> (Matrix U Float, Matrix U Float)
    -> (NeuralNetwork Float, AdamP, AdamP)
  g (w, s, v) dta = (wN, sN, vN)
    where
      (_, _, dW) = pass Train w dta

      wN = zipWith3 f w vN sN
      sN = zipWith f2 s dW
      vN = zipWith f3 v dW

  f :: Layer Float
    -> (Matrix U Float, Vector U Float)
    -> (Matrix U Float, Vector U Float)
    -> Layer Float
  f (LinearDFA s w_ b_ ww) (vW, vB) (sW, sB) = LinearDFA s w1 b1 ww
    where
      w1 = compute $ delay w_ - lr `_scale` vW / ((sqrtA $ delay sW) `addC` epsilon)
      b1 = compute $ delay b_ - lr `_scale` vB / ((sqrtA $ delay sB) `addC` epsilon)
  f _ _ _ = error "To be implemented"

  f2 :: (Matrix U Float, Vector U Float)
      -> Gradients Float
      -> (Matrix U Float, Vector U Float)
  f2 (sW, sB) (DFA dW dB) =
    ( compute $ (beta2 `_scale` delay sW) + (1 - beta2) `_scale` (delay dW^2)
    , compute $ (beta2 `_scale` delay sB) + (1 - beta2) `_scale` (delay dB^2))
  f2 _ _ = error "To be implemented"

  f3 :: (Matrix U Float, Vector U Float)
      -> Gradients Float
      -> (Matrix U Float, Vector U Float)
  f3 (vW, vB) (DFA dW dB) =
    ( compute $ beta1 `_scale` delay vW + (1 - beta1) `_scale` delay dW
    , compute $ beta1 `_scale` delay vB + (1 - beta1) `_scale` delay dB)
  f3 _ _ = error "To be implemented"

addC :: (Num e, Source r ix e) => Array r ix e -> e -> Array D ix e
addC m c = A.map (c +) m

-- | Strict left fold
iterN :: Monad m => Int -> (a -> m a) -> a -> m a
iterN n f x0 = foldM (\x _ -> f x) x0 [1 .. n]

-- | Generate random weights and biases
genWeights :: (Int, Int) -> IO (Matrix U Float, Vector U Float)
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

-- | Perform a binary classification
inferBinary :: NeuralNetwork Float -> Matrix U Float -> Matrix U Float
inferBinary net dta =
  let prediction = forward net dta
  -- Thresholding the NN output
  in  compute $ A.map (\a -> if a < 0.5 then 0 else 1) prediction

maxIndex :: (Ord a, Num b, Enum b) => [a] -> b
maxIndex xs = snd $ maximumBy (comparing fst) (zip xs [0 ..])

winnerTakesAll
  :: Matrix U Float  -- ^ Mini-batch of vectors
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

_accuracy :: NeuralNetwork Float -> (Matrix U Float, Matrix U Float) -> Float
-- NB: better avoid double conversion to and from one-hot-encoding
_accuracy net (batch, labelsOneHot) =
  let batchResults = winnerTakesAll $ forward net batch
      expected     = winnerTakesAll labelsOneHot
  in  accuracy expected batchResults

avgAccuracy
  :: Monad m
  => NeuralNetwork Float
  -> SerialT m (Matrix U Float, Matrix U Float)
  -> m Float
avgAccuracy net stream = s // len
 where
  results = S.map (_accuracy net) stream
  s       = S.sum results
  len     = fromIntegral <$> S.length results
  (//)    = liftA2 (/)

-- | Average elements in each column
mean :: Matrix U Float -> Vector U Float
mean ar = compute $ m `_scale` _sumRows ar
  where m = recip $ fromIntegral (rows ar)

-- | Variance over each column
var :: Matrix U Float -> Vector U Float
var ar = compute $ m `_scale` r
 where
  mu    = br (Sz1 nRows) $ mean ar
  nRows = rows ar
  r0    = compute $ (delay ar - mu) .^ 2
  r     = _sumRows r0
  m     = recip $ fromIntegral nRows

-- | Sum values in each column and produce a delayed 1D Array
_sumRows :: Matrix U Float -> Array D Ix1 Float
_sumRows = A.foldlWithin Dim2 (+) 0.0

-- | Sum values in each row and produce a delayed 1D Array
_sumCols :: Matrix U Float -> Array D Ix1 Float
_sumCols = A.foldlWithin Dim1 (+) 0.0
