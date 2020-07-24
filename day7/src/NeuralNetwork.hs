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

-- Neural network layers: Linear, Activation
data Layer a = Linear (Matrix a) (Vector a)
               -- Similar to Linear, but having a random matrix for
               -- direct feedback alignment as the last argument
               | LinearDFA FActivation (Matrix a) (Matrix a)
               | Activation FActivation

type NeuralNetwork a = [Layer a]

data Gradients a = -- Weight and bias gradients
                   LinearGradients (Matrix a) (Vector a)
                   | DFA (Matrix a)
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
      prod = fromMaybe (error ("linearW': shape mismatch " ++ show (size trX, size dy))) (trX |*| dy)
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
forward net dta =
  let (_, predic, _) = pass Eval net (dta, undefined)
   in predic

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
  -> (Matrix Float, Matrix Float, [Gradients Float])
  -- ^ NN computation from forward pass and weights gradients
pass _ net (x, tgt) = _pass x net
 where
  -- Computes a tuple of:
  -- 1) Gradients for further backward pass
  -- 2) NN prediciction
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

  _pass inp (LinearDFA fact w ww : layers) = (loss1, predic, DFA df : t)
   where
    -- Forward
    lin = compute $ fromMaybe (error "DFA1: Out of bounds") (inp |*| w)
    y = getActivation fact lin

    (loss1, predic, t) = _pass y layers

    -- Direct feedback
    df = linearW' inp dY1
      where
        dY = getActivation' fact lin loss1  -- Gradient of the activation
        dY1 = compute $ fromMaybe (error "DFA2: Out of bounds") (dY |*| ww)

  _pass inp (Activation symbol : layers) = (dY, predic, NoGrad : t)
   where
    y             = getActivation symbol inp  -- Forward

    (dZ, predic, t) = _pass y layers

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

-- | Stochastic gradient descend
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
  g net dta = let (_, _, dW) = pass Train net dta in zipWith f net dW

  f :: Layer Float -> Gradients Float -> Layer Float

  -- Update Linear layer weights
  f (Linear w b) (LinearGradients dW dB) = Linear
    (compute $ delay w - lr `_scale` dW)
    (compute $ delay b - lr `_scale` dB)

  f (LinearDFA fact w ww) (DFA dW) = (LinearDFA fact w1 ww)
    where
      w1 = compute $ delay w - lr `_scale` dW

  -- No parameters to change
  f layer NoGrad = layer

  f _     _      = error "Layer/gradients mismatch"

-- | Strict left fold
iterN :: Monad m => Int -> (a -> m a) -> a -> m a
iterN n f x0 = foldM (\x _ -> f x) x0 [1 .. n]

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

  -- Zero biases for simplicity
  _genBiases n = return $ A.replicate Par (Sz n) 0
  -- _genBiases n = randn (Sz n)

-- | Perform a binary classification
inferBinary :: NeuralNetwork Float -> Matrix Float -> Matrix Float
inferBinary net dta =
  let prediciction = forward net dta
  -- Thresholding the NN output
  in  compute $ A.map (\a -> if a < 0.5 then 0 else 1) prediciction

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
