{-# LANGUAGE FlexibleContexts #-}

-- | = Direct feedback alignment

import           Data.Massiv.Array       hiding ( map
                                                , zip
                                                , unzip
                                                )
import qualified Data.Massiv.Array             as A
import qualified Data.Massiv.Array.Manifest.Vector
                                               as A
import           Streamly
import qualified Streamly.Prelude              as S
import           Text.Printf                    ( printf )
import           Control.DeepSeq                ( force )
import           Control.Monad.Trans.Maybe
import           Data.IDX
import qualified Data.Vector.Unboxed           as V
import           Data.List.Split                ( chunksOf )

import           NeuralNetwork
import           Shuffle                        ( shuffleIO )

loadMNIST :: FilePath -> FilePath -> IO (Maybe [(Matrix U Float, Matrix U Float)])
loadMNIST fpI fpL = runMaybeT $ do
  i <- MaybeT $ decodeIDXFile fpI
  l <- MaybeT $ decodeIDXLabelsFile fpL
  d <- MaybeT . return $ force $ labeledIntData l i
  return $ map _conv d
 where
  _conv :: (Int, V.Vector Int) -> (Matrix U Float, Matrix U Float)
  _conv (label, v) = (v1, toOneHot10 label)
   where
    v0 = V.map ((`subtract` 0.5) . (/ 255) . fromIntegral) v
    v1 = A.fromVector' Par (Sz2 1 784) v0

toOneHot10 :: Int -> Matrix U Float
toOneHot10 n =
  A.makeArrayR U Par (Sz2 1 10) (\(_ :. j) -> if j == n then 1 else 0)

mnistStream
  :: Int -> FilePath -> FilePath -> IO (SerialT IO (Matrix U Float, Matrix U Float))
mnistStream batchSize fpI fpL = do
  Just dta <- loadMNIST fpI fpL
  dta2     <- shuffleIO dta

  -- Split data into batches
  let (vs, labs) = unzip dta2
      merge :: [Matrix U Float] -> Matrix U Float
      merge = A.compute . A.concat' 2
      vs'   = map merge $ chunksOf batchSize vs
      labs' = map merge $ chunksOf batchSize labs
      dta'  = zip vs' labs'
  return $ S.fromList dta'

data TrainSettings = TrainSettings
  { _printEpochs :: Int  -- Print every N epochs
  , _totalEpochs :: Int  -- Number of training epochs
  }

train
  :: TrainSettings
  -> NeuralNetwork Float
  -> ( SerialT IO (Matrix U Float, Matrix U Float)
     , SerialT IO (Matrix U Float, Matrix U Float)
     )
  -> IO (NeuralNetwork Float)
train TrainSettings { _printEpochs = printEpochs, _totalEpochs = totalEpochs } net (trainS, testS)
  = do
    (net', _) <- iterN
      (totalEpochs `div` printEpochs)
      (\(net0, j) -> do
        net1 <- adam adamParams printEpochs net0 trainS
        -- net1 <- sgd 0.005 printEpochs net0 trainS

        tacc <- net1 `avgAccuracy` trainS :: IO Float
        putStr $ printf "%d Training accuracy %.1f" (j :: Int) tacc

        acc <- net1 `avgAccuracy` testS :: IO Float
        putStrLn $ printf "  Validation accuracy %.1f" acc

        return (net1, j + printEpochs)
      )
      (net, 1)
    return net'

main :: IO ()
main = do
  trainS <- mnistStream 1000
                        "data/train-images-idx3-ubyte"
                        "data/train-labels-idx1-ubyte"
  testS <- mnistStream 1000
                       "data/t10k-images-idx3-ubyte"
                       "data/t10k-labels-idx1-ubyte"

  let [i, h1, h2, o] = [784, 300, 50, 10]
  (w1, b1) <- genWeights (i, h1)
  (w2, b2) <- genWeights (h1, h2)
  (w3, b3) <- genWeights (h2, o)

  -- NB: Generate fixed random matrices
  let rng = (-0.1, 0.1)
  ww1 <- genNormalized rng (o, h1)
  ww2 <- genNormalized rng (o, h2)

  -- The final layer receives the loss directly
  let ww3 = compute (identityMatrix (Sz1 o))

  -- NB We assume that in DFA network there are only LinearDFA layers
  -- so that the loss is available to all layers,
  -- instead of the gradients
  let net =
        [ LinearDFA Relu w1 b1 ww1
        , LinearDFA Relu w2 b2 ww2
        , LinearDFA Id w3 b3 ww3
        ]

  putStrLn "Direct feedback alignment"
  net' <- train
    TrainSettings { _printEpochs = 1, _totalEpochs = 30 }
    net
    (trainS, testS)

  return ()

genNormalized :: (Float, Float) -> (Int, Int) -> IO (Matrix U Float)
genNormalized rng (o, h) = do
  -- Draw from the uniform distribution
  ww <- rand rng (Sz2 o h)
  let m = 1.0 / fromIntegral h
  -- Normalize the matrix by the number of outputs
  return (m `scale` ww)
