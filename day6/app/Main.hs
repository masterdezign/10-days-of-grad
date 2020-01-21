{-# LANGUAGE FlexibleContexts #-}

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

loadMNIST :: FilePath -> FilePath -> IO (Maybe [(Matrix Float, Matrix Float)])
loadMNIST fpI fpL = runMaybeT $ do
  i <- MaybeT $ decodeIDXFile fpI
  l <- MaybeT $ decodeIDXLabelsFile fpL
  d <- MaybeT . return $ force $ labeledIntData l i
  return $ map _conv d
 where
  _conv :: (Int, V.Vector Int) -> (Matrix Float, Matrix Float)
  _conv (label, v) = (v1, toOneHot10 label)
   where
    v0 = V.map ((`subtract` 0.5) . (/ 255) . fromIntegral) v
    v1 = A.fromVector' Par (Sz2 1 784) v0

toOneHot10 :: Int -> Matrix Float
toOneHot10 n =
  A.makeArrayR U Par (Sz2 1 10) (\(_ :. j) -> if j == n then 1 else 0)

mnistStream
  :: Int -> FilePath -> FilePath -> IO (SerialT IO (Matrix Float, Matrix Float))
mnistStream batchSize fpI fpL = do
  Just dta <- loadMNIST fpI fpL
  dta2     <- shuffleIO dta

  -- Split data into batches
  let (vs, labs) = unzip dta2
      merge :: [Matrix Float] -> Matrix Float
      merge = A.compute . A.concat' 2
      vs'   = map merge $ chunksOf batchSize vs
      labs' = map merge $ chunksOf batchSize labs
      dta'  = zip vs' labs'
  return $ S.fromList dta'

data TrainSettings = TrainSettings
  { _printEpochs :: Int  -- Print every N epochs
  , _lr :: Float  -- Learning rate
  , _totalEpochs :: Int  -- Number of training epochs
  }

train
  :: TrainSettings
  -> NeuralNetwork Float
  -> ( SerialT IO (Matrix Float, Matrix Float)
     , SerialT IO (Matrix Float, Matrix Float)
     )
  -> IO (NeuralNetwork Float)
train TrainSettings { _printEpochs = printEpochs, _lr = lr, _totalEpochs = totalEpochs } net (trainS, testS)
  = do
    (net', _) <- iterN
      (totalEpochs `div` printEpochs)
      (\(net0, j) -> do
        net1 <- sgd lr printEpochs net0 trainS

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
  -- NB batch size (!)
  trainS <- mnistStream 100
                        "data/train-images-idx3-ubyte"
                        "data/train-labels-idx1-ubyte"
  testS <- mnistStream 1000
                       "data/t10k-images-idx3-ubyte"
                       "data/t10k-labels-idx1-ubyte"

  -- NB use more binary neurons compared to a normal network (!)
  let infl = 4

  let [i, h1, h2, o] = [784, infl * 300, infl * 50, 10]
  (w1, _) <- genWeights (i, h1)
  let ones n = A.replicate Par (Sz1 n) 1 :: Vector Float
      zeros n = A.replicate Par (Sz1 n) 0 :: Vector Float
  (w2, _) <- genWeights (h1, h2)
  (w3, _) <- genWeights (h2, o)

  -- NB: BinarizedLinear has only weights, no biases.
  -- The reason is that Batchnorm1d layer has trainable
  -- parameter beta performing similar transformation.
  let net =
        [ BinarizedLinear w1
        , Batchnorm1d (zeros h1) (ones h1) (ones h1) (zeros h1)
        , Activation Sign

        , BinarizedLinear w2
        , Batchnorm1d (zeros h2) (ones h2) (ones h2) (zeros h2)
        , Activation Sign

        , BinarizedLinear w3
        -- NB this batchnorm (!)
        , Batchnorm1d (zeros o) (ones o) (ones o) (zeros o)
        ]

  putStrLn "Training a binarized network"
  net' <- train
    TrainSettings { _printEpochs = 1, _lr = 0.1, _totalEpochs = 30 }
    net
    (trainS, testS)

  return ()
